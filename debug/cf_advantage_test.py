import numpy as np
import matplotlib.pyplot as plt


def my_softmax(logits: np.ndarray):
  logits_shifted = logits - np.max(logits)
  exp_logits = np.exp(logits_shifted)
  normalization = np.sum(exp_logits)

  p = exp_logits / (normalization + (normalization == 0))
  return p



def get_cf_advantages(matrix: np.ndarray, joint_action: np.ndarray, baselines: np.ndarray):
  p1_util = matrix[*joint_action]

  p1_cf_advantage = p1_util - matrix[baselines[0], joint_action[1]]
  p2_cf_advantage = matrix[joint_action[0], baselines[1]] - p1_util
  #breakpoint()
  return p1_cf_advantage, p2_cf_advantage

def sample_actions(policies: list[np.ndarray], rng_gen : np.random.Generator):
  joint_action = []
  for p in policies:
    num_actions = p.shape[0]
    action = rng_gen.choice(num_actions, p=p)
    joint_action.append(action)
  return np.asarray(joint_action)

def ppo_update(matrix: np.ndarray, rng_gen: np.random.Generator, old_pols: list, old_logits: list,  
               baselines: np.ndarray, epsilon: float, num_updates: int, lr: float):
 
  logits = [l for l in old_logits]
  joint_action = sample_actions(old_pols, rng_gen)
  advantages = get_cf_advantages(matrix, joint_action, baselines)
  for i in range(num_updates):
    new_logits = []
    new_pols = [my_softmax(l) for l in logits]
    for l, old_p, a, p, adv in zip(logits, old_pols, joint_action, new_pols, advantages):
      ratio = p[a] / old_p[a]
      signal = ratio * adv
      inv_prob = 1 / old_p[a]
      clipped_ratio = np.clip(ratio, 1 - epsilon, 1 + epsilon)
      clipped_signal = clipped_ratio * adv
      # print(f"Ratio {ratio}")
      # print(f"Clipped ratio {ratio}")
      # print(f"Signal: {signal}")
      # print(f"Clipped signal {clipped_signal}")
      # We take the minimum, if the minimum would be the clipped signal,
      # we get zero gradient and copy the previous logit
      if signal > clipped_signal:
        new_logits.append(l)
        continue
      num_actions = p.shape[0]
      #Derivation of softmax: probs * (indication - prob)
      #Multiplied by the 1 / old_p[a] * advantage
      softmax_delta = p * (np.eye(num_actions)[a] - p)
      softmax_delta = signal * softmax_delta
      #softmax_delta = inv_prob * adv * softmax_delta
      new_logit = l + lr * softmax_delta
      new_logits.append(new_logit)
    logits = new_logits
  return logits, advantages

  


def cf_advantage_dynamics(matrix: np.ndarray, seed: int = 42, num_iters: int = 1000, 
                          baselines: list = [0, 0], epsilon: float = 0.05, grad_steps_per_step:int = 10,
                          lr: float = 0.1):
  R = matrix.shape[0]
  C = matrix.shape[1]
  logits = [np.ones(R), np.ones(C)]
  logits[0][1] = 5
  logits[1][1] = 5
  baselines = np.asarray(baselines)
  rng_gen = np.random.default_rng(seed)
  first_action_probs = []
  advantages = []
  for i in range(num_iters):
    policies = [my_softmax(l) for l in logits]
    first_action_probs.append([policies[0][0], policies[1][0]])
    logits, adv = ppo_update(matrix, rng_gen, policies, logits, baselines, epsilon, grad_steps_per_step, lr)
    advantages.append(adv)
  policies = [my_softmax(l) for l in logits]
  steps = np.arange(num_iters)
  first_action_probs = np.asarray(first_action_probs)
  advantages = np.asarray(advantages)
  #breakpoint()
  plt.plot(steps, first_action_probs[:, 0], label="Player 1")
  plt.plot(steps, first_action_probs[:, 1], label="Player 2")
  plt.xlabel("Number of steps")
  plt.ylabel("First action probability")
  plt.title("Matching pennies policies.")
  plt.legend()
  plt.savefig("matching_pennies.png")
  plt.close()  
  return policies



def main():
  matching_pennies = np.asarray([[1, -1], [-1, 1]])
  found_policies = cf_advantage_dynamics(matching_pennies, num_iters=800, grad_steps_per_step=1, lr=1)
  print(f"Found policies for matching pennies {found_policies}")


if __name__ == "__main__":
  main()