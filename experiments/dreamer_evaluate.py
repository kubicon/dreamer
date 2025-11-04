
from argparse import ArgumentParser
import os
import numpy as np
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp

from dreamer import Dreamer
from experiments.eval_utils import model_walk_test, cartesian_product, WalkCarry
from train_utils import load_model


parser = ArgumentParser()
parser.add_argument("--model_dir", type=str, default="trained_networks/dreamer/point_card_matching_3/seed99/network_seed42", help="Path to the directory of saved models")
parser.add_argument("--restore_step", type=int, default=-1, help="Saved step of the model to restore. If -1 check all saved models within that folder")

parser.add_argument("--seed", type=int, default=-1, help="Seed for the key to be used in gameplay. -1 for a random seed.")

parser.add_argument("--verbose", action="store_true", help="A flag whether to also print information about states being checked")

parser.add_argument("--render_tree", action="store_true", help="A flag whether to create the model EFG-style tree and render it.")



def check_state_all_outcomes(model: Dreamer, carry: WalkCarry, eps: float, outcome_threshold: float = 0.1, verbose=False):
  """Checks a state whether all the possible outcomes produce valid output."""
  mistake_probs = np.zeros(3)
  stoch_state = np.asarray(carry.stoch_state)
  real_obs = model.game.get_info(carry.game_state)[1]
  #print(f"Checking state {carry.game_state}, all outcomes")
  #TODO: Could that be done more efficiently without the loop over classes?
  num_classes = stoch_state.shape[0]
  deter_states = (stoch_state >= outcome_threshold).astype(int)
  class_indices, category_indices = np.nonzero(deter_states)
  per_class_valids = []
  for i in range(num_classes):
    single_class_indices = category_indices[class_indices == i]
    per_class_valids.append(single_class_indices)

  combinations = cartesian_product(*per_class_valids)
  #print(f"Num outcomes: {len(combinations)}")
  for comb in combinations:
    sampled_deter = jax.nn.one_hot(comb, stoch_state.shape[-1])
    decoded_obs = model.get_decoder(model.optimizers.decoder_optimizer.model, carry.hidden_state, sampled_deter)
    probs = [stoch_state[i, comb_part] for i, comb_part in enumerate(comb)]
    joint_probs = np.prod(probs)
    pred_reward, pred_terminal = model.get_reward_and_terminal(model.optimizers.predictor_optimizer.model, carry.hidden_state, sampled_deter)
    max_dif = jnp.max(jnp.abs(real_obs - decoded_obs))
    if max_dif >= eps:
      mistake_probs[0] += joint_probs
      #print(f"Real obs and decoded obs differ by more than {eps} for outcome {comb} with probabilties {probs}")
      #print(f"Max difference: {max_dif}")
      # print(f"Real obs {real_obs}")
      # print(f"Decoded obs {decoded_obs}")
    if pred_terminal != carry.terminal:
      mistake_probs[1] += joint_probs
      #print(f"Predicted terminal {pred_terminal} does not match real terminal for outcome {comb} with probabilties {probs} {carry.terminal}")
    if jnp.abs(carry.reward - pred_reward) >= eps:
      mistake_probs[2] += joint_probs
      #print(f"Predicted reward {pred_reward} differs from real reward {carry.reward} for outcome {comb} with probabilties {probs} by more than {eps}")
  #Ordered as observation, terminal, reward
  return mistake_probs
  #breakpoint()

def check_state_one_outcome(model: Dreamer, carry: WalkCarry,  eps: float, verbose=False):
  """Check whether the given sampled deterministic state
  produces valid output. Used for checking one particular chance outcome"""
  mistake_probs = np.zeros(3)
  differences = np.zeros(3)
  real_obs = model.game.get_info(carry.game_state)[1]
  #print(f"Checking state {carry.game_state}, closest outcome")
  decoded_obs = model.get_decoder(model.optimizers.decoder_optimizer.model, carry.hidden_state, carry.deter_state)
  pred_reward, pred_terminal = model.get_reward_and_terminal(model.optimizers.predictor_optimizer.model, carry.hidden_state, carry.deter_state)
  max_dif = jnp.max(jnp.abs(real_obs - decoded_obs))
  reward_dif = jnp.abs(carry.reward - pred_reward)

  differences[0] = max_dif
  if max_dif >= eps:
    mistake_probs[0] = 1
    #print(f"Real obs and decoded obs differ by more than {eps}.")
    #print(f"Max difference: {max_dif}")
    # print(f"Real obs {real_obs}")
    # print(f"Decoded obs {decoded_obs}")
  differences[1] = int(pred_terminal != carry.terminal)
  if pred_terminal != carry.terminal:
    mistake_probs[1] = 1
    #print(f"Predicted terminal {pred_terminal} does not match real terminal {carry.terminal}.")
  differences[2] = reward_dif
  if reward_dif >= eps:
    mistake_probs[2] = 1
    #print(f"Predicted reward {pred_reward} differs from real reward {carry.reward} by more than {eps}.")
  #breakpoint()
   #Ordered as observation, terminal, reward
  return mistake_probs, differences

def main():
  args = parser.parse_args()
  model_dir = args.model_dir
  steps = []
  all_mistake_probs = []
  if not model_dir.startswith("/"):
    model_dir = os.getcwd() + "/" + model_dir
  if not os.path.exists(model_dir):
      raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
  for filename in os.listdir(model_dir):
    name, filetype = filename.split(".")
    if not filetype == "pkl":
      continue
    step = int(name.split("_")[-1])

    if not args.restore_step == -1 or step == args.restore_step:
      continue

    model_path = model_dir + "/"  + filename
    

    model = load_model(model_path)
    assert isinstance(model, Dreamer), f"Loaded model should be an instance of multi agent Dreamer not {model.__class__}"
    print(f"Restored model from {model_path}")
    mistake_probs = model_walk_test(model,
                    all_outcome_check_fn = check_state_all_outcomes,
                    one_outcome_check_fn = check_state_one_outcome,
                    verbose=args.verbose,
                    visualise_tree=args.render_tree)
    all_mistake_probs.append(mistake_probs)
    steps.append(step)
    # print(f"Obs average mistake probability {mistake_probs[0]}")
    # print(f"Terminal average mistake probability {mistake_probs[1]}")
    # print(f"Reward average mistake probability {mistake_probs[2]}")
  if len(all_mistake_probs) == 0:
    raise FileNotFoundError(f"Model file {model_dir} and restore step {args.restore_step}. Did not find any file. Make sure"
                            "the directory contains a file in a form of step_restore_step.pkl, "
                            "where restore_step is either the specified number, or arbitrary integer if -1.")
  all_mistake_probs = np.asarray(all_mistake_probs)
  steps = np.asarray(steps)
  sort_indices = np.argsort(steps)
  sorted_mistake_probs = all_mistake_probs[sort_indices]
  sorted_steps = steps[sort_indices]
  fig, ax = plt.subplots()
  ax.plot(sorted_steps, sorted_mistake_probs[:, 0], label="Observation")
  ax.plot(sorted_steps, sorted_mistake_probs[:, 1], label="Terminal")
  ax.plot(sorted_steps, sorted_mistake_probs[:, 2], label="Reward")
  ax.legend()
  ax.set_xlabel("Training step")
  ax.set_ylabel("Average mistake probability")
  ax.set_title("Evaluation of Dreamer model")
  empty = ""
  game_params = model.game.params_dict()
  params_str = f'{empty.join(f"_{value}" for key, value in game_params.items())}'
  plt.savefig(f"plots/mistake_probs/{model.game.game_name()}{params_str}.pdf")

if __name__ == "__main__":
  main()