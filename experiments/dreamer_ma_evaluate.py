from argparse import ArgumentParser
import os
import sys
import numpy as np
import jax
import flax.nnx as nnx
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt

from dreamer_ma import DreamerMA
from rnad_dreamer_joint import RNaDDreamerJoint
from dreamer_actor_critic import DreamerActorCritic
from train_utils import load_model
from experiments.eval_utils import model_walk_test, cartesian_product, WalkCarry
#from pyinstrument import Profiler


parser = ArgumentParser()
parser.add_argument("--model_dir", type=str, default="trained_networks/dreamer/goofspiel_3/seed99/network_seed42", help="Path to the directory of saved models")
parser.add_argument("--restore_step", type=int, default=-1, help="Saved step of the model to restore. If -1, checks all models within that folder.")

parser.add_argument("--verbose", action="store_true", help="A flag whether to also print information about states being checked")
parser.add_argument("--render_tree", action="store_true", help="A flag whether to create the model EFG-style tree and render it.")


def check_state_all_outcomes(model: DreamerMA, carry: WalkCarry,  eps: float, outcome_threshold: float = 0.1, verbose = False):
  """Checks for a stochastic state whether all deterministic 
  states, where their components have pbt >= outcome_threshold produce valid results."""
  mistake_probs = np.zeros(5)
  stoch_state = np.asarray(carry.stoch_state)
  stoch_state_max_probs = stoch_state.max(axis=-1)
  #Make sure that the threshold does not filter out
  # outcomes so that there is none left for some categorical
  threshold = min(outcome_threshold, np.min(stoch_state_max_probs))
  _, real_p1_iset, real_p2_iset, _ = model.game.get_info(carry.game_state)
  #real_obs = np.stack([real_p1_iset, real_p2_iset], axis=0)
  #print(f"Checking state {carry.game_state} with threshold {threshold}, all outcomes")
  #TODO: Could that be done more efficiently without the loop over classes?
  num_classes = stoch_state.shape[0]
  deter_states = (stoch_state >= threshold).astype(int)
  class_indices, category_indices = np.nonzero(deter_states)
  per_class_valids = []
  for i in range(num_classes):
    single_class_indices = category_indices[class_indices == i]
    per_class_valids.append(single_class_indices)

  combinations = cartesian_product(*per_class_valids)
  #print(f"Num outcomes: {len(combinations)}")
  for comb in combinations:
    sampled_deter = jax.nn.one_hot(comb, stoch_state.shape[-1])
    p1_decoded_iset = model.get_decoder(model.optimizers.p1_decoder_optimizer.model, carry.hidden_state, sampled_deter)
    p2_decoded_iset = model.get_decoder(model.optimizers.p2_decoder_optimizer.model, carry.hidden_state, sampled_deter)
    probs = [stoch_state[i, comb_part] for i, comb_part in enumerate(comb)]
    joint_prob = np.prod(probs)
    pred_reward, pred_terminal, pred_legal = model.get_predictor(model.optimizers.predictor_optimizer.model, model.optimizers.legal_actions_optimizer.model, carry.hidden_state, sampled_deter)
    p1_iset_max_difference = jnp.max(jnp.abs(real_p1_iset - p1_decoded_iset))
    p2_iset_max_difference = jnp.max(jnp.abs(real_p2_iset - p2_decoded_iset))
    if p1_iset_max_difference >= eps:
      mistake_probs[0] += joint_prob
      #print(f"Real iset and decoded iset for player 1 differ by more than {eps} for outcome {comb} with probabilties {probs}")
      #print(f"Max difference {p1_iset_max_difference}")
    if p2_iset_max_difference >= eps:
      mistake_probs[1] += joint_prob
      #print(f"Real iset and decoded iset for player 2 differ by more than {eps} for outcome {comb} with probabilties {probs}")
      #print(f"Max difference {p2_iset_max_difference}")
    if pred_terminal != carry.terminal:
      mistake_probs[2] += joint_prob
      #print(f"Predicted terminal {pred_terminal} does not match real terminal {carry.terminal} for outcome {comb} with probabilties {probs}. ")
    if jnp.abs(carry.reward - pred_reward) >= eps:
      mistake_probs[3] += joint_prob
      #print(f"Predicted reward {pred_reward} differs from real reward {carry.reward} for outcome {comb} with probabilties {probs} by more than {eps}")
    #Do not check legal actions in terminal states
    if not carry.terminal and jnp.any(pred_legal != carry.legals):
      mistake_probs[4] += joint_prob
      #print(f"Predicted legal actions {pred_legal} do not match real legal actions {carry.legals} for outcome {comb} with probabilties {probs}.")
  #breakpoint()
  #Ordered p1_iset, p2_iset, terminal, reward, legals
  return mistake_probs

def check_state_one_outcome(model: DreamerMA, carry:WalkCarry, eps:float, verbose = False):
  """Check whether the best fitting deterministic state for the state
  produces valid results. Used for post-chance node states, to check 
  whether it corresponds to the correct outcome."""
  mistake_probs = np.zeros(5)
  differences = np.zeros(5)
  _, real_p1_iset, real_p2_iset, _ = model.game.get_info(carry.game_state)
  p1_decoded_iset = model.get_decoder(model.optimizers.p1_decoder_optimizer.model, carry.hidden_state, carry.deter_state)
  p2_decoded_iset = model.get_decoder(model.optimizers.p2_decoder_optimizer.model, carry.hidden_state, carry.deter_state)
  pred_reward, pred_terminal, pred_legal = model.get_predictor(model.optimizers.predictor_optimizer.model, model.optimizers.legal_actions_optimizer.model, carry.hidden_state, carry.deter_state)
  p1_iset_max_difference = jnp.max(jnp.abs(real_p1_iset - p1_decoded_iset))
  p2_iset_max_difference = jnp.max(jnp.abs(real_p2_iset - p2_decoded_iset))
  reward_difference = jnp.abs(carry.reward - pred_reward)
  legal_diference = not carry.terminal and jnp.any(pred_legal != carry.legals)
  det_prob = jnp.prod(carry.stoch_state[carry.deter_state.astype(jnp.bool)])

  differences[0] = p1_iset_max_difference
  if p1_iset_max_difference >= eps:
    mistake_probs[0] = det_prob
    #print(f"Real iset and decoded iset for player 1 differ by more than {eps}.")
    #print(f"Max difference {p1_iset_max_difference}")
  differences[1] = p2_iset_max_difference
  if p2_iset_max_difference >= eps:
    mistake_probs[1] = det_prob
    #print(f"Real iset and decoded iset for player 2 differ by more than {eps}.")
    #print(f"Max difference {p2_iset_max_difference}")
  differences[2] = int(pred_terminal != carry.terminal)
  if pred_terminal != carry.terminal:
    mistake_probs[2] = det_prob
    #print(f"Predicted terminal {pred_terminal} does not match real terminal {carry.terminal}. ")
  differences[3] = reward_difference
  if reward_difference >= eps:
    mistake_probs[3] = det_prob
    #print(f"Predicted reward {pred_reward} differs from real reward {carry.reward} by more than {eps}.")
  #Do not check legal actions in terminal states
  differences[4] = int(legal_diference)
  if legal_diference:
    mistake_probs[4] = det_prob
    #print(f"Predicted legal actions {pred_legal} do not match real legal actions {carry.legals}.")
  #Ordered p1_iset, p2_iset, terminal, reward, legals
  #print(f"Mistake probs {mistake_probs}")
  return mistake_probs, differences

def main():
  args = parser.parse_args()
  model_dir = args.model_dir
  all_mistake_probs = []
  steps = []
  distribution_mismatch_probs = []
  if not model_dir.startswith("/"):
    model_dir = os.getcwd() + "/" + model_dir
  if not os.path.exists(model_dir):
      raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
  #profiler = Profiler()
  print("Starting evaluation")
  start_time = time.time()
  #profiler.start()
  first = True
  model = None
  plot_subdir_str = "dreamer_only"
  for filename in os.listdir(model_dir):
    if not os.path.isfile(os.path.join(model_dir, filename)):
      continue
    name, filetype = filename.split(".")
    if not filetype == "pkl":
      continue
    step = int(name.split("_")[-1])

    if not (args.restore_step == -1 or step == args.restore_step):
      continue

    model_path = model_dir + "/"  + filename
    
    #This assumes all the models were trained with the same config
    # else it will break
    if first:
      model = load_model(model_path)
       #To allow for retrieving the world model of already trained RNaD
      # particularly relevant when training jointly.
      if isinstance(model, RNaDDreamerJoint):
        plot_subdir_str = "joint_rnad"
        model = model.world_model
      elif isinstance(model, DreamerActorCritic):
        plot_subdir_str = "joint"
        model = model.world_model
      elif not isinstance(model, DreamerMA):
        raise ValueError(f"The given model should be instance of RNaDDreamer, DreamerActorCritic or DreamerMA, not {model.__class__}")
      first=False
    else:
      temp_model = load_model(model_path)
       #To allow for retrieving the world model of already trained RNaD
      # particularly relevant when training jointly.
      if isinstance(temp_model, RNaDDreamerJoint):
        plot_subdir_str = "joint_rnad"
        temp_model = temp_model.world_model
      elif isinstance(temp_model, DreamerActorCritic):
        plot_subdir_str = "joint"
        temp_model = temp_model.world_model
      elif not isinstance(temp_model, DreamerMA):
        raise ValueError(f"The given model should be instance of RNaDDreamer, DreamerActorCritic or DreamerMA, not {model.__class__}")
      #TODO: Updating this way still forces retracing of get_info and
      # initialize_structures of the game, since it is called in init. In general
      # we just need the state of the optimizers object from the model
      # and the rest of the operations are redundant.
      nnx.update(model.optimizers, nnx.split(temp_model.optimizers)[1])
      #model.optimizers = model.update_nnx(model.optimizers, nnx.split(temp_model.optimizers)[1])

    print(f"Restored model from {model_path}")
    #breakpoint()
    mistake_probs = model_walk_test(model,
                    all_outcome_check_fn = check_state_all_outcomes,
                    one_outcome_check_fn = check_state_one_outcome,
                    verbose=args.verbose,
                    visualise_tree=args.render_tree)
    all_mistake_probs.append(mistake_probs)
    steps.append(step)
    # print(f"Mistake probs {mistake_probs}")
    # print(f"Player 1 iset average mistake probability {mistake_probs[0]}")
    # print(f"Player 2 iset average mistake probability {mistake_probs[1]}")
    # print(f"Terminal average mistake probability {mistake_probs[2]}")
    # print(f"Reward average mistake probability {mistake_probs[3]}")
    # print(f"Legal actions average mistake probability {mistake_probs[4]}")
  print("Ended evaluation")
  print(f"Evaluation took {time.time() - start_time:.2f} seconds.")
  #profiler.stop()
  #print(profiler.output_text(color=True, unicode=True))
  if len(all_mistake_probs) == 0:
    raise FileNotFoundError(f"Model directory {model_dir} and restore step {args.restore_step}. Did not find any file. Make sure"
                            "the directory contains a file in a form of step_restore_step.pkl, "
                            "where restore_step is either the specified number, or arbitrary integer if -1.")
  all_mistake_probs = np.asarray(all_mistake_probs)
  steps = np.asarray(steps)
  sort_indices = np.argsort(steps)
  sorted_mistake_probs = all_mistake_probs[sort_indices]
  sorted_steps = steps[sort_indices]

  fig, ax = plt.subplots()
  ax.plot(sorted_steps, sorted_mistake_probs[:, 0], label="Player 1 iset")
  ax.plot(sorted_steps, sorted_mistake_probs[:, 1], label="Player 2 iset")
  ax.plot(sorted_steps, sorted_mistake_probs[:, 2], label="Terminal")
  ax.plot(sorted_steps, sorted_mistake_probs[:, 3], label="Reward")
  ax.plot(sorted_steps, sorted_mistake_probs[:, 4], label="Legal actions")
  ax.legend()
  ax.set_xlabel("Training step")
  ax.set_ylabel("Average mistake probability")
  ax.set_title("Evaluation of DreamerMA model")
  empty = ""
  game_params = model.game.params_dict()
  params_str = f'{empty.join(f"_{value}" for key, value in game_params.items())}'
  plt_dir = f"plots/mistake_probs/{plot_subdir_str}"
  if not os.path.exists(plt_dir):
    os.makedirs(plt_dir)
  plt.savefig(f"{plt_dir}/{model.game.game_name()}{params_str}.pdf")
  print(f"Saved plot at {plt_dir}/{model.game.game_name()}{params_str}.pdf")

if __name__ == "__main__":
  main()