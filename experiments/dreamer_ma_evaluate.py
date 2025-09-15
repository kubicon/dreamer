from argparse import ArgumentParser
import os
import numpy as np
import jax
import chex
import jax.numpy as jnp

from dreamer_ma import DreamerMA
from train_utils import load_model
from experiments.eval_utils import model_walk_test, cartesian_product, WalkCarry

parser = ArgumentParser()
parser.add_argument("--model_dir", type=str, default="trained_networks/dreamer/goofspiel_3/seed99/network_seed42", help="Path to the directory of saved models")
parser.add_argument("--restore_step", type=int, default=1000, help="Saved step of the model to restore")




def check_state_all_outcomes(model: DreamerMA, carry: WalkCarry,  eps: float, outcome_threshold: float = 0.1):
  """Checks for a stochastic state whether all deterministic 
  states, where their components have pbt >= outcome_threshold produce valid results."""
  stoch_state = np.asarray(carry.stoch_state)
  stoch_state_max_probs = stoch_state.max(axis=-1)
  #Make sure that the threshold does not filter out
  # outcomes so that there is none left for some categorical
  threshold = min(outcome_threshold, np.min(stoch_state_max_probs))
  _, real_p1_iset, real_p2_iset, _ = model.game.get_info(carry.game_state)
  #real_obs = np.stack([real_p1_iset, real_p2_iset], axis=0)
  print(f"Checking state {carry.game_state} with threshold {threshold}, all outcomes")
  #TODO: Could that be done more efficiently without the loop over classes?
  num_classes = stoch_state.shape[0]
  deter_states = (stoch_state >= threshold).astype(int)
  class_indices, category_indices = np.nonzero(deter_states)
  per_class_valids = []
  for i in range(num_classes):
    single_class_indices = category_indices[class_indices == i]
    per_class_valids.append(single_class_indices)

  combinations = cartesian_product(*per_class_valids)
  print(f"Num outcomes: {len(combinations)}")
  for comb in combinations:
    sampled_deter = jax.nn.one_hot(comb, stoch_state.shape[-1])
    p1_decoded_iset = model.get_decoder(model.optimizers.p1_decoder_optimizer.model, carry.hidden_state, sampled_deter)
    p2_decoded_iset = model.get_decoder(model.optimizers.p2_decoder_optimizer.model, carry.hidden_state, sampled_deter)
    probs = [stoch_state[i, comb_part] for i, comb_part in enumerate(comb)]
    pred_reward, pred_terminal, pred_legal = model.get_predictor(model.optimizers.predictor_optimizer.model, model.optimizers.legal_actions_optimizer.model, carry.hidden_state, sampled_deter)
    p1_iset_max_difference = jnp.max(jnp.abs(real_p1_iset - p1_decoded_iset))
    p2_iset_max_difference = jnp.max(jnp.abs(real_p2_iset - p2_decoded_iset))
    if p1_iset_max_difference >= eps:
      print(f"Real iset and decoded iset for player 1 differ by more than {eps} for outcome {comb} with probabilties {probs}")
      print(f"Max difference {p1_iset_max_difference}")
    if p2_iset_max_difference >= eps:
      print(f"Real iset and decoded iset for player 2 differ by more than {eps} for outcome {comb} with probabilties {probs}")
      print(f"Max difference {p2_iset_max_difference}")
    if jnp.abs(carry.reward - pred_reward) >= eps:
      print(f"Predicted reward {pred_reward} differs from real reward {carry.reward} for outcome {comb} with probabilties {probs} by more than {eps}")
    #Do not check legal actions in terminal states
    if not carry.terminal and jnp.any(pred_legal != carry.legals):
      print(f"Predicted legal actions {pred_legal} do not match real legal actions {carry.legals} for outcome {comb} with probabilties {probs}.")
    if pred_terminal != carry.terminal:
      print(f"Predicted terminal {pred_terminal} does not match real terminal {carry.terminal} for outcome {comb} with probabilties {probs}. ")

def check_state_one_outcome(model: DreamerMA, carry:WalkCarry, eps:float):
  """Check whether the best fitting deterministic state for the state
  produces valid results. Used for post-chance node states, to check 
  whether it corresponds to the correct outcome."""
  _, real_p1_iset, real_p2_iset, _ = model.game.get_info(carry.game_state)
  p1_decoded_iset = model.get_decoder(model.optimizers.p1_decoder_optimizer.model, carry.hidden_state, carry.deter_state)
  p2_decoded_iset = model.get_decoder(model.optimizers.p2_decoder_optimizer.model, carry.hidden_state, carry.deter_state)
  pred_reward, pred_terminal, pred_legal = model.get_predictor(model.optimizers.predictor_optimizer.model, model.optimizers.legal_actions_optimizer.model, carry.hidden_state, sampled_deter)
  p1_iset_max_difference = jnp.max(jnp.abs(real_p1_iset - p1_decoded_iset))
  p2_iset_max_difference = jnp.max(jnp.abs(real_p2_iset - p2_decoded_iset))
  if p1_iset_max_difference >= eps:
    print(f"Real iset and decoded iset for player 1 differ by more than {eps}.")
    print(f"Max difference {p1_iset_max_difference}")
  if p2_iset_max_difference >= eps:
    print(f"Real iset and decoded iset for player 2 differ by more than {eps}.")
    print(f"Max difference {p2_iset_max_difference}")
  if jnp.abs(carry.reward - pred_reward) >= eps:
    print(f"Predicted reward {pred_reward} differs from real reward {carry.reward} by more than {eps}.")
  #Do not check legal actions in terminal states
  if not carry.terminal and jnp.any(pred_legal != carry.legals):
    print(f"Predicted legal actions {pred_legal} do not match real legal actions {carry.legals}.")
  if pred_terminal != carry.terminal:
    print(f"Predicted terminal {pred_terminal} does not match real terminal {carry.terminal}. ")

def main():
  args = parser.parse_args()
  model_path = args.model_dir
  if not model_path.startswith("/"):
    model_path = os.getcwd() + "/" + model_path
  model_path = model_path + f"/step_{args.restore_step}.pkl"
  if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} does not exist.")

  model = load_model(model_path)
  assert isinstance(model, DreamerMA), f"Loaded model should be an instance of multi agent Dreamer not {model.__class__}"
  print(f"Restored model from {model_path}")
  model_walk_test(model,
                  all_outcome_check_fn = check_state_all_outcomes,
                  one_outcome_check_fn = check_state_one_outcome)

if __name__ == "__main__":
  main()