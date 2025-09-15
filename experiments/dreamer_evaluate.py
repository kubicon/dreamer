
from argparse import ArgumentParser
import os
import numpy as np
import jax
import chex
import jax.numpy as jnp

from dreamer import Dreamer
from experiments.eval_utils import model_walk_test, cartesian_product, WalkCarry
from train_utils import load_model


parser = ArgumentParser()
parser.add_argument("--model_dir", type=str, default="trained_networks/dreamer/point_card_matching_3/seed99/network_seed42", help="Path to the directory of saved models")
parser.add_argument("--restore_step", type=int, default=1000, help="Saved step of the model to restore")

parser.add_argument("--seed", type=int, default=-1, help="Seed for the key to be used in gameplay. -1 for a random seed.")





def check_state_all_outcomes(model: Dreamer, carry: WalkCarry, eps: float, outcome_threshold: float = 0.1):
  """Checks a state whether all the possible outcomes produce valid output.
  TODO: For chance nodes, this also checks the outcomes that belong
  to different chance outcomes, so for chance nodes it will report discrepancies,
  even though the model learned correctly to distinguish between the chance outcomes."""
  stoch_state = np.asarray(carry.stoch_state)
  real_obs = model.game.get_info(carry.game_state)[1]
  print(f"Checking state {carry.game_state}, all outcomes")
  #TODO: Could that be done more efficiently without the loop over classes?
  num_classes = stoch_state.shape[0]
  deter_states = (stoch_state >= outcome_threshold).astype(int)
  class_indices, category_indices = np.nonzero(deter_states)
  per_class_valids = []
  for i in range(num_classes):
    single_class_indices = category_indices[class_indices == i]
    per_class_valids.append(single_class_indices)

  combinations = cartesian_product(*per_class_valids)
  print(f"Num outcomes: {len(combinations)}")
  for comb in combinations:
    sampled_deter = jax.nn.one_hot(comb, stoch_state.shape[-1])
    decoded_obs = model.get_decoder(model.optimizers.decoder_optimizer.model, carry.hidden_state, sampled_deter)
    probs = [stoch_state[i, comb_part] for i, comb_part in enumerate(comb)]
    pred_reward, pred_terminal = model.get_reward_and_terminal(model.optimizers.predictor_optimizer.model, carry.hidden_state, sampled_deter)
    max_dif = jnp.max(jnp.abs(real_obs - decoded_obs))
    if max_dif >= eps:
      print(f"Real obs and decoded obs differ by more than {eps} for outcome {comb} with probabilties {probs}")
      print(f"Max difference: {max_dif}")
      # print(f"Real obs {real_obs}")
      # print(f"Decoded obs {decoded_obs}")
    if jnp.abs(carry.reward - pred_reward) >= eps:
      print(f"Predicted reward {pred_reward} differs from real reward {carry.reward} for outcome {comb} with probabilties {probs} by more than {eps}")
    if pred_terminal != carry.terminal:
      print(f"Predicted terminal {pred_terminal} does not match real terminal for outcome {comb} with probabilties {probs} {carry.terminal}")

def check_state_one_outcome(model: Dreamer, carry: WalkCarry,  eps: float):
  """Check whether the given sampled deterministic state
  produces valid output. Used for checking one particular chance outcome"""
  real_obs = model.game.get_info(carry.game_state)[1]
  print(f"Checking state {carry.game_state}, closest outcome")
  decoded_obs = model.get_decoder(model.optimizers.decoder_optimizer.model, carry.hidden_state, carry.deter_state)
  pred_reward, pred_terminal = model.get_reward_and_terminal(model.optimizers.predictor_optimizer.model, carry.hidden_state, carry.deter_state)
  max_dif = jnp.max(jnp.abs(real_obs - decoded_obs))
  if max_dif >= eps:
    print(f"Real obs and decoded obs differ by more than {eps}.")
    print(f"Max difference: {max_dif}")
    # print(f"Real obs {real_obs}")
    # print(f"Decoded obs {decoded_obs}")
  if jnp.abs(carry.reward - pred_reward) >= eps:
    print(f"Predicted reward {pred_reward} differs from real reward {carry.reward} by more than {eps}.")
  if pred_terminal != carry.terminal:
    print(f"Predicted terminal {pred_terminal} does not match real terminal {carry.terminal}.")

def main():
  args = parser.parse_args()
  model_path = args.model_dir
  if not model_path.startswith("/"):
    model_path = os.getcwd() + "/" + model_path
  seed = args.seed
  if seed == -1:
    seed = np.random.randint(0, 2**32 - 1)
  model_path = model_path + f"/step_{args.restore_step}.pkl"
  if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} does not exist.")

  model = load_model(model_path)
  assert isinstance(model, Dreamer), "Loaded model is not an instance of Dreamer."
  print(f"Restored model from {model_path}")
  model_walk_test(model,
                   all_outcome_check_fn = check_state_all_outcomes, 
                   one_outcome_check_fn = check_state_one_outcome)

if __name__ == "__main__":
  main()