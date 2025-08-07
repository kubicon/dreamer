
from argparse import ArgumentParser
import os
import numpy as np
import jax
import jax.numpy as jnp

from dreamer import Dreamer
from games.point_card_matching import PointCardMatching, PointCardMatchingStochastic
from train_utils import load_model, get_reference_policy

from itertools import product

parser = ArgumentParser()
parser.add_argument("--model_dir", type=str, default="trained_networks/point_card_matching_3/seed99/network_seed42", help="Path to the directory of saved models")
parser.add_argument("--restore_step", type=int, default=1000, help="Saved step of the model to restore")

parser.add_argument("--seed", type=int, default=-1, help="Seed for the key to be used in gameplay. -1 for a random seed.")


def model_walk_test_deterministic(model:Dreamer, seed:int, eps:float = 0.05):
  assert isinstance(model.game, PointCardMatching), f"This test assumes deterministic point card matching game, not {model.game.__class__}"
  key = jax.random.key(seed)
  key, init_key = jax.random.split(key)

  def _tree_walk(state, legals, hidden_state, stoch_state, reward, terminal, key, depth=0):
    legals = np.asarray(legals)
    real_obs = model.game.get_info(state)[1]
    stoch_state = jax.nn.softmax(stoch_state, axis=-1)
    max_probs = jnp.max(stoch_state, axis=-1)
    max_indices = jnp.argmax(stoch_state, axis=-1)
    max_deter_state = jax.nn.one_hot(max_indices, stoch_state.shape[-1], axis=-1)
    decoded_obs = model.get_decoder(model.optimizers.decoder_optimizer.model, hidden_state, max_deter_state)
    pred_reward, pred_terminal = model.get_reward_and_terminal(model.optimizers.predictor_optimizer.model,hidden_state, max_deter_state)
    print(f"In state {state}")
    if jnp.abs(reward - pred_reward) >= eps:
      print(f"Predicted reward {pred_reward} differs from real reward {reward} by more than {eps}")
    if pred_terminal != terminal:
      print(f"Predicted terminal {pred_terminal} does not match real terminal {terminal}")
    if jnp.max(jnp.abs(real_obs - decoded_obs)) >= eps:
      print(f"Real obs and decoded obs differ by more than {eps}")
      print(f"Real obs {real_obs}")
      print(f"Decoded obs {decoded_obs}")
    if jnp.max(jnp.abs(1 - max_probs)) >= eps:
      print(f"Stoch state differs from deterministic by more than {eps}")
      print(f"Stoch state max_probs {max_probs}")
      represented_stoch = jax.nn.softmax(model.optimizers.encoder_optimizer.model(hidden_state, real_obs), axis=-1)
      repr_max_probs = jnp.max(represented_stoch, axis=-1)
      print(f"Represented (posterior) stochastic state max_probs {repr_max_probs}")
      
    if terminal:
      return
    pi = np.asarray(get_reference_policy(state, legals))
    for ai, a in enumerate(pi):
      if a < eps:
        continue
      key, action_key = jax.random.split(key)
      next_state, next_terminal, next_reward, next_legals = model.game.apply_action(state, action_key, depth, ai)
      ai_oh = jax.nn.one_hot(ai, legals.shape[0])
      gru_input = jnp.concatenate([max_deter_state.ravel(), ai_oh.ravel()], axis=0)
      next_hidden = model.optimizers.sequence_optimizer.model(hidden_state, gru_input)
      next_stoch_state = model.optimizers.dynamics_optimizer.model(next_hidden)
      _tree_walk(next_state, next_legals, next_hidden, next_stoch_state, next_reward, next_terminal, key, depth+1)
  
  init_state, init_legals = model.game.initialize_structures(init_key)
  init_hidden = jnp.zeros(model.config.hidden_state_size) 
  init_obs = model.game.get_info(init_state)[1]
  init_stoch_state = model.optimizers.encoder_optimizer.model(init_hidden, init_obs)
  _tree_walk(init_state, init_legals, init_hidden, init_stoch_state, 0,  False, key)


def check_outcomes(stoch_state: jax.Array, is_chance:bool, num_chance_outcomes:int,  eps:float) ->list:
  """Checks validity of learned distributions. In non chance levels, all categoricals
  should be deterministic. In chance level, one should have ideally two 0.5 probabilities.
  Returns either the deterministic state corresponding to the max probability outcome, or
  the two, which will be picked from the distribution corresponding to the two 0.5. For now 
  assumed to be used only with one categorical"""
  
  #repr_stoch_state = jax.nn.softmax(stoch_state, axis=-1)

  max_probs = jnp.max(stoch_state, axis=-1)
  max_indices = jnp.argmax(stoch_state, axis=-1)
  #repr_max_probs = jnp.max(repr_stoch_state, axis=-1)

  #FROM HERE ON I THINK IT ONLY WORKS FOR SINGLE CATEGORICAL
  chance_max_probs, chance_max_indices = jax.lax.top_k(stoch_state, num_chance_outcomes)
  chance_probs =  1 / num_chance_outcomes
  #Find the categorical that is closest to the uniform distribution
  uniform_distance = jnp.sum((chance_max_probs - chance_probs) ** 2, axis=-1)
  chance_dist_idx = jnp.argmin(uniform_distance)
  uniform_categorical = chance_max_probs[chance_dist_idx, :]
  uniform_categorical_indices = chance_max_indices[chance_dist_idx, :]
  #repr_two_max_probs, _ = jax.lax.top_k(repr_stoch_state, 2)
  if is_chance:
    if jnp.max(jnp.abs(chance_probs - uniform_categorical)) >= eps:
      print(f"Stochastic state differs from a stochastic uniform by more than {eps}")
      print(f"Stochastic state  max probs {uniform_categorical}")
      #print(f"Represented (posterior) stochastic state two max probs {repr_two_max_probs}")
  else:
    if jnp.max(jnp.abs(1 - max_probs)) >= eps:
      print(f"Stochastic state differs from deterministic more than {eps}")
      print(f"Stochastic state max probs {max_probs}")
      #print(f"Represented (posterior) stochastic state max probs {repr_max_probs}")

  chance_max_dets = jax.nn.one_hot(uniform_categorical_indices, stoch_state.shape[-1], axis=-1)
  deter_state = jax.nn.one_hot(max_indices, stoch_state.shape[-1], axis=-1)
  def _make_det_from_chance(chance_det, chance_idx, argmax_state):
    return jnp.concatenate([argmax_state[:chance_idx, :], chance_det[None, ...], argmax_state[chance_idx + 1:, :]], axis=0)
  next_deters = [_make_det_from_chance(det, chance_dist_idx, deter_state) for det in chance_max_dets] if is_chance else [deter_state]
  return next_deters

def get_closest_deter(model: Dreamer, next_hidden_state, next_deters, next_state):
  """Find the deterministic state of the possible outcomes that is the best fit
  to the next state based on decoder."""
  min_dist = jnp.inf
  closest_deter = None
  real_obs = model.game.get_info(next_state)[1]
  for next_deter in next_deters:
    decoded_obs = model.get_decoder(model.optimizers.decoder_optimizer.model, next_hidden_state, next_deter)
    dist = jnp.sum((real_obs - decoded_obs) ** 2)
    if dist < min_dist:
      min_dist = dist
      closest_deter = next_deter
  return closest_deter

def check_terminal(model: Dreamer, terminal_stoch_state, terminal_hidden_state, terminal_game_state, real_terminal: bool, real_reward: float,  eps: float, outcome_threshold: float = 0.1):
  """Checks for a terminal state whether all the possible outcomes produce valid output.."""
  terminal_stoch_state = np.asarray(terminal_stoch_state)
  real_obs = model.game.get_info(terminal_game_state)[1]
  print(f"Checking terminal state {terminal_game_state}")
  #TODO: Could that be done more efficiently without using the itertools product?
  # And loop over classes?
  num_classes = terminal_stoch_state.shape[0]
  terminal_deter_states = (terminal_stoch_state >= outcome_threshold).astype(int)
  class_indices, category_indices = np.nonzero(terminal_deter_states)
  per_class_valids = []
  for i in range(num_classes):
    single_class_indices = category_indices[class_indices == i]
    per_class_valids.append(single_class_indices)

  combinations = list(product(*per_class_valids))
  for comb in combinations:
    sampled_deter = jax.nn.one_hot(comb, terminal_stoch_state.shape[-1])
    decoded_obs = model.get_decoder(model.optimizers.decoder_optimizer.model, terminal_hidden_state, sampled_deter)
    probs = [terminal_stoch_state[i, comb_part] for i, comb_part in enumerate(comb)]
    pred_reward, pred_terminal = model.get_reward_and_terminal(model.optimizers.predictor_optimizer.model, terminal_hidden_state, sampled_deter)
    if jnp.max(jnp.abs(real_obs - decoded_obs)) >= eps:
      print(f"Real obs and decoded obs differ by more than {eps} for outcome {comb} with probabilties {probs}")
      print(f"Real obs {real_obs}")
      print(f"Decoded obs {decoded_obs}")
    if jnp.abs(real_reward - pred_reward) >= eps:
      print(f"Predicted reward {pred_reward} differs from real reward {real_reward} for outcome {comb} with probabilties {probs} by more than {eps}")
    if pred_terminal != real_terminal:
      print(f"Predicted terminal {pred_terminal} does not match real terminal for outcome {comb} with probabilties {probs} {real_terminal}")
  

def model_walk_test_stochastic(model:Dreamer, seed:int, eps:float = 0.05):
  assert isinstance(model.game, PointCardMatchingStochastic), f"This test assumes stochastic point card matching game, not {model.game.__class__}"
  key = jax.random.key(seed)
  key, init_key = jax.random.split(key)

  def _tree_walk(state, legals, hidden_state, deter_state, reward, terminal, key, depth=0):
    legals = np.asarray(legals)
    real_obs = model.game.get_info(state)[1]

    decoded_obs = model.get_decoder(model.optimizers.decoder_optimizer.model, hidden_state, deter_state)
    pred_reward, pred_terminal = model.get_reward_and_terminal(model.optimizers.predictor_optimizer.model,hidden_state, deter_state)
    print(f"In state {state}")

    if jnp.abs(reward - pred_reward) >= eps:
      print(f"Predicted reward {pred_reward} differs from real reward {reward} by more than {eps}")
    if pred_terminal != terminal:
      print(f"Predicted terminal {pred_terminal} does not match real terminal {terminal}")
    if jnp.max(jnp.abs(real_obs - decoded_obs)) >= eps:
      print(f"Real obs and decoded obs differ by more than {eps}")
      print(f"Real obs {real_obs}")
      print(f"Decoded obs {decoded_obs}")
    if terminal:
      return
       
    pi = np.asarray(get_reference_policy(state, legals))
    for ai, a in enumerate(pi):
      if a < eps:
        continue
      key, action_key = jax.random.split(key)
      next_state, next_terminal, next_reward, next_legals = model.game.apply_action(state, action_key, depth, ai)
      is_chance = depth == model.game.chance_turn
      ai_oh = jax.nn.one_hot(ai, legals.shape[0])
      gru_input = jnp.concatenate([deter_state.ravel(), ai_oh.ravel()], axis=0)
      next_hidden = model.optimizers.sequence_optimizer.model(hidden_state, gru_input)
      next_stoch_state = jax.nn.softmax(model.optimizers.dynamics_optimizer.model(next_hidden), axis=-1)

      #represented_next_stoch = jax.nn.softmax(model.optimizers.encoder_optimizer.model(next_hidden, real_obs), axis=-1)
      next_states = model.game.generate_all_chance_outcomes(next_state) if is_chance else [next_state]
      next_deters = check_outcomes(next_stoch_state, is_chance, model.game.chance_outcomes, eps)
     
        
      for next_state in next_states:
        if next_terminal:
          check_terminal(model, next_stoch_state, next_hidden, next_state, next_terminal, next_reward, eps=0.3)
        closest_deter = get_closest_deter(model, next_hidden, next_deters, next_state)
        _tree_walk(next_state, next_legals, next_hidden, closest_deter, next_reward, next_terminal, key, depth+1)
  
  init_state, init_legals = model.game.initialize_structures(init_key)
  init_hidden = jnp.zeros(model.config.hidden_state_size) 
  init_obs = model.game.get_info(init_state)[1]
  init_stoch_state = jax.nn.softmax(model.optimizers.encoder_optimizer.model(init_hidden, init_obs))
  init_deter = check_outcomes(init_stoch_state, False, model.game.chance_outcomes, eps)[0]
  _tree_walk(init_state, init_legals, init_hidden, init_deter, 0,  False, key)

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
  assert model.game.game_name() in ["point_card_matching", "point_card_matching_stochastic"], f"Loaded model should be trained some point card matching game not {model.game.game_name()}"
  print(f"Restored model from {model_path}")
  if model.game.game_name() == "point_card_matching_stochastic": 
    model_walk_test_stochastic(model, seed)
  else:
    model_walk_test_deterministic(model, seed)

if __name__ == "__main__":
  main()