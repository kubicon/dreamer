
from argparse import ArgumentParser
import os
import numpy as np
import jax
import jax.numpy as jnp

from dreamer import Dreamer
from games.point_card_matching import PointCardMatching, PointCardMatchingStochastic
from train_utils import load_model, get_reference_policy

parser = ArgumentParser()
parser.add_argument("--model_dir", type=str, default="trained_networks/point_card_matching_3/seed99/network_seed42", help="Path to the directory of saved models")
parser.add_argument("--restore_step", type=int, default=1000, help="Saved step of the model to restore")

parser.add_argument("--seed", type=int, default=-1, help="Seed for the key to be used in gameplay. -1 for a random seed.")


def model_walk_test_deterministic(model:Dreamer, seed:int, eps:float = 0.05):
  assert isinstance(model.game, PointCardMatching), f"This test assumes deterministic point card matching game, not {model.game.__class__}"
  key = jax.random.key(seed)
  key, init_key = jax.random.split(key)

  def _tree_walk(state, legals, hidden_state, stoch_state, key, depth=0):
    legals = np.asarray(legals)
    real_obs = model.game.get_info(state)[1]
    stoch_state = jax.nn.softmax(stoch_state, axis=-1)
    max_probs = jnp.max(stoch_state, axis=-1)
    max_indices = jnp.argmax(stoch_state, axis=-1)
    max_deter_state = jax.nn.one_hot(max_indices, stoch_state.shape[-1], axis=-1)
    decoded_obs = model.get_decoder(model.optimizers.decoder_optimizer.model, hidden_state, max_deter_state)
    reward, terminal = model.get_reward_and_terminal(model.optimizers.predictor_optimizer.model,hidden_state, max_deter_state)
    print(f"In state {state}")
    print(f"Predicted reward {reward} predicted terminal {terminal}")
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
      
    
    pi = np.asarray(get_reference_policy(state, legals))
    for ai, a in enumerate(pi):
      if a < eps:
        continue
      key, action_key = jax.random.split(key)
      next_state, next_terminal, next_reward, next_legals = model.game.apply_action(state, action_key, depth, ai)
      ai_oh = jax.nn.one_hot(ai, legals.shape[0])
      if next_terminal:
        continue
      gru_input = jnp.concatenate([max_deter_state.ravel(), ai_oh.ravel()], axis=0)
      next_hidden = model.optimizers.sequence_optimizer.model(hidden_state, gru_input)
      next_stoch_state = model.optimizers.dynamics_optimizer.model(next_hidden)
      _tree_walk(next_state, next_legals, next_hidden, next_stoch_state, key, depth+1)
  
  init_state, init_legals = model.game.initialize_structures(init_key)
  init_hidden = jnp.zeros(model.config.hidden_state_size) 
  init_obs = model.game.get_info(init_state)[1]
  init_stoch_state = model.optimizers.encoder_optimizer.model(init_hidden, init_obs)
  _tree_walk(init_state, init_legals, init_hidden, init_stoch_state, key)


def check_outcomes(stoch_state: jax.Array, is_chance:bool, eps:float) ->list:
  """Checks validity of learned distributions. In non chance levels, all categoricals
  should be deterministic. In chance level, one should have ideally two 0.5 probabilities.
  Returns either the deterministic state corresponding to the max probability outcome, or
  the two, which will be picked from the distribution corresponding to the two 0.5. For now 
  assumed to be used only with one categorical"""
  
  stoch_state = jax.nn.softmax(stoch_state, axis=-1)
  #repr_stoch_state = jax.nn.softmax(stoch_state, axis=-1)

  max_probs = jnp.max(stoch_state, axis=-1)
  max_indices = jnp.argmax(stoch_state, axis=-1)
  #repr_max_probs = jnp.max(repr_stoch_state, axis=-1)

  #FROM HERE ON I THINK IT ONLY WORKS FOR SINGLE CATEGORICAL
  two_max_probs, two_max_indices = jax.lax.top_k(stoch_state, 2)
  #repr_two_max_probs, _ = jax.lax.top_k(repr_stoch_state, 2)
  if is_chance:
    if jnp.max(jnp.abs(0.5 - two_max_probs)) >= eps:
      print(f"Stochastic state differs from a stochastic uniform by more than {eps}")
      print(f"Stochastic state two_max_probs {two_max_probs}")
      #print(f"Represented (posterior) stochastic state two max probs {repr_two_max_probs}")
  else:
    if jnp.max(jnp.abs(1 - max_probs)) >= eps:
      print(f"Stochastic state differs from deterministic more than {eps}")
      print(f"Stochastic state max probs {max_probs}")
      #print(f"Represented (posterior) stochastic state max probs {repr_max_probs}")
  #TODO: This only works for a single categorical for now
  two_max_dets = jax.nn.one_hot(two_max_indices, stoch_state.shape[-1], axis=-1)
  next_deters = [det for det in two_max_dets[0]] if is_chance else jax.nn.one_hot(max_indices, stoch_state.shape[-1], axis=-1)
  return next_deters


def model_walk_test_stochastic(model:Dreamer, seed:int, eps:float = 0.05):
  assert isinstance(model.game, PointCardMatchingStochastic), f"This test assumes stochastic point card matching game, not {model.game.__class__}"
  key = jax.random.key(seed)
  key, init_key = jax.random.split(key)

  def _tree_walk(state, legals, hidden_state, deter_state, key, depth=0):
    legals = np.asarray(legals)
    real_obs = model.game.get_info(state)[1]

    decoded_obs = model.get_decoder(model.optimizers.decoder_optimizer.model, hidden_state, deter_state)
    reward, terminal = model.get_reward_and_terminal(model.optimizers.predictor_optimizer.model,hidden_state, deter_state)
    print(f"In state {state}")
    print(f"Predicted reward {reward} predicted terminal {terminal}")


    if jnp.max(jnp.abs(real_obs - decoded_obs)) >= eps:
      print(f"Real obs and decoded obs differ by more than {eps}")
      print(f"Real obs {real_obs}")
      print(f"Decoded obs {decoded_obs}")
       
    pi = np.asarray(get_reference_policy(state, legals))
    for ai, a in enumerate(pi):
      if a < eps:
        continue
      key, action_key = jax.random.split(key)
      next_state, next_terminal, next_reward, next_legals = model.game.apply_action(state, action_key, depth, ai)
      is_chance = depth == model.game.num_cards - 3
      ai_oh = jax.nn.one_hot(ai, legals.shape[0])
      if next_terminal:
        continue
      
      gru_input = jnp.concatenate([deter_state.ravel(), ai_oh.ravel()], axis=0)
      next_hidden = model.optimizers.sequence_optimizer.model(hidden_state, gru_input)
      next_stoch_state = model.optimizers.dynamics_optimizer.model(next_hidden)

      #represented_next_stoch = model.optimizers.encoder_optimizer.model(next_hidden, real_obs)
      next_states = model.game.generate_all_chance_outcomes(next_state) if is_chance else [next_state]
      next_deters = check_outcomes(next_stoch_state, is_chance, eps)
      for next_state, next_deter_state in zip(next_states, next_deters):
        _tree_walk(next_state, next_legals, next_hidden, next_deter_state, key, depth+1)
  
  init_state, init_legals = model.game.initialize_structures(init_key)
  init_hidden = jnp.zeros(model.config.hidden_state_size) 
  init_obs = model.game.get_info(init_state)[1]
  init_stoch_state = model.optimizers.encoder_optimizer.model(init_hidden, init_obs)
  init_deter = check_outcomes(init_stoch_state, False, eps)[0]
  _tree_walk(init_state, init_legals, init_hidden, init_deter, key)

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
  #model_walk_test_deterministic(model, seed)
  model_walk_test_stochastic(model, seed)

if __name__ == "__main__":
  main()