
from argparse import ArgumentParser
import os
import numpy as np
import jax
import jax.numpy as jnp

from dreamer import Dreamer
from train_utils import load_model, get_reference_policy

parser = ArgumentParser()
parser.add_argument("--model_dir", type=str, default="trained_networks/point_card_matching_3/seed99/network_seed42", help="Path to the directory of saved models")
parser.add_argument("--restore_step", type=int, default=1000, help="Saved step of the model to restore")

parser.add_argument("--seed", type=int, default=-1, help="Seed for the key to be used in gameplay. -1 for a random seed.")


def model_walk_test_deterministic(model:Dreamer, seed:int, eps:float = 1e-3):
  assert model.game.game_name() == "point_card_matching", f"This test assumes deterministic point card matching game, not {model.game.game_name()}"
  key = jax.random.key(seed)
  key, init_key = jax.random.split(key)

  def _tree_walk(state, legals, hidden_state, stoch_state, key, depth=0):
    legals = np.asarray(legals)
    stoch_state = jax.nn.softmax(stoch_state, axis=-1)
    max_probs = jnp.max(stoch_state, axis=-1)
    if jnp.max(jnp.abs(1 - max_probs)) >= eps:
      print(f"In state {state}, stoch state differs from deterministic by more than {eps}")
      print(f"Stoch state max probs {max_probs}")
    
    max_indices = jnp.argmax(stoch_state, axis=-1)
    max_deter_state = jax.nn.one_hot(max_indices, stoch_state.shape[-1], axis=-1)
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
  model_walk_test_deterministic(model, seed)

if __name__ == "__main__":
  main()