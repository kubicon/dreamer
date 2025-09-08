from argparse import ArgumentParser
import os
import numpy as np
import jax
import jax.numpy as jnp

from dreamer_ma import DreamerMA
from games.jax_goofspiel import JaxGoofspiel
from train_utils import load_model, get_reference_policy

from itertools import product

parser = ArgumentParser()
parser.add_argument("--model_dir", type=str, default="trained_networks/dreamer/goofspiel_3/seed99/network_seed42", help="Path to the directory of saved models")
parser.add_argument("--restore_step", type=int, default=1000, help="Saved step of the model to restore")




def check_state(model: DreamerMA, stoch_state, hidden_state, game_state, real_legal, real_terminal: bool, real_reward: float,  eps: float, outcome_threshold: float = 0.1):
  """Checks for a stochastic state whether all the possible outcomes produce valid output.."""
  stoch_state = np.asarray(stoch_state)
  stoch_state_max_probs = stoch_state.max(axis=-1)
  #Make sure that the threshold does not filter out
  # outcomes so that there is none left for some categorical
  threshold = min(outcome_threshold, np.min(stoch_state_max_probs))
  _, real_p1_iset, real_p2_iset, _ = model.game.get_info(game_state)
  #real_obs = np.stack([real_p1_iset, real_p2_iset], axis=0)
  print(f"Checking state {game_state} with threshold {threshold}")
  #TODO: Could that be done more efficiently without using the itertools product?
  # And loop over classes?
  num_classes = stoch_state.shape[0]
  deter_states = (stoch_state >= threshold).astype(int)
  class_indices, category_indices = np.nonzero(deter_states)
  per_class_valids = []
  for i in range(num_classes):
    single_class_indices = category_indices[class_indices == i]
    per_class_valids.append(single_class_indices)

  combinations = list(product(*per_class_valids))
  print(f"Num outcomes: {len(combinations)}")
  for comb in combinations:
    sampled_deter = jax.nn.one_hot(comb, stoch_state.shape[-1])
    p1_decoded_iset = model.get_decoder(model.optimizers.p1_decoder_optimizer.model, hidden_state, sampled_deter)
    p2_decoded_iset = model.get_decoder(model.optimizers.p2_decoder_optimizer.model, hidden_state, sampled_deter)
    probs = [stoch_state[i, comb_part] for i, comb_part in enumerate(comb)]
    pred_reward, pred_terminal, pred_legal = model.get_predictor(model.optimizers.predictor_optimizer.model, model.optimizers.legal_actions_optimizer.model, hidden_state, sampled_deter)
    p1_iset_max_difference = jnp.max(jnp.abs(real_p1_iset - p1_decoded_iset))
    p2_iset_max_difference = jnp.max(jnp.abs(real_p2_iset - p2_decoded_iset))
    if p1_iset_max_difference >= eps:
      print(f"Real iset and decoded iset for player 1 differ by more than {eps} for outcome {comb} with probabilties {probs}")
      print(f"Max difference {p1_iset_max_difference}")
    if p2_iset_max_difference >= eps:
      print(f"Real iset and decoded iset for player 2 differ by more than {eps} for outcome {comb} with probabilties {probs}")
      print(f"Max difference {p2_iset_max_difference}")
    if jnp.abs(real_reward - pred_reward) >= eps:
      print(f"Predicted reward {pred_reward} differs from real reward {real_reward} for outcome {comb} with probabilties {probs} by more than {eps}")
    #Do not check legal actions in terminal states
    if not real_terminal and jnp.any(pred_legal != real_legal):
      print(f"Predicted legal actions {pred_legal} do not match real legal actions {real_legal} for outcome {comb} with probabilties {probs}.")
    if pred_terminal != real_terminal:
      print(f"Predicted terminal {pred_terminal} does not match real terminal {real_terminal} for outcome {comb} with probabilties {probs}. ")


def model_walk_test_deterministic(model:DreamerMA, eps:float = 0.05):
  assert isinstance(model.game, JaxGoofspiel), f"This test assumes deterministic goofspiel game, not {model.game.__class__}"

  def _tree_walk(state, legals, hidden_state, stoch_state, reward, terminal, depth=0):
    legals = np.asarray(legals)
    # if terminal:
    #   check_state(model, stoch_state, hidden_state, state, terminal, reward, eps=0.3)
    #   return
    stoch_state = jax.nn.softmax(stoch_state, axis=-1)
    max_probs = jnp.max(stoch_state, axis=-1)
    max_indices = jnp.argmax(stoch_state, axis=-1)
    max_deter_state = jax.nn.one_hot(max_indices, stoch_state.shape[-1], axis=-1)
    check_state(model, stoch_state, hidden_state, state, legals, terminal, reward, eps=0.1)
    if terminal:
      return
    #_, real_p1_iset, real_p2_iset, _ = model.game.get_info(state)
    # p1_decoded_iset = model.get_decoder(model.optimizers.p1_decoder_optimizer.model, hidden_state, max_deter_state)
    # p2_decoded_iset = model.get_decoder(model.optimizers.p2_decoder_optimizer.model, hidden_state, max_deter_state)
    # pred_reward, pred_terminal = model.get_reward_and_terminal(model.optimizers.predictor_optimizer.model,hidden_state, max_deter_state)
    # #print(f"In state {state}")
    # if jnp.abs(reward - pred_reward) >= eps:
    #   print(f"Predicted reward {pred_reward} differs from real reward {reward} by more than {eps}")
    # if pred_terminal != terminal:
    #   print(f"Predicted terminal {pred_terminal} does not match real terminal {terminal}")
    # if jnp.max(jnp.abs(real_p1_iset - p1_decoded_iset)) >= 0.1:
    #   print(f"Real iset and decoded iset for player 1 differ by more than 0.1")
    #   print(f"Max difference {jnp.max(jnp.abs(real_p1_iset - p1_decoded_iset))}")
    # if jnp.max(jnp.abs(real_p2_iset - p2_decoded_iset)) >= 0.1:
    #   print(f"Real iset and decoded iset for player 2 differ by more than 0.1")
    #   print(f"Max difference {jnp.max(jnp.abs(real_p2_iset - p2_decoded_iset))}")
    # if jnp.max(jnp.abs(1 - max_probs)) >= eps:
    #   print(f"Stoch state differs from deterministic by more than {eps}")
    #   print(f"Stoch state max_probs {max_probs}")
    #   real_obs = jnp.stack([real_p1_iset, real_p2_iset], axis=0)
    #   represented_stoch = jax.nn.softmax(model.optimizers.encoder_optimizer.model(hidden_state, real_obs), axis=-1)
    #   repr_max_probs = jnp.max(represented_stoch, axis=-1)
    #   print(f"Represented (posterior) stochastic state max_probs {repr_max_probs}")
    pi = np.asarray(get_reference_policy(state, legals))
    for ai1, a1 in enumerate(pi[0]):
      if a1 < eps:
        continue
      for ai2, a2 in enumerate(pi[1]):
        if a2 < eps:
          continue
        joint_action = jnp.array([ai1, ai2])
        next_state, next_terminal, next_reward, next_legals = model.game.apply_action(state, joint_action)
        ai_oh = jax.nn.one_hot(joint_action, legals.shape[-1])
        gru_input = jnp.concatenate([max_deter_state.ravel(), ai_oh.ravel()], axis=0)
        next_hidden = model.optimizers.sequence_optimizer.model(hidden_state, gru_input)
        next_stoch_state = model.optimizers.dynamics_optimizer.model(next_hidden)
        _tree_walk(next_state, next_legals, next_hidden, next_stoch_state, next_reward, next_terminal,depth+1)
  
  init_state, init_legals = model.game.initialize_structures()
  init_hidden = jnp.zeros(model.config.hidden_state_size)
  _, init_p1_iset, init_p2_iset, _ = model.game.get_info(init_state) 
  init_obs = jnp.stack([init_p1_iset, init_p2_iset], axis=0)
  init_stoch_state = model.optimizers.encoder_optimizer.model(init_hidden, init_obs)
  _tree_walk(init_state, init_legals, init_hidden, init_stoch_state, 0,  False)

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
  assert isinstance(model.game, JaxGoofspiel), f"Loaded model should be trained on a goofspiel game not on {model.game.game_name()}"
  print(f"Restored model from {model_path}")
  model_walk_test_deterministic(model)

if __name__ == "__main__":
  main()