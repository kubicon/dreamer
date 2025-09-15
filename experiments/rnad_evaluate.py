from argparse import ArgumentParser
import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import os
import matplotlib.pyplot as plt


from train_utils import load_model
from games.jax_game import GameState
from rnad_dreamer import RNaDDreamer, RNaDConfig

parser = ArgumentParser()

parser.add_argument("--model_dir", type=str, default="trained_networks/rnad/goofspiel_3/seed99/network_seed42", help="Path to the directory of saved models")
parser.add_argument("--restore_step", type=int, default=10000, help="Saved step of the model to restore")

parser.add_argument("--seed", type=int, default=-1, help="Seed for the key to be used in gameplay. -1 for a random seed.")

def stringify(x)->str :
   x = np.asarray(x)
   return np.array2string(x)
   
   

def model_walk_deterministic(model:RNaDDreamer):
    """Just a sanity check to see whether the learned policies make sense at all for now.
    Assumes no chance nodes in the game."""
    game = model.world_model.game
    def _tree_walk(game_state: GameState, legals: jax.Array,depth=0):
        """Recursively walk through the game state tree."""
        legals = np.asarray(legals)
        _, p1_iset, p2_iset, _ = game.get_info(game_state)
        joint_iset = jnp.stack([p1_iset, p2_iset], axis=0)
        #TODO: For now the model is learned using the decoder 
        # on original isets so it can be passed like that. Later, it might be necessary to
        # get some latent transformation first
        pi = model.get_policy_both(model.optimizers.optimizer.model, joint_iset, legals)
        print(f"At state {game_state}")
        print(f"Model learned policy: {pi}")
        for ai1, a1 in enumerate(legals[0]):
           if a1 < 0.5:
              continue
           for ai2, a2 in enumerate(legals[1]):
            if a2 < 0.5:
                continue
            joint_action = jnp.asarray([ai1, ai2])
            next_game_state, next_terminal, next_reward, next_legals =  game.apply_action(game_state, joint_action)
            if next_terminal:
              continue
            _tree_walk(next_game_state, next_legals, depth+1)

    init_state, init_legals = game.initialize_structures()
    _tree_walk(init_state, init_legals)

def isets_close(iset1, iset2, tolerance=0.05):
   return np.linalg.norm(iset1 - iset2) <= tolerance

def find_iset_index(iset_map: list, ref_iset, tolerance=0.05):
  """Finds the iset index in the given iset map
  based on closeness and returns it, or -1
  if the iset is not found in the map"""
  for i, iset in enumerate(iset_map):
    if isets_close(iset, ref_iset, tolerance=tolerance):
        return i
  return -1

def create_iset_map(curr_iset, amount_actions, curr_legal):
    """Creates an map where at index i there is an iset corresponding to the index.
    Also returns per iset legal actions like this, per history player iset indices and per
    history player action indices (actions are differentiated by which infoset they are taken)"""
    isets = [[], []]
    iset_map = [[], []]
    iset_legal = [[], []]
    for pl in range(curr_iset.shape[0]):
      first_iset_id = len(iset_map[pl])
      for i in range(curr_iset.shape[1]): 
        curr_index = -1
        for j in range(first_iset_id, len(iset_map[pl])):
          if isets_close(iset_map[pl][j], curr_iset[pl, i]):
            curr_index = j
            break
        if curr_index < 0:
          curr_index = len(iset_map[pl])
          iset_map[pl].append(curr_iset[pl, i])
          iset_legal[pl].append(curr_legal[pl, i])
        isets[pl].append(curr_index)
        
    isets = np.array(isets)
    actions = isets[..., None] * amount_actions + np.arange(amount_actions)[None, None, ...] 
    iset_map = [np.array(i) for i in iset_map]
    iset_legal = [np.array(i) for i in iset_legal]
    return iset_map, iset_legal, isets, actions

def model_best_response(model: RNaDDreamer):
  """Compute counterfactual best response policies for both players and their 
  respective values.Returned as br value of p2 against p1
  , br value of p1 against p2, p1_br_policy, p2_br_policy.
  In this 2p0s setting, the sum of the BR-values = NashConv."""
  game = model.world_model.game
  p1_br = {}
  p2_br = {}
  game_actions = game.num_distinct_actions()


  depth_continuations = [] #[D, H(D), A(D), A(D)]
  depth_chance_probabilities = [] #[D, H(D), A(D), A(D)]
  depth_rewards = [] #[D, H(D), A(D), A(D)] only from player one perspective
  depth_actions = [] #[D, Pl, H(D), A(D)]
  depth_iset_map = [] #[D, Pl, I]
  depth_iset_legal = [] # [D, Pl, I, A(D)]
  depth_history_iset = [] # [D, Pl, H(D)]
  depth_history_legal = [] # [D, Pl, H(D), A(D), A(D)]
  depth_is_chance = [] # [D, H(D)]
  depth_history_reaches = [] # [D, Pl + 1, H(D)], includes chance reaches
  depth_behavior_policy = [] #[D, Pl, H(D), A(D)]

  vectorized_get_info = jax.vmap(game.get_info, in_axes=(0), out_axes=(0, 0, 0, 0))
  #vmap over the internal per action dimension first 
  # and then over the outer H(D, dimension)
  vectorized_next_state = jax.vmap(jax.vmap(game.apply_action, in_axes=(None, 0), out_axes=(0, 0, 0, -2)), in_axes=(0, 0), out_axes=(0, 0, 0, -3))
  vectorized_is_chance = jax.vmap(game.is_chance, in_axes=0, out_axes=0)
  vectorized_chance_info = jax.vmap(game.get_outcomes_and_probs, in_axes=0, out_axes=(0, 0, 0))
  #vmap over the H(D) dimension first and then over the player dimension
  vectorized_get_policy = nnx.vmap(nnx.vmap(model._jit_get_policy, in_axes=(None, 0, 0), out_axes=0), in_axes=(None, 0, 0), out_axes=0)

  def _construct_structures(game_states: GameState, legals_non_padded: np.ndarray, reaches: np.ndarray, depth=0):
    # Denoting this as A(D)
    max_actions = max(game.depth_chance_outcomes(depth), game_actions)
    # print(f"Handling depth {depth} with max action {max_actions}")
    # print(f"Num states: {legals_non_padded.shape[1]}")

    legals = np.pad(legals_non_padded, ((0, 0), (0, 0), (0, max_actions - legals_non_padded.shape[-1])), constant_values=0)
    is_chance = vectorized_is_chance(game_states)
    #The convention is that player 2 is considered to "act" in chance node
    # and player 1 has an invalid action
    chance_legals = np.stack([np.eye(max_actions)[0], np.ones(max_actions)], axis=0)
    legals = np.where(is_chance[None, :, None], chance_legals[:, None, :], legals)

    state_tensors, p1_isets, p2_isets, public_states = vectorized_get_info(game_states)
    #[H(D), max_{d}A(D), ...]
    chance_outcomes, chance_probs = vectorized_chance_info(game_states)
    #Chance probs are padded, but we only actually need the first A(D) of them, 
    # since the others will surely correspond to a non-valid chance outcomes
    # The padding here is just in case there are more legal
    # regular actions than chance outcomes
    chance_probs = np.pad(chance_probs, ((0, 0), (0, max(0, max_actions - chance_probs.shape[-1]))), constant_values=0)
    #[H(D), A(D)]
    chance_probs = np.take_along_axis(chance_probs, np.arange(max_actions)[None, ...], axis=-1)
    #make sure to set to 1 for non-chance nodes
    chance_probs = np.where(is_chance[..., None], chance_probs, 1)
    #and reshape into a [H(D), A(D), A(D) structure]
    # We can use tile, even though it will repeat the chance probabilities 
    # For each player 1 action, because player 1 has only 1 legal action in the chance node
    action_chance_probs = np.tile(chance_probs[..., None, :], (1, max_actions, 1))

    #This is assumed to be done by the environment, just to be sure
    invalid_iset = np.zeros_like(p1_isets[0])
    curr_iset = np.stack((p1_isets, p2_isets))
    curr_iset = np.where(is_chance[None, :, None], invalid_iset[None, None, ...], curr_iset)
    iset_map, iset_legal, isets, actions = create_iset_map(curr_iset, max_actions, legals)
    p1_legal_iset, p2_legal_iset = iset_legal[0], iset_legal[1]
    p1_legal_iset, p2_legal_iset = p1_legal_iset > 0, p2_legal_iset > 0
    
    p1_legal, p2_legal = legals[0], legals[1]
    legal = p1_legal[..., None] * p2_legal[..., None, :]

   
    pi = vectorized_get_policy(model.optimizers.optimizer.model, curr_iset, legals_non_padded)
    #[Pl, H(D), A(D)]
    pi = np.pad(pi, ((0, 0), (0, 0), (0, max_actions - pi.shape[-1])), constant_values=0)
    #Get reaches for each player.
    # They are ordered as reaches[i] = reaches of player i + 1
    # and chance is last
    player_realization = np.where(is_chance[None, :, None], 1, pi)
    #[Pl + 1, H(D), A(D)]
    next_reaches = reaches[..., None] * np.concatenate([player_realization, chance_probs[None, ...]], axis=0)
    #[H(D), A(D), A(D)]
    p1_action_realization = np.tile(next_reaches[0, ..., None], (1, 1, max_actions))
    #[H(D), A(D), A(D)]
    p2_action_realization = np.tile(next_reaches[1, ..., None, :], (1, max_actions, 1))
    #[Pl + 1, H(D), A(D), A(D)]
    # Made for a consistent shape with the nonzero filtering later
    action_realization = np.stack([p1_action_realization, p2_action_realization, np.tile(next_reaches[2, ..., None, :], (1, max_actions, 1))], axis=0)
    
    
    p1_actions = np.reshape(np.tile(np.repeat(np.arange(max_actions), max_actions), curr_iset.shape[1]), (curr_iset.shape[1], -1))
    p2_actions = np.reshape(np.tile(np.tile(np.arange(max_actions), max_actions), curr_iset.shape[1]), (curr_iset.shape[1], -1))
    joint_actions = np.stack((p1_actions, p2_actions), axis=-1)

    #Assumes that apply_action works correctly in chance nodes,
    # as in picking the appropriate chance outcome.
    #For games with larger amount of actions than chance nodes,
    # this will index out of bounds, nevertheless, because it operates under
    # jit, it should return some value and not throw an error. Which value does
    # not matter, as internally it will still decide to use the variant without chance.
    next_states, next_terminal, next_utilities, next_legals = vectorized_next_state(game_states, joint_actions)
    #collapse the action and H(D) dimension into one dimension
    # for next states, legals and reaches
    next_states = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), next_states)
    next_legals = np.reshape(next_legals, (next_legals.shape[0], -1, next_legals.shape[-1]))
    action_realization = np.reshape(action_realization, (action_realization.shape[0], -1))
    
    next_utilities = np.reshape(next_utilities, legal.shape)
    next_terminal = np.reshape(next_terminal, legal.shape)
    
    action_utility = next_utilities * legal

    # Next state will be expanded further if it is not terminal
    # was produced by legal action and it is not a chance outcome that 
    # never happens
    non_terminal = ~next_terminal * legal * (action_chance_probs >= 1e-8)
    
    
    # From [H(D), A1, A2] should select [H(D + 1)] 
    # nonzero() returns indices which are non zero in tuple 
    nonzeros = np.flatnonzero(non_terminal)
    next_states = jax.tree_util.tree_map(lambda x: x[nonzeros], next_states)
    next_legals = next_legals[:, nonzeros]
    next_reaches = action_realization[:, nonzeros]
    
    # This should be -1 everywhere, except the part where you have next history. Therey you go by terminal and just add 1
    next_history = (np.cumsum(non_terminal).reshape(non_terminal.shape) * non_terminal) - 1
    
    depth_continuations.append(next_history.astype(int))
    depth_chance_probabilities.append(action_chance_probs)
    depth_rewards.append(action_utility)
    depth_iset_map.append(iset_map)
    depth_iset_legal.append(iset_legal)
    depth_history_legal.append(legal)
    depth_history_iset.append(isets)
    depth_is_chance.append(is_chance)
    depth_actions.append(actions)
    depth_history_reaches.append(reaches)
    depth_behavior_policy.append(pi)
    if np.all(next_history < 0):
      return
    
    _construct_structures(next_states, next_legals, next_reaches, depth=depth + 1)


  init_state, init_legals = game.initialize_structures()
  init_state_padded = jax.tree_util.tree_map(lambda x: x[None, ...], init_state)
  _construct_structures(init_state_padded, init_legals[:, None, :], reaches=np.ones((3, 1)))
  depth_rewards = [np.stack((r, -r), axis=0) for r in depth_rewards]
  state_value = np.zeros((2, 1 ))
  for d in range(len(depth_history_iset) -1, -1, -1):
    p1_joint_action_value = np.where(depth_continuations[d] < 0, depth_rewards[d][0], state_value[0][depth_continuations[d]])
    p2_joint_action_value = np.where(depth_continuations[d] < 0, depth_rewards[d][1], state_value[1][depth_continuations[d]])
    
    p1_chance_weighted_value = np.squeeze(np.sum(p1_joint_action_value * depth_chance_probabilities[d] * depth_history_legal[d], axis=(-1, -2)))
    p2_chance_weighted_value = np.squeeze(np.sum(p2_joint_action_value * depth_chance_probabilities[d] * depth_history_legal[d], axis=(-1, -2)))
    
    p1_action_value = np.sum(p1_joint_action_value * depth_behavior_policy[d][1][..., None, :], -1)
    p2_action_value = np.sum(p2_joint_action_value * depth_behavior_policy[d][0][..., None], -2)
    
    p1_action_cf_value = p1_action_value * depth_history_reaches[d][1][..., None] * depth_history_reaches[d][2][..., None]
    p2_action_cf_value = p2_action_value * depth_history_reaches[d][0][..., None] * depth_history_reaches[d][2][..., None]
    
    
    p1_iset_action_value = np.bincount(depth_actions[d][0].flatten(), p1_action_cf_value.flatten()).reshape(-1, depth_actions[d].shape[-1])
    p2_iset_action_value = np.bincount(depth_actions[d][1].flatten(), p2_action_cf_value.flatten()).reshape(-1, depth_actions[d].shape[-1])
    
    p1_iset_action_value_masked = np.where(depth_iset_legal[d][0] == 1, p1_iset_action_value, np.min(p1_iset_action_value) - 1)
    p2_iset_action_value_masked = np.where(depth_iset_legal[d][1] == 1, p2_iset_action_value, np.min(p2_iset_action_value) - 1) 
    p1_br_action = np.argmax(p1_iset_action_value_masked, -1)
    p2_br_action = np.argmax(p2_iset_action_value_masked, -1)
    
    p1_history_br = p1_br_action[depth_history_iset[d][0]]
    p2_history_br = p2_br_action[depth_history_iset[d][1]]
    
    
    p1_br_policy = np.eye(p1_iset_action_value.shape[-1])[p1_br_action]
    p2_br_policy = np.eye(p2_iset_action_value.shape[-1])[p2_br_action]
    
    for i, iset in enumerate(depth_iset_map[d][0]):
      #All zeros isets are invalid isets
      if np.sum((iset) **2) <= 1e-5:
        continue
      p1_br[stringify(iset)] = p1_br_policy[i]
    for i, iset in enumerate(depth_iset_map[d][1]):
      #All zeros isets are invalid isets
      if np.sum((iset) **2) <= 1e-5:
        continue
      p2_br[stringify(iset)] = p2_br_policy[i]
    
    p1_history_value = np.squeeze(np.take_along_axis(p1_action_value, p1_history_br[..., None], 1))
    p2_history_value = np.squeeze(np.take_along_axis(p2_action_value, p2_history_br[..., None], 1)) 
    
    state_value = np.where(np.squeeze(depth_is_chance[d]), np.stack((p1_chance_weighted_value, p2_chance_weighted_value), axis=0), np.stack((p1_history_value, p2_history_value), 0))
  return state_value[1], state_value[0], p1_br, p2_br
   


def test_loaded(args):
  print(f"Evaluating model from {args.model_dir} at step {args.restore_step} with seed {args.seed}")
  model_path = args.model_dir
  if not model_path.startswith("/"):
    model_path = os.getcwd() + "/" + model_path
  model_path = model_path + f"/step_{args.restore_step}.pkl"
  if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} does not exist.")

  model = load_model(model_path)
  assert isinstance(model, RNaDDreamer), f"Loaded model should be an instance of RNaDDreamer not {model.__class__}"
  #model_walk_deterministic(model)
  p2_br_val, p1_br_val, p1_br, p2_br = model_best_response(model)
  print(f"P2 best response value against p1: {p2_br_val}")
  print(f"P1 best response value against p2 {p1_br_val}")
  #breakpoint()

def test_retrain(args):
  print(f"Evaluating retraining model from {args.model_dir} at step {args.restore_step} with seed {args.seed}")
  model_path = args.model_dir
  if not model_path.startswith("/"):
    model_path = os.getcwd() + "/" + model_path
  model_path = model_path + f"/step_{args.restore_step}.pkl"
  if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} does not exist.")

  model = load_model(model_path)
  assert isinstance(model, RNaDDreamer), f"Loaded model should be an instance of RNaDDreamer not {model.__class__}"
  config = model.config
  neurd_steps = 600
  after_init_step_multiplier = 4
  init_policy_steps = 10
  new_config = RNaDConfig(
      batch_size=config.batch_size,
      seed=config.seed,
      use_learned_model = config.use_learned_model,

      eta=config.eta,
      sampling_epsilon=config.sampling_epsilon,
      state_sample_threshold=config.state_sample_threshold,

      # Entropy schedule parameters
      entropy_schedule_size = (neurd_steps, after_init_step_multiplier * neurd_steps),
      entropy_schedule_repeats = (init_policy_steps, 1),
      
      #V-Trace parameters
      rho_vtrace = config.rho_vtrace,
      c_vtrace = config.c_vtrace,
      gamma_vtrace = config.gamma_vtrace,
      lambda_vtrace = config.lambda_vtrace,

      # NeuRD parameters
      neurd_clip = config.neurd_clip,
      neurd_threshold = config.neurd_threshold,

      # Ordered as (hidden_layer_features, num_hidden_layers)
      rnad_network_details = config.rnad_network_details,

      learning_rate = config.learning_rate,
      network_seed = config.network_seed
  )
  p1_exploitabilities = []
  p2_exploitabilities = []
  clean_model = RNaDDreamer(model.world_model, new_config)
  for i in range(init_policy_steps):
    for j in range(neurd_steps):
       clean_model.step()
    # print(f"Step {clean_model.learner_steps}")
    # print(f"Policy switch step {clean_model.policy_switch_steps}")
    #model_walk_deterministic(clean_model, seed)
    p2_br_val, p1_br_val, p1_br, p2_br = model_best_response(clean_model)
    print(f"P2 best response value against p1: {p2_br_val}")
    print(f"P1 best response value against p2 {p1_br_val}")
    p1_exploitabilities.append(p2_br_val)
    p2_exploitabilities.append(p1_br_val)
    #breakpoint()
  p1_exploitabilities = np.asarray(p1_exploitabilities)
  p2_exploitabilities = np.asarray(p2_exploitabilities)
  policy_switch_steps = np.arange(init_policy_steps)
  plt.plot(policy_switch_steps, p1_exploitabilities, label="Player 1 exploitability")
  plt.plot(policy_switch_steps, p2_exploitabilities, label="Player 2 exploitability")
  plt.legend()
  plt.savefig(f"plots/br_values/{model.world_model.game.game_name()}/neurd_steps{neurd_steps}.pdf")
  
        
  

def main():
  args = parser.parse_args()
  test_retrain(args)
  #test_loaded(args)
  

if __name__ == "__main__":
  main()