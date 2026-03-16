

import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx


from train_utils import uniform_policy
from experiments.eval_utils import cartesian_product, stringify, find_closest_index, create_infoset_map, unroll_chance_node



from games.jax_game import JaxGame, GameState
from games.model_game import DreamerModelGame, ModelGameState

from dreamer_ma import DreamerMA
from ma_rssm import MARSSM

def extract_model_policy(model: DreamerMA|None, game: JaxGame | DreamerModelGame, uniform=False)-> tuple[list, list]:
  """Extracts policies for the whole game from the RNaD model and 
  returns them as per depth
  infoset map and behavioral policies. Can also instead
   create a uniform policy, if the uniform=True option is provided or the model is None"""
  depth_behaviorals = []
  depth_infoset_map = []
  game_actions = game.num_distinct_actions()

  vectorized_get_info = jax.vmap(game.get_info, in_axes=(0), out_axes=(0, 0, 0, 0))
  #vmap over the internal per action dimension first 
  # and then over the outer H(D, dimension)
  vectorized_next_state = jax.vmap(jax.vmap(game.apply_action, in_axes=(None, 0), out_axes=(0, 0, 0, -2)), in_axes=(0, 0), out_axes=(0, 0, 0, -3))
  vectorized_is_chance = jax.vmap(game.is_chance, in_axes=0, out_axes=0)
  vectorized_chance_info = jax.vmap(game.get_outcomes_and_probs, in_axes=0, out_axes=(0, 0))
  #vmap over the H(D) dimension first and then over the player dimension
  if uniform or not model:
    vectorized_get_policy = jax.vmap(jax.vmap(uniform_policy, in_axes=(0, 0), out_axes=0), in_axes=(0, 0), out_axes=0)
  else:
    vectorized_net = MARSSM.vmap_over_net(model.optimizer.model.actor, in_axes=[(0, 0), (0, 0)], out_axes=[0, 0])
    vectorized_get_policy = lambda x, y : vectorized_net(x, y)[0]
  def _tree_walk(game_states: GameState, legals_non_padded: jax.Array, depth=0):
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

    state_tensors, p1_infosets, p2_infosets, public_states = vectorized_get_info(game_states)
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
    invalid_infoset = np.zeros_like(p1_infosets[0])
    curr_infoset = np.stack((p1_infosets, p2_infosets))
    curr_infoset = np.where(is_chance[None, :, None], invalid_infoset[None, None, ...], curr_infoset)
    infoset_map, infoset_legal, infosets, actions = create_infoset_map(curr_infoset, max_actions, legals_non_padded)
    
    p1_legal, p2_legal = legals[0], legals[1]
    legal = p1_legal[..., None] * p2_legal[..., None, :]

    pi = vectorized_get_policy(jnp.asarray(infoset_map), jnp.asarray(infoset_legal))
    
    
    p1_actions = np.reshape(np.tile(np.repeat(np.arange(max_actions), max_actions), curr_infoset.shape[1]), (curr_infoset.shape[1], -1))
    p2_actions = np.reshape(np.tile(np.tile(np.arange(max_actions), max_actions), curr_infoset.shape[1]), (curr_infoset.shape[1], -1))
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
    next_states = jax.tree.map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), next_states)
    next_legals = np.reshape(next_legals, (next_legals.shape[0], -1, next_legals.shape[-1]))
    next_terminal = np.reshape(next_terminal, legal.shape)
    

    # Next state will be expanded further if it is not terminal
    # was produced by legal action and it is not a chance outcome that 
    # never happens
    non_terminal = ~next_terminal * legal * (action_chance_probs >= 1e-8)
    
    
    # From [H(D), A1, A2] should select [H(D + 1)] 
    # nonzero() returns indices which are non zero in tuple 
    nonzeros = np.flatnonzero(non_terminal)
    next_states = jax.tree.map(lambda x: x[nonzeros], next_states)
    next_legals = next_legals[:, nonzeros]
    # print(f"Depth {depth}")
    # print(f"Next utilities {next_utilities.reshape(next_terminal.shape)}")
    # print(f"Next terminal {next_terminal}")
    # print(f"Next legals: {next_legals}")
    
    # This should be -1 everywhere, except the part where you have next history. Therey you go by terminal and just add 1
    next_history = (np.cumsum(non_terminal).reshape(non_terminal.shape) * non_terminal) - 1

    depth_behaviorals.append(np.asarray(pi))
    depth_infoset_map.append(np.asarray(infoset_map))

    if np.all(next_history < 0):
      return
    _tree_walk(next_states, next_legals, depth + 1)
  init_state, init_legals = game.initialize_structures()
  init_state_padded = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None, ...], init_state)
  _tree_walk(init_state_padded, init_legals[:, None, ...])
  return depth_infoset_map, depth_behaviorals

def policy_expected_value(game: JaxGame|DreamerModelGame, policy: tuple[list, list], eps=1e-5):
  """Computes expected return for all players while following given
  policy, represented as per depth infoset map and per depth behaviorals for
  each infoset. We operate with two player zero sum games, so will return
  p1_val, p2_val, where p2_val = -p1_val."""
  num_players = game.num_players()
  num_actions = game.num_distinct_actions()
  infoset_map, behaviorals = policy
  is_model_game = isinstance(game, DreamerModelGame)
  expected_return = 0
  #TODO: Think on how to vectorize this from DFS to BFS
  def _tree_walk(game_state: GameState, terminal, returns, log_reaches: np.ndarray, depth=0):
    if terminal:
      nonlocal expected_return
      reaches = np.exp(log_reaches)
      reach_weighted_return = returns * np.prod(reaches)
      expected_return +=  reach_weighted_return
      return
    if game.is_chance(game_state):
      num_chance_outcomes = int(game.state_valid_chance_outcomes(game_state)) if is_model_game else game.depth_chance_valid_outcomes(depth)
      next_game_states, next_terminal, next_rewards, next_legals, next_probs = unroll_chance_node(game, game_state, num_chance_outcomes)
      next_legals = np.asarray(next_legals)
      next_terminal = np.asarray(next_terminal)
      for i, next_legal in enumerate(next_legals):
        next_state = jax.tree.map(lambda x: x[i], next_game_states)
        #The chance reaches are last in reaches
        next_log_reaches = log_reaches + np.asarray((0, 0, np.log(next_probs[i])))
        _tree_walk(next_state, next_terminal[i], returns + next_rewards[i], next_log_reaches, depth= depth + 1)
      return

    _, p1_infoset, p2_infoset, _ = game.get_info(game_state)
    infosets = np.stack([p1_infoset, p2_infoset])
    joint_pi = []
    for pl, infoset in enumerate(infosets):
      idx = find_closest_index(infoset_map[depth][pl], infoset)
      # This is because the original game could have been
      # turn based, in which case one depth could contain only infosets for one player
      if idx == -1:
        joint_pi.append(np.eye(num_actions)[0])
        continue
      joint_pi.append(behaviorals[depth][pl][idx])
    joint_pi = np.asarray(joint_pi)
    player_reaches = np.exp(log_reaches[:-1])
    reachable_mask = player_reaches[..., None] * joint_pi >= eps
    actions = np.tile(np.arange(joint_pi.shape[-1]), (num_players,1)).reshape(joint_pi.shape)
    valid_actions = [actions[i][reachable_mask[i]] for i in range(num_players)]
    joint_actions = cartesian_product(*valid_actions)
    for a in joint_actions:
      both_reaches = np.take_along_axis(joint_pi, a[..., None], axis=-1).flatten()
      next_log_reaches = log_reaches + np.log(np.concatenate([both_reaches, np.ones(1)], axis=0))
      next_state, next_terminal, next_rewards, next_legals = game.apply_action(game_state, a)
      _tree_walk(next_state, next_terminal, returns + next_rewards, next_log_reaches, depth=depth+1)
  init_state, init_legals = game.initialize_structures()
  _tree_walk(init_state, False, 0, np.zeros(num_players + 1))
  return expected_return, -expected_return

def compare_policies(game: JaxGame| DreamerModelGame, given_pols: tuple[list, list], ref_pols: tuple[list, list], eps=0.05):
  """Compares two policies in every state of the game.
  Usually used to compare learned model policy with some reference policy.
  Assumes a two player game. Policies are expected to
  be supplied as a tuple of per depth infoset map and per depth
  behavior policies."""
  num_players = game.num_players()
  num_actions = game.num_distinct_actions()
  given_map, given_behaviorals = given_pols
  ref_map, ref_behaviorals = ref_pols
  is_model_game = isinstance(game, DreamerModelGame)

  def _tree_walk(game_state: GameState|ModelGameState, depth=0):
    if game.is_chance(game_state):
      num_chance_outcomes = int(game.state_valid_chance_outcomes(game_state)) if is_model_game else  game.depth_chance_valid_outcomes(depth)
      next_game_states, next_terminal, next_rewards, next_legals, next_probs = unroll_chance_node(game, game_state, num_chance_outcomes)
      next_legals = np.asarray(next_legals)
      next_terminal = np.asarray(next_terminal)
      for i, next_legal in enumerate(next_legals):
        if next_terminal[i]:
          continue
        next_state = jax.tree_util.tree_map(lambda x: x[i], next_game_states)
        _tree_walk(next_state, depth= depth + 1)
      return
    

    _, p1_infoset, p2_infoset, _ = game.get_info(game_state)
    infosets = np.stack([p1_infoset, p2_infoset])
    ref_joint_pi = []
    for pl, infoset in enumerate(infosets):
      ref_idx = find_closest_index(ref_map[depth][pl], infoset)
      # This is because the original game could have been
      # turn based, in which case one depth will contain only infosets for one player
      if ref_idx > -1:
        given_idx = find_closest_index(given_map[depth][pl], infoset)
        assert given_idx > -1, f"Given policy is missing infoset {infoset} for player {pl}, even though it is in reference policy."
        ref_pi = ref_behaviorals[depth][pl][ref_idx]
        ref_joint_pi.append(ref_pi)
        given_pi = given_behaviorals[depth][pl][given_idx]
        if np.sum((ref_pi - given_pi) ** 2) >= 1e-5:
          print(f"Given policy {given_pi} and reference policy {ref_pi} differ by more then {eps} for infoset {infoset}")
          print(f"Game state: {game_state}")
    num_pols = len(ref_joint_pi)
    if num_pols < num_players:
      for i in range(num_pols, num_players):
        #This is an invalid action at position 0 played deterministically
        ref_joint_pi.append(np.eye(num_actions)[0])
    ref_joint_pi = np.asarray(ref_joint_pi)
    reachable_mask = ref_joint_pi >= eps
    actions = np.tile(np.arange(ref_joint_pi.shape[-1]), (num_players,1)).reshape(ref_joint_pi.shape)
    valid_actions = [actions[i][reachable_mask[i]] for i in range(num_players)]
    joint_actions = cartesian_product(*valid_actions)
    for a in joint_actions:
      next_state, next_terminal, next_rewards, next_legals = game.apply_action(game_state, a)
      if next_terminal:
        continue
      _tree_walk(next_state, depth=depth+1)
  init_state, init_legals = game.initialize_structures()
  _tree_walk(init_state)
       

def model_best_response(model: DreamerMA, game: JaxGame | DreamerModelGame, custom_policy: tuple[list, list] = None):
  """Compute counterfactual best response policies for both players and their 
  respective values.Returned as br value of p2 against p1
  , br value of p1 against p2, p1_br_policy, p2_br_policy.
  In this 2p0s setting, the sum of the BR-values = NashConv.
  If custom policies are supplied then per_depth depth infoset maps 
  and behavioral policies are supplied and used instead"""
  p1_br = {}
  p2_br = {}
  game_actions = game.num_distinct_actions()
  checking_model = custom_policy is None


  depth_continuations = [] #[D, H(D), A(D), A(D)]
  depth_chance_probabilities = [] #[D, H(D), A(D), A(D)]
  depth_rewards = [] #[D, H(D), A(D), A(D)] only from player one perspective
  depth_actions = [] #[D, Pl, H(D), A(D)]
  depth_infoset_map = [] #[D, Pl, I]
  depth_infoset_legal = [] # [D, Pl, I, A(D)]
  depth_history_infoset = [] # [D, Pl, H(D)]
  depth_history_legal = [] # [D, Pl, H(D), A(D), A(D)]
  depth_is_chance = [] # [D, H(D)]
  depth_history_reaches = [] # [D, Pl + 1, H(D)], includes chance reaches
  depth_behavior_policy = [] #[D, Pl, H(D), A(D)]

  vectorized_get_info = jax.vmap(game.get_info, in_axes=(0), out_axes=(0, 0, 0, 0))
  #vmap over the internal per action dimension first 
  # and then over the outer H(D, dimension)
  vectorized_next_state = jax.vmap(jax.vmap(game.apply_action, in_axes=(None, 0), out_axes=(0, 0, 0, -2)), in_axes=(0, 0), out_axes=(0, 0, 0, -3))
  vectorized_is_chance = jax.vmap(game.is_chance, in_axes=0, out_axes=0)
  vectorized_chance_info = jax.vmap(game.get_outcomes_and_probs, in_axes=0, out_axes=(0, 0))
  if checking_model:
    ma_rssm = model.optimizer.model
    #vmap over the H(D) dimension first and then over the player dimension
    vectorized_get_policy = MARSSM.vmap_over_net(ma_rssm.actor, in_axes=[(0, 0), (0, 0)], out_axes=[0, 0])
    

  else:
    """TODO: Think on how to vectorize it"""
    pols_map, pols_behaviorals = custom_policy
    def vectorized_get_policy(depth, infosets, legals):
      behaviorals = []
      for pl, player_info in enumerate(zip(infosets, legals)):
        behaviorals.append([])
        pl_infosets, pl_legals = player_info
        for infoset, legal in zip(pl_infosets, pl_legals):
          infoset_idx = find_closest_index(pols_map[depth][pl], infoset)
          infoset_behavioral = pols_behaviorals[depth][pl][infoset_idx] if infoset_idx > -1 else legal / np.sum(legal)
          behaviorals[pl].append(infoset_behavioral)
      return np.asarray(behaviorals)

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

    state_tensors, p1_infosets, p2_infosets, public_states = vectorized_get_info(game_states)
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
    invalid_infoset = np.zeros_like(p1_infosets[0])
    curr_infoset = np.stack((p1_infosets, p2_infosets))
    curr_infoset = np.where(is_chance[None, :, None], invalid_infoset[None, None, ...], curr_infoset)
    infoset_map, infoset_legal, infosets, actions = create_infoset_map(curr_infoset, max_actions, legals)
    p1_legal_infoset, p2_legal_infoset = infoset_legal[0], infoset_legal[1]
    p1_legal_infoset, p2_legal_infoset = p1_legal_infoset > 0, p2_legal_infoset > 0
    
    p1_legal, p2_legal = legals[0], legals[1]
    legal = p1_legal[..., None] * p2_legal[..., None, :]

    pi = vectorized_get_policy(curr_infoset, legals_non_padded)[0] if checking_model else vectorized_get_policy(depth, curr_infoset, legals_non_padded)
    #jax.debug.breakpoint()
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
    
    
    p1_actions = np.reshape(np.tile(np.repeat(np.arange(max_actions), max_actions), curr_infoset.shape[1]), (curr_infoset.shape[1], -1))
    p2_actions = np.reshape(np.tile(np.tile(np.arange(max_actions), max_actions), curr_infoset.shape[1]), (curr_infoset.shape[1], -1))
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
    depth_infoset_map.append(infoset_map)
    depth_infoset_legal.append(infoset_legal)
    depth_history_legal.append(legal)
    depth_history_infoset.append(infosets)
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
  
  for d in range(len(depth_history_infoset) -1, -1, -1):
    p1_joint_action_value = np.where(depth_continuations[d] < 0, depth_rewards[d][0], state_value[0][depth_continuations[d]])
    p2_joint_action_value = np.where(depth_continuations[d] < 0, depth_rewards[d][1], state_value[1][depth_continuations[d]])
    
    p1_chance_weighted_value = np.sum(p1_joint_action_value * depth_chance_probabilities[d] * depth_history_legal[d], axis=(-1, -2))
    p2_chance_weighted_value = np.sum(p2_joint_action_value * depth_chance_probabilities[d] * depth_history_legal[d], axis=(-1, -2))
    
    p1_action_value = np.sum(p1_joint_action_value * depth_behavior_policy[d][1][..., None, :], -1)
    p2_action_value = np.sum(p2_joint_action_value * depth_behavior_policy[d][0][..., None], -2)
    
    p1_action_cf_value = p1_action_value * depth_history_reaches[d][1][..., None] * depth_history_reaches[d][2][..., None]
    p2_action_cf_value = p2_action_value * depth_history_reaches[d][0][..., None] * depth_history_reaches[d][2][..., None]
    
    
    p1_infoset_action_value = np.bincount(depth_actions[d][0].flatten(), p1_action_cf_value.flatten()).reshape(-1, depth_actions[d].shape[-1])
    p2_infoset_action_value = np.bincount(depth_actions[d][1].flatten(), p2_action_cf_value.flatten()).reshape(-1, depth_actions[d].shape[-1])
    
    p1_infoset_action_value_masked = np.where(depth_infoset_legal[d][0] == 1, p1_infoset_action_value, np.min(p1_infoset_action_value) - 1)
    p2_infoset_action_value_masked = np.where(depth_infoset_legal[d][1] == 1, p2_infoset_action_value, np.min(p2_infoset_action_value) - 1) 
    p1_br_action = np.argmax(p1_infoset_action_value_masked, -1)
    p2_br_action = np.argmax(p2_infoset_action_value_masked, -1)
    
    p1_history_br = p1_br_action[depth_history_infoset[d][0]]
    p2_history_br = p2_br_action[depth_history_infoset[d][1]]
    
    
    p1_br_policy = np.eye(p1_infoset_action_value.shape[-1])[p1_br_action]
    p2_br_policy = np.eye(p2_infoset_action_value.shape[-1])[p2_br_action]
    
    for i, infoset in enumerate(depth_infoset_map[d][0]):
      #All zeros infosets are invalid infosets
      if np.sum((infoset) **2) <= 1e-5:
        continue
      p1_br[stringify(infoset)] = p1_br_policy[i]
    for i, infoset in enumerate(depth_infoset_map[d][1]):
      #All zeros infosets are invalid infosets
      if np.sum((infoset) **2) <= 1e-5:
        continue
      p2_br[stringify(infoset)] = p2_br_policy[i]
    
    p1_history_value = np.squeeze(np.take_along_axis(p1_action_value, p1_history_br[..., None], 1), axis=-1)
    p2_history_value = np.squeeze(np.take_along_axis(p2_action_value, p2_history_br[..., None], 1), axis=-1)
    
    state_value = np.where(depth_is_chance[d][None, ...], np.stack((p1_chance_weighted_value, p2_chance_weighted_value), axis=0), np.stack((p1_history_value, p2_history_value), 0))
  state_value = state_value.squeeze(-1)
  return state_value[1], state_value[0], p1_br, p2_br


def nash_conv(model: DreamerMA, game: JaxGame | DreamerModelGame, custom_policy: tuple[list, list] = None):
  """Computes NashConv of the model, or the given policy

  Args:
      model (DreamerMA): The trained DreamerMA model.
      game (JaxGame | DreamerModelGame): Game that the model was trained on
      custom_policy (tuple[list, list], optional): An already extracted policy, represented
      as (per depth infoset map, per depth infoset behaviorals). If it is supplied, will compute
       NashConv of this policy rather than extracting it from the model. Defaults to None.
  """
  p2_br_val, p1_br_val, p1_br, p2_br = model_best_response(model, game, custom_policy=custom_policy)
  return p1_br_val + p2_br_val