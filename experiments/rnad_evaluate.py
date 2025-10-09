from argparse import ArgumentParser
import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import os
import time
import matplotlib.pyplot as plt


from train_utils import load_model
from experiments.eval_utils import cartesian_product, stringify, find_closest_index, create_iset_map, unroll_chance_node
from games.jax_game import JaxGame, GameState
from rnad_dreamer import RNaDDreamer, RNaDConfig, JointOptimizers

parser = ArgumentParser()

parser.add_argument("--model_dir", type=str, default="trained_networks/rnad/goofspiel_3/seed99/network_seed42", help="Path to the directory of saved models")
parser.add_argument("--restore_step", type=int, default=10000, help="Saved step of the model to restore. If checking entire directory, -1 is also supported for all steps")

parser.add_argument("--seed", type=int, default=-1, help="Seed for the key to be used in gameplay. -1 for a random seed.")
parser.add_argument("--scale_factor", type=float, default=1.0, help="Scale factor to multiply all rewards by. Useful if the game implementation scaled rewards in a different way than traditional implementations."
                    "Then, this should be the inverse of the game scaling factor. For example, JaxLeduc divides all rewards by 13, so to get values appriopriately scaled as in literature, this should be set to 13.")

experiment_parsers = parser.add_subparsers(dest="experiment_type", required=True, help="Which experiment type to run. Currently available are: loaded"
                                          "evaluate best responses against particular loaded model, or all models in the directory if restore_step is -1" \
                                          "retrain : retrain a given model from a scratch and return a plot of its best response values with respect to iterations"
                                          "nash: evaluate expected values of the model, best response values against it and also of a saved reference nash equilibrium strategy.")

loaded_parser = experiment_parsers.add_parser(name="loaded", help="Evaluate best responses against particular loaded model, or all models in the directory if restore_step is -1")

retrain_parser = experiment_parsers.add_parser(name="retrain", help="Retrain a given model from a scratch and return a plot of its best response values with respect to iterations")
retrain_parser.add_argument("--neurd_steps", type=int, default=100, help="How many NeuRD steps to perform. Game solving steps between policy switches")
retrain_parser.add_argument("--policy_steps", type=int, default=10, help="How many policy switch steps of the algorithm to perform.")


nash_parser = experiment_parsers.add_parser(name="nash", help="Evaluate expected values of the model, best response values against it and also of a saved reference nash equilibrium strategy.")
nash_parser.add_argument("--nash_strategy_path", type=str, default="experiments/goofspiel_nash.pkl", help="Path to the saved nash strategy in pickle format. Must be formatted as a tuple of behavioral strategies per tree depth and iset map per tree_depth.")
   

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
        pi = model.get_policy_both(model.optimizers.rnad_optimizer.model, joint_iset, legals)
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

def extract_model_policy(model: RNaDDreamer)-> tuple[list, list]:
  """Extracts policies for the whole game from the RNaD model and 
  returns them as per depth
  iset map and behavioral policies."""
  depth_behaviorals = []
  depth_iset_map = []
  game = model.world_model.game
  game_actions = game.num_distinct_actions()

  vectorized_get_info = jax.vmap(game.get_info, in_axes=(0), out_axes=(0, 0, 0, 0))
  #vmap over the internal per action dimension first 
  # and then over the outer H(D, dimension)
  vectorized_next_state = jax.vmap(jax.vmap(game.apply_action, in_axes=(None, 0), out_axes=(0, 0, 0, -2)), in_axes=(0, 0), out_axes=(0, 0, 0, -3))
  vectorized_is_chance = jax.vmap(game.is_chance, in_axes=0, out_axes=0)
  vectorized_chance_info = jax.vmap(game.get_outcomes_and_probs, in_axes=0, out_axes=(0, 0))
  #vmap over the H(D) dimension first and then over the player dimension
  vectorized_get_policy = nnx.vmap(nnx.vmap(model._jit_get_policy, in_axes=(None, 0, 0), out_axes=0), in_axes=(None, 0, 0), out_axes=0)
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
    iset_map, iset_legal, isets, actions = create_iset_map(curr_iset, max_actions, legals_non_padded)
    
    p1_legal, p2_legal = legals[0], legals[1]
    legal = p1_legal[..., None] * p2_legal[..., None, :]

    pi = vectorized_get_policy(model.optimizers.rnad_optimizer.model, jnp.asarray(iset_map), jnp.asarray(iset_legal))
    
    
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
    next_terminal = np.reshape(next_terminal, legal.shape)
    

    # Next state will be expanded further if it is not terminal
    # was produced by legal action and it is not a chance outcome that 
    # never happens
    non_terminal = ~next_terminal * legal * (action_chance_probs >= 1e-8)
    
    
    # From [H(D), A1, A2] should select [H(D + 1)] 
    # nonzero() returns indices which are non zero in tuple 
    nonzeros = np.flatnonzero(non_terminal)
    next_states = jax.tree_util.tree_map(lambda x: x[nonzeros], next_states)
    next_legals = next_legals[:, nonzeros]
    
    # This should be -1 everywhere, except the part where you have next history. Therey you go by terminal and just add 1
    next_history = (np.cumsum(non_terminal).reshape(non_terminal.shape) * non_terminal) - 1

    depth_behaviorals.append(np.asarray(pi))
    depth_iset_map.append(np.asarray(iset_map))

    if np.all(next_history < 0):
      return
    _tree_walk(next_states, next_legals, depth + 1)
  init_state, init_legals = game.initialize_structures()
  init_state_padded = jax.tree_util.tree_map(lambda x: x[None, ...], init_state)
  _tree_walk(init_state_padded, init_legals[:, None, ...])
  return depth_iset_map, depth_behaviorals

def policy_expected_value(game: JaxGame, policy: tuple[list, list], eps=1e-5):
  """Computes expected return for all players while following given
  policy, represented as per depth iset map and per depth behaviorals for
  each iset. We operate with two player zero sum games, so will return
  p1_val, p2_val, where p2_val = -p1_val. FIXME: Reports wrong value 
  for leduc nash (the difference is roughly -0.08 real vs -0.0004 reported.
  Check why that happens!)."""
  num_players = game.num_players()
  num_actions = game.num_distinct_actions()
  iset_map, behaviorals = policy
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
      num_chance_outcomes = game.depth_chance_valid_outcomes(depth)
      next_game_states, next_terminal, next_rewards, next_legals, next_probs = unroll_chance_node(game, game_state, num_chance_outcomes)
      next_legals = np.asarray(next_legals)
      next_terminal = np.asarray(next_terminal)
      for i, next_legal in enumerate(next_legals):
        next_state = jax.tree_util.tree_map(lambda x: x[i], next_game_states)
        #The chance reaches are last in reaches
        next_log_reaches = log_reaches + np.asarray((0, 0, np.log(next_probs[i])))
        _tree_walk(next_state, next_terminal[i], returns + next_rewards[i], next_log_reaches, depth= depth + 1)
      return

    _, p1_iset, p2_iset, _ = game.get_info(game_state)
    isets = np.stack([p1_iset, p2_iset])
    joint_pi = []
    for pl, iset in enumerate(isets):
      idx = find_closest_index(iset_map[depth][pl], iset)
      # This is because the original game could have been
      # turn based, in which case one depth will contain only isets for one player
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

def compare_policies(game: JaxGame, given_pols: tuple[list, list], ref_pols: tuple[list, list], eps=0.05):
  """Compares two policies in every state of the game.
  Usually used to compare learned model policy with some reference policy.
  Assumes a two player game. Policies are expected to
  be supplied as a tuple of per depth iset map and per depth
  behavior policies."""
  num_players = game.num_players()
  num_actions = game.num_distinct_actions()
  given_map, given_behaviorals = given_pols
  ref_map, ref_behaviorals = ref_pols

  def _tree_walk(game_state: GameState, depth=0):
    if game.is_chance(game_state):
      num_chance_outcomes = game.depth_chance_valid_outcomes(depth)
      next_game_states, next_terminal, next_rewards, next_legals, next_probs = unroll_chance_node(game, game_state, num_chance_outcomes)
      next_legals = np.asarray(next_legals)
      next_terminal = np.asarray(next_terminal)
      for i, next_legal in enumerate(next_legals):
        if next_terminal[i]:
          continue
        next_state = jax.tree_util.tree_map(lambda x: x[i], next_game_states)
        _tree_walk(next_state, depth= depth + 1)
      return
    

    _, p1_iset, p2_iset, _ = game.get_info(game_state)
    isets = np.stack([p1_iset, p2_iset])
    ref_joint_pi = []
    for pl, iset in enumerate(isets):
      ref_idx = find_closest_index(ref_map[depth][pl], iset)
      # This is because the original game could have been
      # turn based, in which case one depth will contain only isets for one player
      if ref_idx > -1:
        given_idx = find_closest_index(given_map[depth][pl], iset)
        assert given_idx > -1, f"Given policy is missing iset {iset} for player {pl}, even though it is in reference policy."
        ref_pi = ref_behaviorals[depth][pl][ref_idx]
        ref_joint_pi.append(ref_pi)
        given_pi = given_behaviorals[depth][pl][given_idx]
        if np.sum((ref_pi - given_pi) ** 2) >= 1e-5:
          print(f"Given policy {given_pi} and reference policy {ref_pi} differ by more then {eps} for iset {iset}")
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
       

def model_best_response(model: RNaDDreamer, custom_policy: tuple[list, list] = None):
  """Compute counterfactual best response policies for both players and their 
  respective values.Returned as br value of p2 against p1
  , br value of p1 against p2, p1_br_policy, p2_br_policy.
  In this 2p0s setting, the sum of the BR-values = NashConv.
  If custom policies are supplied then per_depth depth iset maps 
  and behavioral policies are supplied and used instead"""
  game = model.world_model.game
  p1_br = {}
  p2_br = {}
  game_actions = game.num_distinct_actions()
  checking_model = custom_policy is None


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
  vectorized_chance_info = jax.vmap(game.get_outcomes_and_probs, in_axes=0, out_axes=(0, 0))
  if checking_model:
    #vmap over the H(D) dimension first and then over the player dimension
    vectorized_get_policy = nnx.vmap(nnx.vmap(model._jit_get_policy, in_axes=(None, 0, 0), out_axes=0), in_axes=(None, 0, 0), out_axes=0)

  else:
    """TODO: Think on how to vectorize it"""
    pols_map, pols_behaviorals = custom_policy
    def vectorized_get_policy(depth, isets, legals):
      behaviorals = []
      for pl, player_info in enumerate(zip(isets, legals)):
        behaviorals.append([])
        pl_isets, pl_legals = player_info
        for iset, legal in zip(pl_isets, pl_legals):
          iset_idx = find_closest_index(pols_map[depth][pl], iset)
          iset_behavioral = pols_behaviorals[depth][pl][iset_idx] if iset_idx > -1 else legal / np.sum(legal)
          behaviorals[pl].append(iset_behavioral)
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

    pi = vectorized_get_policy(model.optimizers.rnad_optimizer.model, curr_iset, legals_non_padded) if checking_model else vectorized_get_policy(depth, curr_iset, legals_non_padded)
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
    
    p1_chance_weighted_value = np.sum(p1_joint_action_value * depth_chance_probabilities[d] * depth_history_legal[d], axis=(-1, -2))
    p2_chance_weighted_value = np.sum(p2_joint_action_value * depth_chance_probabilities[d] * depth_history_legal[d], axis=(-1, -2))
    
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
    
    p1_history_value = np.squeeze(np.take_along_axis(p1_action_value, p1_history_br[..., None], 1), axis=-1)
    p2_history_value = np.squeeze(np.take_along_axis(p2_action_value, p2_history_br[..., None], 1), axis=-1)
    
    state_value = np.where(depth_is_chance[d][None, ...], np.stack((p1_chance_weighted_value, p2_chance_weighted_value), axis=0), np.stack((p1_history_value, p2_history_value), 0))
  state_value = state_value.squeeze(-1)
  return state_value[1], state_value[0], p1_br, p2_br
   


def test_loaded(args):
  # print(f"Evaluating model from {args.model_dir} at step {args.restore_step} with seed {args.seed}")
  # model_path = args.model_dir
  # if not model_path.startswith("/"):
  #   model_path = os.getcwd() + "/" + model_path
  # model_path = model_path + f"/step_{args.restore_step}.pkl"
  # if not os.path.exists(model_path):
  #   raise FileNotFoundError(f"Model file {model_path} does not exist.")

  # model = load_model(model_path)
  # assert isinstance(model, RNaDDreamer), f"Loaded model should be an instance of RNaDDreamer not {model.__class__}"
  # #The JAX version of Leduc divides all rewards by the max
  # # bet amount 13, but the saved values do not account for it. Just rescale it for correspondence
  # scale_factor = 13 if model.world_model.game.game_name() == "leduc" else 1
  # #model_walk_deterministic(model)
  # p2_br_val, p1_br_val, p1_br, p2_br = model_best_response(model)
  # p1_br_val, p2_br_val = scale_factor * p1_br_val, scale_factor * p2_br_val
  # print(f"P2 best response value against p1: {p2_br_val}")
  # print(f"P1 best response value against p2 {p1_br_val}")
  model_dir = args.model_dir
  p1_exploitabilities = []
  p2_exploitabilities = []
  steps = []
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
  plot_subdir_str = "rnad_only"
  for filename in os.listdir(model_dir):
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
      assert isinstance(model, RNaDDreamer), f"The given model should be an instance of RNaDDreamer, not {model.__class__}"
      if isinstance(model.optimizers, JointOptimizers):
        plot_subdir_str = "compound"
      first=False
    else:
      temp_model = load_model(model_path)
      assert isinstance(temp_model, RNaDDreamer), f"The given model should be an instance of RNaDDreamer, not {model.__class__}"
      #TODO: Updating this way still forces retracing of get_info and
      # initialize_structures of the game, since it is called in init. In general
      # we just need the state of the optimizers object from the model
      # and the rest of the operations are redundant.
      nnx.update(model.optimizers, nnx.split(temp_model.optimizers)[1])
      model.world_model.learner_steps = temp_model.world_model.learner_steps
      #model.optimizers = model.update_nnx(model.optimizers, nnx.split(temp_model.optimizers)[1])
    print(f"Restored model from {model_path}")
    if model.config.use_learned_model:
      print(f"Model is learned on Dreamer, that took {model.world_model.learner_steps} steps.")
    else:
      print(f"Model is trained on the original game.")
    p2_br_val, p1_br_val, p1_br, p2_br = model_best_response(model)
    p1_br_val, p2_br_val = args.scale_factor * p1_br_val, args.scale_factor * p2_br_val
    p1_exploitabilities.append(p2_br_val)
    p2_exploitabilities.append(p1_br_val)
    steps.append(step)
  print("Ended evaluation")
  print(f"Evaluation took {time.time() - start_time:.2f} seconds.")
  #profiler.stop()
  #print(profiler.output_text(color=True, unicode=True))
  if len(steps) == 0:
    raise FileNotFoundError(f"Model directory {model_dir} and restore step {args.restore_step}. Did not find any file. Make sure"
                            "the directory contains a file in a form of step_restore_step.pkl, "
                            "where restore_step is either the specified number, or arbitrary integer if -1.")
  p1_exploitabilities = np.asarray(p1_exploitabilities)
  steps = np.asarray(steps)
  p2_exploitabilities = np.asarray(p2_exploitabilities)
  sort_indices = np.argsort(steps)
  sorted_p1_exploitabilities = p1_exploitabilities[sort_indices]
  sorted_steps = steps[sort_indices]
  sorted_p2_exploitabilities = p2_exploitabilities[sort_indices]

  fig, ax = plt.subplots()
  ax.plot(sorted_steps, sorted_p1_exploitabilities, label="Player 1 exploitability")
  ax.plot(sorted_steps, sorted_p2_exploitabilities, label="Player 2 exploitability")
  ax.legend()
  ax.set_xlabel("Training step")
  ax.set_ylabel("Exploitability")
  ax.set_title("Exploitability of RNaD Dreamer")
  empty = ""
  game_params = model.world_model.game.params_dict()
  params_str = f'{empty.join(f"_{value}" for key, value in game_params.items())}'
  plt_dir = f"plots/exploitabilities/{plot_subdir_str}"
  if not os.path.exists(plt_dir):
    os.makedirs(plt_dir)
  plt.savefig(f"{plt_dir}/{model.world_model.game.game_name()}{params_str}.pdf")

def test_retrain(args):
  """TODO: Redo this to support the joint training. 
  Also, allow full entropy schedule. Probably, instead of a retrain method
  it would be better to just evaluate a saved directory and plot the br_values."""
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
  neurd_steps = args.neurd_steps
  policy_steps = args.policy_steps
  new_config = RNaDConfig(
      batch_size=config.batch_size,
      seed=config.seed,
      use_learned_model = config.use_learned_model,

      eta=config.eta,
      sampling_epsilon=config.sampling_epsilon,
      state_sample_threshold=config.state_sample_threshold,

      # Entropy schedule parameters
      entropy_schedule_size = (neurd_steps, neurd_steps),
      entropy_schedule_repeats = (policy_steps, 1),
      
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
  print(f"Config {config}")
  clean_model = RNaDDreamer(model.world_model, new_config)
  print(f"Step {clean_model.learner_steps}")
  print(f"Policy switch step {clean_model.policy_switch_steps}")
  p2_br_val, p1_br_val, p1_br, p2_br = model_best_response(clean_model)
  print(f"P2 best response value against p1: {p2_br_val}")
  print(f"P1 best response value against p2 {p1_br_val}")
  for i in range(policy_steps):
    for j in range(neurd_steps):
       clean_model.step()
    print(f"Step {clean_model.learner_steps}")
    print(f"Policy switch step {clean_model.policy_switch_steps}")
    #model_walk_deterministic(clean_model, seed)
    p2_br_val, p1_br_val, p1_br, p2_br = model_best_response(clean_model)
    p1_br_val, p2_br_val = args.scale_factor * p1_br_val, args.scale_factor * p2_br_val
    print(f"P2 best response value against p1: {p2_br_val}")
    print(f"P1 best response value against p2 {p1_br_val}")
    p1_exploitabilities.append(p2_br_val)
    p2_exploitabilities.append(p1_br_val)
    #breakpoint()
  p1_exploitabilities = np.asarray(p1_exploitabilities)
  p2_exploitabilities = np.asarray(p2_exploitabilities)
  policy_switch_steps = np.arange(policy_steps)
  plt.plot(policy_switch_steps, p1_exploitabilities, label="Player 1 exploitability")
  plt.plot(policy_switch_steps, p2_exploitabilities, label="Player 2 exploitability")
  plt.legend()
  plt.savefig(f"plots/br_values/{model.world_model.game.game_name()}/neurd_steps{neurd_steps}.pdf")

def test_nash(args, saved_nash_path: str):
  model_path = args.model_dir
  if not model_path.startswith("/"):
    model_path = os.getcwd() + "/" + model_path
  model_path = model_path + f"/step_{args.restore_step}.pkl"
  if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} does not exist.")
  
  nash_path= saved_nash_path
  if not nash_path.startswith("/"):
    nash_path = os.getcwd() + "/" + nash_path
  if not os.path.exists(nash_path):
    raise FileNotFoundError(f"Nash policy file {nash_path} does not exist.")
  
  print(f"Evaluating policy of model loaded from {model_path} against nash policy loaded from {saved_nash_path}")

  model = load_model(model_path)
  assert isinstance(model, RNaDDreamer), f"Loaded model should be an instance of RNaDDreamer not {model.__class__}"
  if not model.config.use_learned_model:
    print(f"The model is learned on the real game {model.world_model.game.game_name()}")
  
  p1_nash_val, p2_nash_val, nash_iset_map, nash_behaviorals = load_model(nash_path)
  print(f"Loaded nash policies of game with game value {p1_nash_val} (from player 1 perspective)")
  model_map, model_behaviorals = extract_model_policy(model)
  found_p1_nash, found_p2_nash = policy_expected_value(model.world_model.game, (nash_iset_map, nash_behaviorals), eps=1e-5)
  found_p1_nash, found_p2_nash = args.scale_factor * found_p1_nash, args.scale_factor * found_p2_nash
  print(f"Found nash values: {found_p1_nash} {found_p2_nash}")
  assert np.isclose(found_p1_nash, p1_nash_val, atol=1e-5), f"Found nash value {found_p1_nash} and saved nash value {p1_nash_val} for player 1 differ!"
  assert np.isclose(found_p1_nash, p1_nash_val, atol=1e-5), f"Found nash value {found_p2_nash} and saved nash value {p2_nash_val} for player 2 differ!"
  model_p1_val, model_p2_val = policy_expected_value(model.world_model.game, (model_map, model_behaviorals))
  model_p1_val, model_p2_val = args.scale_factor * model_p1_val, args.scale_factor * model_p2_val
  print(f"Model values {model_p1_val}, {model_p2_val}")
  p2_br_val, p1_br_val, p1_br, p2_br = model_best_response(model)
  p1_br_val, p2_br_val = args.scale_factor * p1_br_val, args.scale_factor * p2_br_val
  print(f"P2 best response value against p1: {p2_br_val}")
  print(f"P1 best response value against p2 {p1_br_val}")
  #compare_policies(model.world_model.game, (model_map, model_behaviorals), (nash_iset_map, nash_behaviorals))
        
  

def main():
  args = parser.parse_args()
  if args.experiment_type == "retrain":
    test_retrain(args)
  elif args.experiment_type == "nash":
    test_nash(args, saved_nash_path=args.nash_strategy_path)
  else:
    test_loaded(args)
  

if __name__ == "__main__":
  main()