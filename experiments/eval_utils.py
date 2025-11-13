import chex
import jax
import jax.numpy as jnp
from functools import partial
import numpy as np

from train_utils import get_reference_policy
from games.jax_game import JaxGame, GameState
from experiments.tree_view_utils import *
from dreamer import Dreamer
from dreamer_ma import DreamerMA





@chex.dataclass
class WalkCarry:
  legals: chex.Array
  game_state: GameState
  hidden_state: chex.Array
  stoch_state:chex.Array
  deter_state: chex.Array
  reward: chex.Array
  terminal: chex.Array
  after_chance: chex.Array

def get_next_outcomes(model: DreamerMA| Dreamer, stoch_state: chex.Array,
                      hidden_state: chex.Array, obs: chex.Array| np.ndarray,
                      threshold: float = 0.05, verbose = False) ->list:
  """Takes all possible stochastic state outcomes and then
  clusters them to the corresponding next outcome, based on 
  l2 distance between decoder and real iset"""
  


  obs = np.asarray(obs)
  num_next_obs = obs.shape[0]
  num_categoricals = stoch_state.shape[0]
  next_deters = [[] for _ in range(num_next_obs)]
  probs = [[] for _ in range(num_next_obs)]
  # Next obs has shape [num_next, num_players, iset_size] for multi-agent
  # and [num_next, obs_size] for single agent
  is_ma = obs.ndim == 3
  get_closes_func = get_closest_next_ma if is_ma else get_closest_next

  num_classes = stoch_state.shape[0]
  deter_states = (stoch_state >= threshold).astype(int)
  class_indices, category_indices = np.nonzero(deter_states)
  per_class_valids = []
  for i in range(num_classes):
    single_class_indices = category_indices[class_indices == i]
    per_class_valids.append(single_class_indices)

  combinations = cartesian_product(*per_class_valids)
  for comb in combinations:
    prob = stoch_state[np.arange(num_categoricals), comb]
    sampled_deter = jax.nn.one_hot(comb, stoch_state.shape[-1])
    next_closest_idx = get_closes_func(model, hidden_state, sampled_deter, obs)
    next_deters[next_closest_idx].append(sampled_deter)
    probs[next_closest_idx].append(prob)
    
  return next_deters, probs

def get_closest_next(model: Dreamer, hidden_state, next_deter, next_obs:np.ndarray):
  """Find the index of the closest next state
  this deterministic state corresponds to. With
  respect to distance between real obs and decoded obs"""
  if next_obs.ndim == 1 or next_obs.shape[0] == 1:
    return 0
  decoded_obs = model.get_decoder(model.optimizers.decoder_optimizer.model, hidden_state, next_deter)
  next_dists = np.sum((decoded_obs[None, ...] - next_obs) ** 2, axis=-1)
  next_closest  = np.argmin(next_dists)
  return next_closest

def get_closest_next_ma(model: DreamerMA, hidden_state, next_deter, next_isets: np.ndarray):
  """Find the index of the closest next state
  this deterministic state corresponds to. With
  respect to distance between real isets of both players
  and decoded isets of both players."""
  if next_isets.ndim == 2 or next_isets.shape[0] == 1:
    return 0
  p1_decoded_iset = model.get_decoder(model.optimizers.p1_decoder_optimizer.model, hidden_state, next_deter)
  p2_decoded_iset = model.get_decoder(model.optimizers.p2_decoder_optimizer.model, hidden_state, next_deter)
  decoded_obs = np.stack([p1_decoded_iset, p2_decoded_iset], axis=0)
  next_dists = np.sum((decoded_obs[None, ...] - next_isets) ** 2, axis=(-1, -2))
  next_closest  = np.argmin(next_dists)
  return next_closest


@partial(jax.jit, static_argnums=(0, 2))
def unroll_chance_node(game: JaxGame, game_state: GameState, num_chance_outcomes:int):
  """Unroll a state that is a chance node, into its outcomes.
  Returns next_states, next_terminal, rewards, next_legals, next_probs
  corresponding to the outcomes and stacked to have shape of [num_chance_outcomes, ...]"""
  outcomes, probs = game.get_outcomes_and_probs(game_state)
  vectorized_apply = jax.vmap(game.apply_action, in_axes=(None, 0), out_axes=(0, 0, 0, 0))
  next_states, next_terminal, rewards, next_legal = vectorized_apply(game_state, outcomes)
  valid = jnp.nonzero(probs >= 1e-5, size=num_chance_outcomes)[0]
  next_states = jax.tree_util.tree_map(lambda x: jnp.take_along_axis(x, jnp.expand_dims(valid, axis=range(1, x.ndim)), axis=0), next_states)
  next_terminal = jnp.take_along_axis(next_terminal, jnp.expand_dims(valid, axis=range(1, next_terminal.ndim)), axis=0)
  rewards = jnp.take_along_axis(rewards, jnp.expand_dims(valid, axis=range(1, rewards.ndim)), axis=0)
  next_legal = jnp.take_along_axis(next_legal, jnp.expand_dims(valid, axis=range(1, next_legal.ndim)), axis=0)
  next_probs = jnp.take_along_axis(probs, jnp.expand_dims(valid, axis=range(1, probs.ndim)), axis=0)
  return next_states, next_terminal, rewards, next_legal, next_probs

def cartesian_product(*arrays):
    """Implementation of cartesian product of 
    N 1D arrays. Taken from https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points"""
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def stringify(x)->str :
   x = np.asarray(x)
   return np.array2string(x)


def isets_close(iset1, iset2, tolerance=0.05):
   return np.sum((iset1 - iset2) ** 2) <= tolerance

def find_closest_index(iset_map: np.ndarray, ref_iset: np.ndarray, tolerance=0.05):
  """Finds the iset index in the given iset map
  based on closeness and returns it, or -1
  if no iset close enough within tolerance is found """
  #Edge case for an empty iset map
  if iset_map.shape == (0,):
    return -1
  iset_distance = np.sum((iset_map - ref_iset[None, ...]) ** 2, axis=-1)
  valid_isets = iset_distance <= tolerance
  # No valid iset was found
  if np.sum(valid_isets) == 0:
    return -1
  # else return the best fitting candidate
  return np.argmin(iset_distance)

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

def model_walk_test(model:Dreamer|DreamerMA,
                    all_outcome_check_fn,
                    one_outcome_check_fn,
                     difference_eps = 0.2, probability_eps = 0.05, probability_threshold=0.05
                     , verbose=False, visualise_tree = False):
  """Walk through the entire game tree in each state, check
  all learned outcomes where the individual components of the deterministic
  state have probability outcome over probability eps. Then perform a tree based
  expansion of all these model states and check whether the model learned well enough in each.
  In case of a chance node, the next model states are clustered to the particular
  outcome based on the closeness of their decoder produced output to the real observation.

  The outcome check functions are 
  supplied for each type of Dreamer separately. All outcome check
  function test all deterministic states that comprise of components with pbt >= probability threshold
  and expects call signature (model, carry, difference_eps, probability_threshold).
  Used for states that are not past chance nodes. They arent used as of yet,
  since the 

  The single outcome variant is used for states past chance nodes and expects call signature
   (model, carry, difference_eps),
   it should already take carry.deter_state as one of the clustered deterministic states.

   Finally, probability eps is used to control which outcomes under the reference policy to ignore
   (if the action had pbt <= probability eps, it will not be expanded). Also, used to check whether the model
   correctly learned the chance node distribution in pre-chance states
   , or almost deterministic distribution in non-chance states.

   Cannot handle more than 1 consecutive chance nodes (but note that 
   these can be represented as a single chance node.)
   Returns a numpy array of statistics of probablity of mistakes averaged over the states.
   For a single agent Dreamer ordered as obs_reconstruction, terminal, reward
   And for a multi agent Dreamer as iset1_reconstruction, iset2_reconstruction, terminal, reward, legal_actions.
  """
  is_ma = model.__class__ is DreamerMA
  def get_single_obs(state: GameState):
    return model.game.get_info(state)[1]
  def get_both_obs(state: GameState):
    _, p1_iset, p2_iset, _ = model.game.get_info(state)
    return jnp.stack([p1_iset, p2_iset], axis=0)
  get_obs_fn = get_both_obs if is_ma else get_single_obs
  #get_closest_deter_fn = get_closest_deter_ma if is_ma else get_closest_deter
  num_players = model.game.num_players()
  mistake_probs = 0
  visited = {}
  vectorized_get_obs = jax.vmap(get_both_obs, in_axes=(0), out_axes=(0))
  model_tree_root = Node("", data={"type": PAST_ACTION, "action": -1}) if visualise_tree else  None

  def get_stoch_from_prediction(logits: chex.Array):
    stoch_unfiltered = np.asarray(jax.nn.softmax(logits, axis=-1))
    stoch_unnormalized = stoch_unfiltered * (stoch_unfiltered >= probability_threshold)
    stoch = stoch_unnormalized / np.sum(stoch_unnormalized, axis=-1, keepdims=True)
    return stoch

  def _tree_walk(carry: WalkCarry, depth=0, reach_probability:float = 1.0, action_outcome_history = "",
                 subtree_parent: Node = None, outcome:int = -1, outcome_prob: float = 0, create_model_node:bool = False):
    nonlocal mistake_probs

    visited[action_outcome_history] = True
    if verbose:
      print(f"Checking state {carry.game_state}")
      print(f"Reach probs {reach_probability}")
    #print(f"Num visited states {num_visited_states}")
    # if carry.terminal:
    #   mistake_cum_probs = mistake_cum_probs + all_outcome_check_fn(model, carry, difference_eps, probability_threshold, verbose)
    #   return
    # else:

    state_mistake_probs, state_differences = one_outcome_check_fn(model, carry, difference_eps, verbose)
    parent = subtree_parent
    if visualise_tree and create_model_node:
      deter_path = parent.name + f"d{outcome}"
      #print(f"Storing node with id {deter_path} and parent {parent.name}")
      deter_node = Node(deter_path,
                      parent = parent,
                      data = {"differences" : state_differences, 
                              "prob": outcome_prob,
                              "type": MODEL_NODE})
      parent = deter_node
    mistake_probs  = mistake_probs + (state_mistake_probs * reach_probability)
    if carry.terminal:
      return
    pi = np.asarray(get_reference_policy(carry.game_state, carry.legals))
    #print(f"Policy: {pi}")
    pi_mask = pi >= probability_eps
    actions = np.tile(np.arange(pi.shape[-1]), (num_players,1)).reshape(pi.shape)
    if is_ma:
      valid_actions = [actions[i][pi_mask[i]] for i in range(num_players)]
      joint_actions = cartesian_product(*valid_actions)
    else:
      joint_actions = actions[pi_mask]
    #print(f"Joint actions: {joint_actions}")
    for a in joint_actions:
      #print(f"Applying action {a}")
      action_parent = parent
      if visualise_tree:
        action_path = parent.name + f"a{a}"
        action_node = Node(action_path,
                         parent = action_parent,
                         data = {"type": PAST_ACTION, "action": a})
        action_parent = action_node
      next_state, next_terminal, next_reward, next_legals = model.game.apply_action(carry.game_state, a)
      ai_oh = jax.nn.one_hot(a, carry.legals.shape[-1])
      next_hidden = model.get_next_hidden(model.optimizers.sequence_optimizer.model, carry.hidden_state, carry.deter_state, ai_oh)
      next_stoch_state = get_stoch_from_prediction(model.get_dynamics(model.optimizers.dynamics_optimizer.model, next_hidden))
            
      is_chance = model.game.is_chance(next_state)
      chance_outcomes = model.game.depth_chance_valid_outcomes(depth + 1)
      if is_chance:
        next_states, next_terminals, next_rewards, next_legals, next_probs = unroll_chance_node(model.game, next_state, chance_outcomes) 

        next_terminals = np.asarray(next_terminals)
        next_rewards = np.asarray(next_rewards)
        next_legals = np.asarray(next_legals)
      else:
        next_states =jax.tree.map(lambda x: x[None, ...], next_state)

        next_terminals = np.asarray(next_terminal)[None, ...]
        next_rewards = np.asarray(next_reward)[None, ...]
        next_legals = np.asarray(next_legals)[None, ...]
      next_obs = vectorized_get_obs(next_states)
      next_obs = np.asarray(next_obs)
      next_deters, next_probs= get_next_outcomes(model, next_stoch_state, next_hidden, next_obs, probability_eps, verbose=verbose)
      #print(f"Next deters: {next_deters}")
      for i in range(next_terminals.shape[0]):
        outcome_parent = action_parent
        next_terminal = next_terminals[i]
        next_reward = next_rewards[i]
        next_legal = next_legals[i]
        next_state = jax.tree_util.tree_map(lambda x: x[i], next_states)
        single_outcome_deters = next_deters[i]
        single_outcome_probs = next_probs[i]
        outcome_prob = np.sum(single_outcome_probs)
        if next_terminals.shape[0] > 1  and visualise_tree:
          outcome_path = parent.name + f"o{i}"
          outcome_node = Node(outcome_path,
                          parent=outcome_parent,
                          data = {"prob": outcome_prob, "type": PAST_CHANCE})
          outcome_parent = outcome_node
        num_deters = len(single_outcome_deters)
        for j, deter in enumerate(single_outcome_deters):
          new_carry = WalkCarry(legals= next_legal,
                                game_state = next_state,
                                hidden_state= next_hidden,
                                stoch_state=next_stoch_state,
                                deter_state=deter,
                                reward=next_reward,
                                terminal=next_terminal,
                                after_chance=is_chance)
          prob = jnp.prod(next_stoch_state[deter.astype(jnp.bool)])  
          _tree_walk(new_carry, depth = depth+ 1 + int(is_chance), subtree_parent = outcome_parent,
                     action_outcome_history= action_outcome_history + f"a{a}o{i}",
                     reach_probability= reach_probability * prob,
                     outcome = j, outcome_prob=single_outcome_probs[j] / outcome_prob,
                     create_model_node=True)
      #return
      #represented_next_stoch = jax.nn.softmax(model.optimizers.encoder_optimizer.model(next_hidden, real_obs), axis=-1)
      
  
  init_state, init_legals = model.game.initialize_structures()
  init_hidden = jnp.zeros(model.hidden_state_size)
  init_chance =  model.game.is_chance(init_state)
  if init_chance:
    chance_outcomes = model.game.depth_chance_valid_outcomes(0)
    next_states, next_terminals, next_rewards, next_legals, next_probs = unroll_chance_node(model.game, init_state, chance_outcomes)
        

    next_terminals = np.asarray(next_terminals)
    next_rewards = np.asarray(next_rewards)
    next_legals = np.asarray(next_legals)
    for i in range(next_terminals.shape[0]):
      outcome_parent = model_tree_root
      next_terminal = next_terminals[i]
      next_reward = next_rewards[i]
      next_legal = next_legals[i]
      next_state = jax.tree_util.tree_map(lambda x: x[i], next_states)
      #This is a special case handled differently than the
      # chance nodes from dynamics, which share a stochastic state
      # and we just pick the deterministic states most likely
      # beloning to the outcome.
      # The first prediction is posterior, so each outcome has its own stochastic state
      # because they are differentiated by the observations.
      init_obs = get_obs_fn(next_state)[None, ...]
      #print(f"Init obs for outcome {i}, is {init_obs}")
      # if verbose:
      #   print(f"Checking state {next_state}")
      init_stoch_state = get_stoch_from_prediction(model.get_encoder(model.optimizers.encoder_optimizer.model, init_hidden, init_obs))
      init_deters, init_probs = get_next_outcomes(model, init_stoch_state, init_hidden, init_obs, probability_eps, verbose)
      init_deters = init_deters[0]
      init_probs = init_probs[0]
      outcome_path = f"o{i}"
      outcome_prob = np.sum(init_probs)
      if visualise_tree:
        init_chance = Node(outcome_path,
                         parent= outcome_parent,
                         data = {"prob": outcome_prob, "type": PAST_CHANCE})
        outcome_parent = init_chance
      #num_init_deters = len(init_deters)
      for j, deter in enumerate(init_deters):
        init_carry = WalkCarry(legals= next_legal,
                              game_state = next_state,
                              hidden_state= init_hidden,
                              stoch_state=init_stoch_state,
                              deter_state=deter,
                              reward=next_reward,
                              terminal=next_terminal,
                              after_chance=init_chance)
        _tree_walk(init_carry, depth=1,
                   action_outcome_history=f"o{i}",
                   subtree_parent = outcome_parent,
                     outcome = j, outcome_prob=init_probs[j] / outcome_prob,
                     create_model_node=True)
    num_visited_states = len(visited)
    avg_mistake_probs = mistake_probs / num_visited_states
    avg_mistake_probs = np.minimum(avg_mistake_probs, 1.0)
    if visualise_tree:
      render_tree(model_tree_root, model)
    return avg_mistake_probs
  init_obs = get_obs_fn(init_state)[None, ...]
  init_stoch_state = get_stoch_from_prediction(model.get_encoder(model.optimizers.encoder_optimizer.model, init_hidden, init_obs))
  if verbose:
    print(f"Checking state {init_state}")
  init_deters, init_probs = get_next_outcomes(model, init_stoch_state, init_hidden, init_obs, probability_eps, verbose)
  init_deters = init_deters[0]
  init_probs = init_probs[0]
  #num_init_deters = len(init_deters)
  for i, deter in enumerate(init_deters):
    init_carry = WalkCarry(legals= init_legals,
                              game_state = init_state,
                              hidden_state= init_hidden,
                              stoch_state=init_stoch_state,
                              deter_state=deter,
                              reward=jnp.array(0),
                              terminal=jnp.array(False),
                              after_chance=jnp.array(False))
    prob = jnp.prod(init_stoch_state[deter.astype(jnp.bool)])
    _tree_walk(init_carry, subtree_parent = model_tree_root,  
                    reach_probability=prob,
                     outcome = i, outcome_prob=init_probs[i],
                     create_model_node=True)
  num_visited_states = len(visited)
  avg_mistake_probs = mistake_probs / num_visited_states
  avg_mistake_probs = np.minimum(avg_mistake_probs, 1.0)
  if visualise_tree:
    render_tree(model_tree_root, model)
  return avg_mistake_probs

