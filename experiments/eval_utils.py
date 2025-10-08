import chex
import jax
import jax.numpy as jnp
from functools import partial
import numpy as np

from train_utils import get_reference_policy
from games.jax_game import JaxGame, GameState
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

def check_outcomes(stoch_state: chex.Array, is_chance:bool, num_chance_outcomes:int,  eps:float) ->list:
  """Checks validity of learned distributions. In non chance levels, all categoricals
  should be deterministic. In chance level, checks whether there is a categorical
  ,that correctly models the chance outcome distribution .
  Returns either the deterministic state corresponding to the max probability outcome (as 1 element list),
   or, in chance level a list of all deterministic states corresponding to the chance outcomes.
    TODO: This does not catch all cases. For example
  for 4 outcomes [[0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0]] would also be a valid learned result.
  But, this assumes that the outcomes are encoded in one distribution and the rest
  are (almost) deterministic. 
  
  TODO: So far this assumes uniform distribution over the legal chance outcomes. If we switch
  to non-uniform, we should also check whether the correct outcome corresponds to the 
  correct probability."""
  
  #repr_stoch_state = jax.nn.softmax(stoch_state, axis=-1)

  distribution_mismatch = 0
  max_probs = jnp.max(stoch_state, axis=-1)
  max_indices = jnp.argmax(stoch_state, axis=-1)
  #repr_max_probs = jnp.max(repr_stoch_state, axis=-1)

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
      distribution_mismatch = 1
  #     print(f"Stochastic state differs from a stochastic uniform by more than {eps}")
  #     print(f"Stochastic state  max probs {uniform_categorical}")
  #     #print(f"Represented (posterior) stochastic state two max probs {repr_two_max_probs}")
  else:
    if jnp.max(jnp.abs(1 - max_probs)) >= eps:
      distribution_mismatch = 1
  #     print(f"Stochastic state differs from deterministic more than {eps}")
  #     print(f"Stochastic state max probs {max_probs}")
      #print(f"Represented (posterior) stochastic state max probs {repr_max_probs}")

  chance_max_dets = jax.nn.one_hot(uniform_categorical_indices, stoch_state.shape[-1], axis=-1)
  deter_state = jax.nn.one_hot(max_indices, stoch_state.shape[-1], axis=-1)
  def _make_det_from_chance(chance_det, chance_idx, argmax_state):
    return jnp.concatenate([argmax_state[:chance_idx, :], chance_det[None, ...], argmax_state[chance_idx + 1:, :]], axis=0)
  next_deters = [_make_det_from_chance(det, chance_dist_idx, deter_state) for det in chance_max_dets] if is_chance else [deter_state]
  return next_deters, distribution_mismatch

def get_closest_deter(model: Dreamer, hidden_state, deters, state: GameState):
  """Find the deterministic state of the possible outcomes that is the best fit
  to the state based on decoder. Single agent version"""
  if len(deters) == 1:
    return deters
  min_dist = jnp.inf
  closest_deter = None
  real_obs = model.game.get_info(state)[1]
  for next_deter in deters:
    decoded_obs = model.get_decoder(model.optimizers.decoder_optimizer.model, hidden_state, next_deter)
    dist = jnp.sum((real_obs - decoded_obs) ** 2)
    if dist < min_dist:
      min_dist = dist
      closest_deter = next_deter
  return closest_deter

def get_closest_deter_ma(model: DreamerMA, hidden_state, deters, state: GameState):
  """Find the deterministic state of the possible outcomes that is the best fit
  to the state based on decoder. Multi-agent version."""
  if len(deters) == 1:
    return deters
  min_dist = np.inf
  closest_deter = None
  _, p1_iset, p2_iset, _ = model.game.get_info(state)
  real_obs = np.stack([p1_iset, p2_iset], axis=0)
  for next_deter in deters:
    p1_decoded_iset = model.get_decoder(model.optimizers.p1_decoder_optimizer.model, hidden_state, next_deter)
    p2_decoded_iset = model.get_decoder(model.optimizers.p2_decoder_optimizer.model, hidden_state, next_deter)
    decoded_obs = np.stack([p1_decoded_iset, p2_decoded_iset], axis=0)
    dist = np.sum((real_obs - decoded_obs) ** 2)
    if dist < min_dist:
      min_dist = dist
      closest_deter = next_deter
  return closest_deter


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
                     difference_eps = 0.2, probability_eps = 0.05, probability_threshold=0.05):
  """Walk through the entire game tree in each state, check
  whether all the Dreamer learned states 
  with a probability above certain threshold represent
  the original state accurately.

  The outcome check functions are 
  supplied for each type of Dreamer separately. All outcome check
  function test all deterministic states that comprise of components with pbt >= probability threshold
  and expects call signature (model, carry, difference_eps, probability_threshold).
  Used for states that are not past chance nodes.

  The single outcome variant is used for states past chance nodes and expects call signature
   (model, carry, difference_eps),
   it should already take carry.deter_state as the deterministic state corresponding to the chance outcome
   (through get_closest_outcome).

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
    return np.stack([p1_iset, p2_iset], axis=0)
  get_obs_fn = get_both_obs if is_ma else get_single_obs
  get_closest_deter_fn = get_closest_deter_ma if is_ma else get_closest_deter
  num_players = model.game.num_players()
  mistake_cum_probs = 0
  num_visited_states = 0
  distribution_mismatches = 0

  def get_stoch_from_prediction(logits: chex.Array):
    stoch_unfiltered = np.asarray(jax.nn.softmax(logits, axis=-1))
    stoch_unnormalized = stoch_unfiltered * (stoch_unfiltered >= probability_threshold)
    stoch = stoch_unnormalized / np.sum(stoch_unnormalized, axis=-1, keepdims=True)
    return stoch

  def _tree_walk(carry: WalkCarry, depth=0):
    nonlocal num_visited_states
    nonlocal mistake_cum_probs
    nonlocal distribution_mismatches
    num_visited_states += 1
    #print(f"Num visited states {num_visited_states}")
    if carry.after_chance:
      mistake_cum_probs  = mistake_cum_probs + one_outcome_check_fn(model, carry, difference_eps)
    else:
      mistake_cum_probs = mistake_cum_probs + all_outcome_check_fn(model, carry, difference_eps, probability_threshold)
    if carry.terminal:
      return
    pi = np.asarray(get_reference_policy(carry.game_state, carry.legals))
    pi_mask = pi >= probability_eps
    actions = np.tile(np.arange(pi.shape[-1]), (num_players,1)).reshape(pi.shape)
    if is_ma:
      valid_actions = [actions[i][pi_mask[i]] for i in range(num_players)]
      joint_actions = cartesian_product(*valid_actions)
    else:
      joint_actions = actions[pi_mask]
    for a in joint_actions:
      next_state, next_terminal, next_reward, next_legals = model.game.apply_action(carry.game_state, a)
      ai_oh = jax.nn.one_hot(a, carry.legals.shape[-1])
      next_hidden = model.get_next_hidden(model.optimizers.sequence_optimizer.model, carry.hidden_state, carry.deter_state, ai_oh)
      next_stoch_state = get_stoch_from_prediction(model.get_dynamics(model.optimizers.dynamics_optimizer.model, next_hidden))
            
      is_chance = model.game.is_chance(next_state)
      chance_outcomes = model.game.depth_chance_valid_outcomes(depth + 1)
      next_deters, dist_mismatch = check_outcomes(next_stoch_state, is_chance, chance_outcomes, probability_eps) 
      distribution_mismatches = distribution_mismatches + dist_mismatch
      if is_chance:
        next_states, next_terminals, next_rewards, next_legals, next_probs = unroll_chance_node(model.game, next_state, chance_outcomes)
        

        next_terminals = np.asarray(next_terminals)
        next_rewards = np.asarray(next_rewards)
        next_legals = np.asarray(next_legals)
          
        for i in range(next_terminals.shape[0]):
          next_terminal = next_terminals[i]
          next_reward = next_rewards[i]
          next_legal = next_legals[i]
          next_state = jax.tree_util.tree_map(lambda x: x[i], next_states)
          closest_deter = get_closest_deter_fn(model, next_hidden, next_deters, next_state)
          new_carry = WalkCarry(legals= next_legal,
                                game_state = next_state,
                                hidden_state= next_hidden,
                                stoch_state=next_stoch_state,
                                deter_state=closest_deter,
                                reward=next_reward,
                                terminal=next_terminal,
                                after_chance=is_chance)    
          _tree_walk(new_carry, depth+2)
        return
      new_carry = WalkCarry(legals= next_legals,
                            game_state = next_state,
                            hidden_state= next_hidden,
                            stoch_state=next_stoch_state,
                            deter_state=next_deters[0],
                            reward=next_reward,
                            terminal=next_terminal,
                            after_chance=is_chance)     
      _tree_walk(new_carry, depth+1)
      #represented_next_stoch = jax.nn.softmax(model.optimizers.encoder_optimizer.model(next_hidden, real_obs), axis=-1)
      
  
  init_state, init_legals = model.game.initialize_structures()
  init_hidden = jnp.zeros(model.config.hidden_state_size)
  init_chance =  model.game.is_chance(init_state)
  if init_chance:
    chance_outcomes = model.game.depth_chance_valid_outcomes(0)
    next_states, next_terminals, next_rewards, next_legals, next_probs = unroll_chance_node(model.game, init_state, chance_outcomes)
        

    next_terminals = np.asarray(next_terminals)
    next_rewards = np.asarray(next_rewards)
    next_legals = np.asarray(next_legals)
    stoch_states = []
    deters = []
    for i in range(next_terminals.shape[0]):
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
      init_obs = get_obs_fn(next_state)
      #print(f"Init obs for outcome {i}, is {init_obs}")
      init_stoch_state = get_stoch_from_prediction(model.get_encoder(model.optimizers.encoder_optimizer.model, init_hidden, init_obs))
      init_deter, dist_mismatch = check_outcomes(init_stoch_state, False, chance_outcomes, probability_eps)
      init_deter = init_deter[0]
      distribution_mismatches += dist_mismatch
      init_carry = WalkCarry(legals= next_legal,
                            game_state = next_state,
                            hidden_state= init_hidden,
                            stoch_state=init_stoch_state,
                            deter_state=init_deter,
                            reward=next_reward,
                            terminal=next_terminal,
                            after_chance=init_chance)
      stoch_states.append(init_stoch_state)
      deters.append(init_deter)
      _tree_walk(init_carry, depth=1)
    avg_mistake_probs = mistake_cum_probs / num_visited_states
    avg_distribution_mismatches = distribution_mismatches / num_visited_states
    return avg_mistake_probs, avg_distribution_mismatches
  init_obs = get_obs_fn(init_state)
  init_stoch_state = get_stoch_from_prediction(model.get_encoder(model.optimizers.encoder_optimizer.model, init_hidden, init_obs))
  init_deter, dist_mismatch = check_outcomes(init_stoch_state, False, model.game.depth_chance_valid_outcomes(0), probability_eps)
  init_deter = init_deter[0]
  distribution_mismatches += dist_mismatch
  init_carry = WalkCarry(legals= init_legals,
                            game_state = init_state,
                            hidden_state= init_hidden,
                            stoch_state=init_stoch_state,
                            deter_state=init_deter,
                            reward=jnp.array(0),
                            terminal=jnp.array(False),
                            after_chance=jnp.array(False)) 
  _tree_walk(init_carry)
  avg_mistake_probs = mistake_cum_probs / num_visited_states
  avg_distribution_mismatches = distribution_mismatches / num_visited_states
  return avg_mistake_probs, avg_distribution_mismatches

