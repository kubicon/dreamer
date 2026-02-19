import chex
import jax
import jax.numpy as jnp
from functools import partial
import numpy as np

import psutil
import time
import os

from games.jax_game import JaxGame, GameState
from dreamer_ma import DreamerMA



##################################################################
### MEMORY AND TIME USAGE DEBUGGING, taken from
### https://stackoverflow.com/questions/938733/total-memory-used-by-python-process
### accessed 09.02.2026
#################################################################
def elapsed_since(start):
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - start))


def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def track(func):
    def wrapper(*args, **kwargs):
        mem_before = get_process_memory()
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        mem_after = get_process_memory()
        print("{}: memory before: {:,}, after: {:,}, consumed: {:,}; exec time: {}".format(
            func.__name__,
            mem_before, mem_after, mem_after - mem_before,
            elapsed_time))
        return result
    return wrapper

#################################################################
### END OF CODE FROM https://stackoverflow.com/questions/938733/total-memory-used-by-python-process
#################################################################


def get_next_outcomes(model: DreamerMA, joint_stoch_state: chex.Array,
                      joint_recurrent_state: chex.Array, obs: chex.Array| np.ndarray,
                      threshold: float = 0.05) ->list:
  """Takes all possible stochastic state outcomes and then
  clusters them to the corresponding next outcome, based on 
  l2 distance between decoder and real iset"""
  


  obs = np.asarray(obs)
  num_next_obs = obs.shape[0]
  num_players ,num_classes, num_categories = joint_stoch_state.shape
  next_deters = [[] for _ in range(num_next_obs)]
  probs = [[] for _ in range(num_next_obs)]
  total_classes = num_players * num_classes

  #Flatten the stoch state over the players
  # to straightforwadly perform the stoch_state
  stoch_state = joint_stoch_state.reshape((-1, num_categories))
  deter_states = (stoch_state >= threshold).astype(int)
  class_indices, category_indices = np.nonzero(deter_states)
  per_class_valids = []
  for i in range(total_classes):
    single_class_indices = category_indices[class_indices == i]
    per_class_valids.append(single_class_indices)

  combinations = cartesian_product(*per_class_valids)
  for comb in combinations:
    prob = np.prod(stoch_state[np.arange(total_classes), comb])
    sampled_deter = jax.nn.one_hot(comb, stoch_state.shape[-1])
    #Reshape back to be per player deter state
    joint_deter = sampled_deter.reshape((num_players, num_classes, num_categories))
    next_closest_idx = get_closest_next_ma(model, joint_recurrent_state, joint_deter, obs)
    next_deters[next_closest_idx].append(joint_deter)
    probs[next_closest_idx].append(prob)
    
  return next_deters, probs


def get_closest_next_ma(model: DreamerMA, joint_recurrent_state, next_joint_deter, next_isets: np.ndarray):
  """Find the index of the closest next state
  this deterministic state corresponds to. With
  respect to distance between real isets of both players
  and decoded isets of both players."""
  if next_isets.ndim == 2 or next_isets.shape[0] == 1:
    return 0
  ma_rssm = model.optimizer.model
  decoded_obs = ma_rssm.get_decoder_all(joint_recurrent_state, next_joint_deter)
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
  next_states = jax.tree.map(lambda x: jnp.take_along_axis(x, jnp.expand_dims(valid, axis=range(1, x.ndim)), axis=0), next_states)
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

