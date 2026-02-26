import chex
import jax
import jax.numpy as jnp
from functools import partial
import numpy as np

import psutil
import resource
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

def get_peak_memory():
    # resource.getrusage returns ru_maxrss in kilobytes on Linux systems. 
    # We multiply by 1024 to convert it to bytes to match psutil.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def track(func):
    def wrapper(*args, **kwargs):
        mem_before = get_process_memory()
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        mem_after = get_process_memory()
        peak_memory = get_peak_memory()
        print("{}: memory before: {:,}, after: {:,}, consumed: {:,}, peak: {:,}; exec time: {}".format(
            func.__name__,
            mem_before, mem_after, mem_after - mem_before, peak_memory,
            elapsed_time))
        return result
    return wrapper

#################################################################
### END OF CODE FROM https://stackoverflow.com/questions/938733/total-memory-used-by-python-process
#################################################################


def get_next_outcomes(model: DreamerMA, stoch_state: chex.Array,
                      recurrent_state: chex.Array, obs: chex.Array| np.ndarray,
                      threshold: float = 0.05) ->list:
  """Takes all possible stochastic state outcomes and then
  clusters them to the corresponding next outcome, based on 
  l2 distance between decoder and real infoset"""
  


  obs = np.asarray(obs)
  num_next_obs = obs.shape[0]
  num_classes, num_categories = stoch_state.shape
  next_deters = [[] for _ in range(num_next_obs)]
  probs = [[] for _ in range(num_next_obs)]
  deter_states = (stoch_state >= threshold).astype(int)
  class_indices, category_indices = np.nonzero(deter_states)
  per_class_valids = []
  for i in range(num_classes):
    single_class_indices = category_indices[class_indices == i]
    per_class_valids.append(single_class_indices)

  combinations = cartesian_product(*per_class_valids)
  for comb in combinations:
    prob = np.prod(stoch_state[np.arange(num_classes), comb])
    sampled_deter = jax.nn.one_hot(comb, stoch_state.shape[-1])
    next_closest_idx = get_closest_next_ma(model, recurrent_state, sampled_deter, obs)
    next_deters[next_closest_idx].append(sampled_deter)
    probs[next_closest_idx].append(prob)
    
  return next_deters, probs


def get_closest_next_ma(model: DreamerMA, recurrent_state, next_deter, next_infosets: np.ndarray):
  """Find the index of the closest next state
  this deterministic state corresponds to. With
  respect to distance between real infosets of both players
  and decoded infosets of both players."""
  if next_infosets.ndim == 2 or next_infosets.shape[0] == 1:
    return 0
  ma_rssm = model.optimizer.model
  decoded_obs = ma_rssm.get_decoder(recurrent_state, next_deter)
  next_dists = np.sum((decoded_obs[None, ...] - next_infosets) ** 2, axis=(-1, -2))
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


def infosets_close(infoset1, infoset2, tolerance=0.05):
   return np.sum((infoset1 - infoset2) ** 2) <= tolerance

def find_closest_index(infoset_map: np.ndarray, ref_infoset: np.ndarray, tolerance=0.05):
  """Finds the infoset index in the given infoset map
  based on closeness and returns it, or -1
  if no infoset close enough within tolerance is found """
  #Edge case for an empty infoset map
  if infoset_map.shape == (0,):
    return -1
  infoset_distance = np.sum((infoset_map - ref_infoset[None, ...]) ** 2, axis=-1)
  valid_infosets = infoset_distance <= tolerance
  # No valid infoset was found
  if np.sum(valid_infosets) == 0:
    return -1
  # else return the best fitting candidate
  return np.argmin(infoset_distance)

def create_infoset_map(curr_infoset, amount_actions, curr_legal):
    """Creates an map where at index i there is an infoset corresponding to the index.
    Also returns per infoset legal actions like this, per history player infoset indices and per
    history player action indices (actions are differentiated by which infoset they are taken)"""
    infosets = [[], []]
    infoset_map = [[], []]
    infoset_legal = [[], []]
    for pl in range(curr_infoset.shape[0]):
      first_infoset_id = len(infoset_map[pl])
      for i in range(curr_infoset.shape[1]): 
        curr_index = -1
        for j in range(first_infoset_id, len(infoset_map[pl])):
          if infosets_close(infoset_map[pl][j], curr_infoset[pl, i]):
            curr_index = j
            break
        if curr_index < 0:
          curr_index = len(infoset_map[pl])
          infoset_map[pl].append(curr_infoset[pl, i])
          infoset_legal[pl].append(curr_legal[pl, i])
        infosets[pl].append(curr_index)
        
    infosets = np.array(infosets)
    actions = infosets[..., None] * amount_actions + np.arange(amount_actions)[None, None, ...] 
    infoset_map = [np.array(i) for i in infoset_map]
    infoset_legal = [np.array(i) for i in infoset_legal]
    return infoset_map, infoset_legal, infosets, actions

