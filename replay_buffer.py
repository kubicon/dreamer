import jax
import jax.numpy as jnp
import chex
import numpy as np

import flax.nnx as nnx
from dataclasses import dataclass
from functools import partial

from distributions import sample_categorical
from networks import *
from games.jax_game import JaxGame, GameState, InformationType
from train_utils import TimeStep, BufferConfig, get_reference_policy, tree_where

@dataclass
class BufferTimeStep():
  """Same structure as TimeStep, only
  is intended to be used on CPU, hence the arrays are numpy"""
  obs: np.ndarray
  action: np.ndarray
  legal: np.ndarray
  policy: np.ndarray
  reward: np.ndarray
  terminal: np.ndarray
  valid: np.ndarray

  def __getitem__(self, indices):
    return BufferTimeStep(
      obs = self.obs[indices],
      action = self.action[indices],
      legal = self.legal[indices],
      policy = self.policy[indices],
      reward = self.reward[indices],
      terminal = self.terminal[indices],
      valid = self.valid[indices]
    )

  def __setitem__(self, indices, value):
    self.obs[indices] = value.obs
    self.action[indices] = value.action
    self.legal[indices] = value.legal
    self.policy[indices] = value.policy
    self.reward[indices] = value.reward
    self.terminal[indices] = value.terminal
    self.valid[indices] = value.valid

class ReplayBuffer():

  def __init__(self, game: JaxGame, config: BufferConfig, world_model_config: DreamerMAConfig):
    """Circular replay buffer for collecting trajectories from the game for dreamer.
    Individual elements are trajectories."""
    self.game = game
    self.key = jax.random.key(config.trajectory_seed)
    self.np_rng = np.random.default_rng(seed=config.buffer_sample_seed)
    self.config = config
    self.wm_config = world_model_config
    self.init_constants()
    self.init_buffer()

  def init_constants(self):
    
    self.action_dimension = self.game.num_distinct_actions()
    self.num_players = self.game.num_players()
    self.trajectory_max = self.game.max_trajectory_length()
    hidden_state_size = self.wm_config.sequential_network_details[0]

    if hidden_state_size < 1:
      hidden_state_size = self.num_players * self.game.information_state_tensor_shape()
    self.hidden_state_size = hidden_state_size
    #Chance nodes are not explicitly stored in the buffer, instead
    # they are skipped and only the next outcome sampled from it 
    # is stored.
    self.non_chance_trajectory_max = self.game.max_trajectory_lenght_no_chance()

    self.use_iset = self.game.information_type() == InformationType.IIG
    self._get_example_timestep()

    #If we supply replay ratio < 1, it is assumed
    # that we want all steps online
    if self.config.replay_ratio < 1:
      self.online_batches = self.wm_config.batch_size
      self.replayed_batches = 0
    else:
      total_minibatch_size = self.wm_config.batch_size * self.non_chance_trajectory_max

      assert total_minibatch_size % self.config.replay_ratio == 0, f"Total size of minibatch {self.non_chance_trajectory_max}x{self.config.batch_size} is not divisible by replay ratio {self.config.replay_ratio}."

      online_steps_per_batch = int(total_minibatch_size / self.config.replay_ratio)

      assert online_steps_per_batch % self.non_chance_trajectory_max == 0, f"The amount of online steps per batch {online_steps_per_batch} needs to be divisible into trajectories of lenght {self.non_chance_trajectory_max}."
      self.online_batches = int(online_steps_per_batch / self.non_chance_trajectory_max)
      self.replayed_batches = self.wm_config.batch_size - self.online_batches

    self.cached_sample = None
  
  
  def init_buffer(self):
    #Use np for the buffer itself, since it is supposed to run
    # on cpu. The actual sampling can be done on gpu
    obs = np.zeros((self.config.buffer_size, self.non_chance_trajectory_max, *self.example_timestep.obs.shape))
    action = np.zeros((self.config.buffer_size, self.non_chance_trajectory_max, *self.example_timestep.action.shape), dtype=np.int8)
    legal = np.zeros((self.config.buffer_size, self.non_chance_trajectory_max, *self.example_timestep.legal.shape), dtype=np.int8)
    policy = np.zeros((self.config.buffer_size, self.non_chance_trajectory_max,  *self.example_timestep.policy.shape))
    reward = np.zeros((self.config.buffer_size, self.non_chance_trajectory_max))
    terminal = np.zeros((self.config.buffer_size, self.non_chance_trajectory_max), dtype=np.bool)
    valid = np.zeros((self.config.buffer_size, self.non_chance_trajectory_max), dtype=np.bool)
    
    self.buffer = BufferTimeStep(
      obs = obs,
      action = action,
      legal = legal,
      policy = policy,
      reward = reward,
      terminal = terminal,
      valid = valid
    )
    self.buffer_index = 0
    self.full = False

  def _get_example_timestep(self):
    #This can produce a chance node, but that 
    # one by default produces invalid isets
    # and legals so it is not a problem 
    example_state, example_legals = self.game.initialize_structures()
    if self.use_iset:
      _, ex_p1_iset, ex_p2_iset, _ = self.game.get_info(example_state)
      ex_obs = jnp.stack([ex_p1_iset, ex_p2_iset], axis=0)
    else:
      deter_size = self.wm_config.encoded_classes * self.wm_config.encoded_categories
      ex_model_state = jnp.zeros(deter_size + self.hidden_state_size + self.num_players)
      players_oh = jnp.eye(self.num_players)
      ex_obs = jnp.concatenate([jnp.stack([ex_model_state, ex_model_state], axis=-2), players_oh], axis=-1)
    legal = jnp.ones(example_legals.shape, dtype=jnp.int8)
    action = jax.nn.one_hot(jnp.argmax(legal, -1), legal.shape[-1]).astype(jnp.int8)
    policy = legal.astype(float) / jnp.sum(legal, axis=-1, keepdims=True)
    self.example_timestep = TimeStep(
                                    obs= ex_obs,
                                    action=action,
                                    legal=legal,
                                    policy = policy,
                                    reward = 0.0,
                                    terminal = False,
                                    valid = False)
    

  def cache_sampling(self, recurrent_network: SequenceModel, encoder_network: JointIsetEncoder, actor_network: ActorNetwork | RNaDNetwork | None = None):
    if actor_network is None:
      #Just put some dummy array there. This is
      # so that I can train Dreamer only without the actor.
      # CRUCIAL: Do NOT try to call the trajectory
      # campling with both the dummy input and with actual
      # input. It will not work (but I cannot think of 
      # any reason why I would want to do that.)
      actor_network = jnp.array(0)
    self.cached_sample = nnx.cached_partial(self.sample_batch_trajectories, recurrent_network,
                                            encoder_network, actor_network)
    
  
  def mixed_sample(self):
    """Handles the case where we both want to sample trajectories online
    and from the buffer. Samples online_batch_size online batches,
    buffer_batch_size batches from replay buffer and concatenates them
    together. The size of these batches is computed based on replay
    ratio, such that there are (batch size * trajectory len) / replay ratio
    online steps per minibatch"""
    if self.online_batches == 0:
      return self.sample_batch(self.replayed_batches)
    elif self.replayed_batches == 0:
      online_batch = self.add_batch(self.online_batches)
      return online_batch
    #Sample first, before putting the new online trajectories there
    buffer_batch = self.sample_batch(self.replayed_batches)
    online_batch = self.add_batch(self.online_batches)
    #The second axis is batch size
    compound_batch = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=1), online_batch, buffer_batch)
    return compound_batch
  

  def env_to_buffer_timestep(self, env_timestep: TimeStep) ->BufferTimeStep:
    """Convert a timestep sampled from the environment 
    to the buffer timestep. Also, transposes from [Time, Batch, ...]
    format in which the environment steps are collected
    into [Batch, Time, ...] format in which they are stored in the buffer"""
    env_timestep = jax.tree.map(lambda x: x.transpose(1, 0, *(range(2, x.ndim))), env_timestep)
    buffer_timestep = BufferTimeStep(
                                    obs= np.asarray(env_timestep.obs),
                                    action= np.asarray(env_timestep.action),
                                    legal= np.asarray(env_timestep.legal),
                                    policy= np.asarray(env_timestep.policy),
                                    reward= np.asarray(env_timestep.reward),
                                    terminal= np.asarray(env_timestep.terminal),
                                    valid= np.asarray(env_timestep.valid))
    return buffer_timestep

  def buffer_to_env_timestep(self, buffer_timestep: BufferTimeStep) ->TimeStep:
    """Convert a timestep sampled from buffer into environment timestep.
    Also, transposes from [Batch, Time, ...] to [Time, Batch, ...], since 
    the training expects time major format."""
    env_timestep = TimeStep(
                                    obs= jnp.asarray(buffer_timestep.obs),
                                    action= jnp.asarray(buffer_timestep.action),
                                    legal= jnp.asarray(buffer_timestep.legal),
                                    policy= jnp.asarray(buffer_timestep.policy),
                                    reward= jnp.asarray(buffer_timestep.reward),
                                    terminal= jnp.asarray(buffer_timestep.terminal),
                                    valid= jnp.asarray(buffer_timestep.valid))
    env_timestep = jax.tree.map(lambda x: x.transpose((1, 0, *range(2, x.ndim))).astype(x.dtype), env_timestep)
    return env_timestep

  def add_single(self, buffer_timestep: BufferTimeStep):
    """Add a single trajectory into the buffer"""
    self.buffer[self.buffer_index] = buffer_timestep

    self.buffer_index = self.buffer_index + 1
    if self.buffer_index == self.config.buffer_size:
      self.buffer_index = 0
      self.full = True

  def add_batch(self, batch_size: int):
    """Sample a batch of trajectories from the environment and add them to the buffer.
    Also returns the trajectories if you want to perform online training on them."""
    assert self.cached_sample is not None, "Calling add batch without the network arguments, but the step was not cached yet. Call cache_sampling first with the network arguments." 
    self.key, subkey = jax.random.split(self.key)
    batch_trajectories = self.cached_sample(batch_size, subkey)
    buffer_timestep = self.env_to_buffer_timestep(batch_trajectories)
    for i in range(batch_size):
      self.add_single(buffer_timestep[i])
    return batch_trajectories

  def add_batch(self, batch_size: int, recurrent_network: SequenceModel| None = None, 
                encoder_network: JointIsetEncoder | None = None, actor_network: ActorNetwork | RNaDNetwork | None =None):
    """Sample a batch of trajectories from the environment and add them to the buffer.
    Also returns the trajectories if you want to perform online training on them."""
    self.key, subkey = jax.random.split(self.key)
    if all([net is not None for net in (recurrent_network, encoder_network, actor_network)]):
      batch_trajectories = self.sample_batch_trajectories(recurrent_network, encoder_network, actor_network, batch_size, subkey)
    else:
      assert self.cached_sample is not None, "The variant of add_batch where one or more of the networks are unset was called, but cached_sample is not set. Please call cache_sampling first."
      batch_trajectories = self.cached_sample(batch_size, subkey)
    buffer_timestep = self.env_to_buffer_timestep(batch_trajectories)
    for i in range(batch_size):
      self.add_single(buffer_timestep[i])
    return batch_trajectories


  def sample_batch(self, batch_size: int) ->TimeStep:
    #empty buffer
    high = self.config.buffer_size if self.full else self.buffer_index
    sampled_indices = self.np_rng.integers(0, high, size=batch_size)
    sampled_timesteps = self.buffer[sampled_indices]
    timesteps_for_train = self.buffer_to_env_timestep(sampled_timesteps)
    return timesteps_for_train
  

  def __getstate__(self):
    return {
      "game": self.game,
      "config": self.config,
      "wm_config": self.wm_config,
      "jax_rngs": self.key,
      "numpy_rng_state": self.np_rng.bit_generator.state,
      "full" : self.full,
      "buffer_index": self.buffer_index,
      "buffer": self.buffer,
    }
    
  
  def __setstate__(self, state):
    self.game = state["game"]
    self.key = state["jax_rngs"]

    self.config = state["config"]
    self.wm_config = state["wm_config"]

    self.np_rng = np.random.default_rng(self.config.buffer_sample_seed)
    self.np_rng.bit_generator.state = state["numpy_rng_state"]
    self.full = state["full"]
    self.buffer_index = state["buffer_index"]
    self.buffer = state["buffer"]

    self.init_constants()
    
    


  @partial(nnx.jit, static_argnums=(0, 4))
  def sample_batch_trajectories(self, recurrent_network: SequenceModel, encoder_network: JointIsetEncoder, actor_network: ActorNetwork | RNaDNetwork, batch_size:int, key):
    batch_keys = jax.random.split(key, batch_size)
    batch_sample_trajectories = nnx.vmap(self.sample_trajectory, in_axes=(None, None, None, 0), out_axes=(1))
    batch_trajectories = batch_sample_trajectories(recurrent_network, encoder_network, actor_network, batch_keys)
    return batch_trajectories

  @partial(nnx.jit, static_argnums=0)
  def sample_trajectory(self, recurrent_network: SequenceModel, encoder_network:JointIsetEncoder, actor_network:ActorNetwork | RNaDNetwork, key) ->TimeStep:
    trajectory_key = jax.random.split(key, self.trajectory_max)
    actions = self.action_dimension
    
    #TODO: Will later have to properly split the segments 
    # and also store the starting model state
    game_state, legal_actions = self.game.initialize_structures()
    init_hidden = jnp.zeros(self.hidden_state_size)
    
    @chex.dataclass(frozen=True)
    class SampleTrajectoryCarry:
      game_state: GameState
      legal_actions: chex.Array
      reward: chex.Array
      terminal: bool
      valid: bool
      hidden_state: chex.Array
      
    init_carry = SampleTrajectoryCarry(
      game_state = game_state,
      legal_actions = legal_actions,
      reward = jnp.array(0),
      terminal = jnp.array(False),
      valid = jnp.array(True),
      hidden_state = init_hidden
    )
    
    
    @nnx.jit
    def choice_wrapper(key, p):
      action = jax.random.choice(key, actions, p=p)
      action_oh = jax.nn.one_hot(action, actions)
      return action, action_oh

    
    vectorized_sample_action = nnx.vmap(choice_wrapper, in_axes=(0, 0), out_axes=0)

    @nnx.scan(in_axes=(nnx.Carry, None, None, None, 0), out_axes=(nnx.Carry, 0))
    def _sample_trajectory(carry: SampleTrajectoryCarry, recurrent_network: SequenceModel, encoder_network:JointIsetEncoder, actor_network:ActorNetwork | RNaDNetwork, key) -> tuple[SampleTrajectoryCarry, chex.Array]:
      
      
      state, p1_iset, p2_iset, public_state = self.game.get_info(carry.game_state)
      action_key, chance_key, deter_sample_key = jax.random.split(key, 3)
      
      obs = jnp.stack((p1_iset, p2_iset), axis=0)
      encoded_stoch = encoder_network(carry.hidden_state, obs)
      encoded_deter = sample_categorical(encoded_stoch, deter_sample_key)
      #per player vmap
      vectorized_get_actor = nnx.vmap(actor_network, in_axes=(0, 0), out_axes=0)
      if not self.use_iset:
        flat_deter = jnp.reshape(encoded_deter, (*encoded_deter.shape[:-2], -1))
        model_state = jnp.concatenate([carry.hidden_state, flat_deter], axis=-1)
        players_oh = jnp.eye(self.num_players)
        obs = jnp.concatenate([jnp.stack([model_state, model_state], axis=-2), players_oh], axis=-1)
      if self.config.on_policy:
        pi = vectorized_get_actor(obs, carry.legal_actions)[0]
      else:
        pi = get_reference_policy(obs, carry.legal_actions)
      is_chance = self.game.is_chance(carry.game_state)
      action_key = jax.random.split(action_key, self.game.num_players())
      action, action_oh = vectorized_sample_action(action_key, pi)
      timestep = TimeStep(
        obs = obs,
        legal = carry.legal_actions.astype(jnp.int8),
        action = action_oh.astype(jnp.int8),
        policy = pi,
        reward = carry.reward,
        valid = carry.valid,
        terminal = carry.terminal
      )
      

      def apply_action():
        return self.game.apply_action(carry.game_state, action)
      def sample_chance():
        outcomes, probs = self.game.get_outcomes_and_probs(carry.game_state)
        # Do not forget for deterministic games to put nonzero probs
        # to sample something for shape consistency
        probs = jnp.where(is_chance, probs, jnp.ones_like(probs)/ probs.shape[0])
        chosen_outcome = jax.random.choice(chance_key, outcomes, p=probs)
        outcome, terminal, reward, chosen_legals = self.game.apply_action(carry.game_state, chosen_outcome)
        return outcome, terminal, reward, chosen_legals
      next_game_state, next_terminal, next_rewards, next_legal = jax.lax.cond(is_chance, sample_chance, apply_action)
      #Action in terminal state is not valid
      next_terminal = jnp.logical_or(carry.terminal, next_terminal)
      next_valid = jnp.logical_not(carry.terminal)
      next_hidden = recurrent_network(carry.hidden_state, encoded_deter, action_oh)
      new_carry = SampleTrajectoryCarry(
        game_state = next_game_state,
        legal_actions=jnp.where(next_terminal, self.example_timestep.legal, next_legal),
        reward = next_rewards,
        terminal = next_terminal,
        valid = next_valid,
        hidden_state = next_hidden
      )
        
      
      timestep = tree_where(carry.valid, timestep, self.example_timestep)
      
      return new_carry, (timestep, is_chance)
    _, ys = _sample_trajectory(init_carry, recurrent_network, encoder_network, actor_network, trajectory_key)
    timestep, is_chance = ys
    #This is used to remove the chance nodes from the trajectory
    non_chance = jnp.nonzero(~is_chance, size=self.non_chance_trajectory_max)[0]
    filtered_timestep = jax.tree.map(lambda x: jnp.take_along_axis(x, jnp.expand_dims(non_chance, axis=range(1, x.ndim)), axis=0).astype(x.dtype), timestep)
    #[Trajectory, ...]
    return filtered_timestep
