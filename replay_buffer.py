import jax
import jax.numpy as jnp
import chex
import numpy as np

from functools import partial
from dataclasses import dataclass
from games.jax_game import JaxGame, GameState
from train_utils import TimeStep, get_reference_policy

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

  def __init__(self, game: JaxGame, trajectory_seed: int, buffer_sample_seed:int, buffer_size: int):
    """Circular replay buffer for collecting trajectories from the game for dreamer.
    Individual elements are trajectories."""
    self.game = game
    self.key = jax.random.key(trajectory_seed)
    self.np_rng = np.random.default_rng(seed=buffer_sample_seed)
    self.jax_seed = trajectory_seed
    self.np_seed = buffer_sample_seed
    self.init(buffer_size)

  def init(self, buffer_size):
    self._get_example_timestep()
    self.action_dimension = self.game.num_distinct_actions()
    self.trajectory_max = self.game.max_trajectory_length()
    #Chance nodes are not explicitly stored in the buffer, instead
    # they are skipped and only the next outcome sampled from it 
    # is stored.
    self.non_chance_trajectory_max = self.game.max_trajectory_lenght_no_chance()

    #Use np for the buffer itself, since it is supposed to run
    # on cpu. The actual sampling can be done on gpu
    obs = np.zeros((buffer_size, self.non_chance_trajectory_max, *self.example_timestep.obs.shape))
    action = np.zeros((buffer_size, self.non_chance_trajectory_max, *self.example_timestep.action.shape), dtype=np.int8)
    legal = np.zeros((buffer_size, self.non_chance_trajectory_max, *self.example_timestep.legal.shape), dtype=np.int8)
    policy = np.zeros((buffer_size, self.non_chance_trajectory_max,  *self.example_timestep.policy.shape))
    reward = np.zeros((buffer_size, self.non_chance_trajectory_max))
    terminal = np.zeros((buffer_size, self.non_chance_trajectory_max), dtype=np.bool)
    valid = np.zeros((buffer_size, self.non_chance_trajectory_max), dtype=np.bool)
    
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
    self.buffer_size = buffer_size
    self.full = False

  def _get_example_timestep(self):
    #This can produce a chance node, but that 
    # one by default produces invalid isets
    # and legals so it is not a problem 
    example_state, example_legals = self.game.initialize_structures()
    _, ex_p1_iset, ex_p2_iset, _ = self.game.get_info(example_state)
    ex_obs = jnp.stack([ex_p1_iset, ex_p2_iset], axis=0)
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
    
  

  def env_to_buffer_timestep(self, env_timestep: TimeStep) ->BufferTimeStep:
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
    if self.buffer_index == self.buffer_size:
      self.buffer_index = 0
      self.full = True

  def add_batch(self, batch_size: int):
    """Sample a batch of trajectories from the environment and add them to the buffer"""
    self.key, subkey = jax.random.split(self.key)
    #breakpoint()
    batch_trajectories = self.sample_batch_trajectories(batch_size, subkey)
    buffer_timestep = self.env_to_buffer_timestep(batch_trajectories)
    for i in range(batch_size):
      self.add_single(buffer_timestep[i])

  
  def sample_batch(self, batch_size: int) ->TimeStep:
    #empty buffer
    high = self.buffer_size if self.full else self.buffer_index
    sampled_indices = self.np_rng.integers(0, high, size=batch_size)
    sampled_timesteps = self.buffer[sampled_indices]
    timesteps_for_train = self.buffer_to_env_timestep(sampled_timesteps)
    return timesteps_for_train
  

  def __getstate__(self):
    return {
      "game": self.game,
      "jax_rngs": self.key,
      "numpy_rng_state": self.np_rng.bit_generator.state,
      "full" : self.full,
      "buffer_index": self.buffer_index,
      "buffer": self.buffer,
      "jax_seed": self.jax_seed,
      "numpy_seed": self.np_rng
    }
    
  
  def __setstate__(self, state):
    self.game = state["game"]
    self.key = state["jax_rngs"]
    self.np_seed = state["numpy_seed"]
    self.np_rng = np.random.default_rng(self.np_seed)
    self.np_rng.bit_generator.state = state["numpy_rng_state"]
    self.full = state["full"]
    self.buffer_index = state["buffer_index"]
    self.buffer = state["buffer"]

    self._get_example_timestep()
    self.action_dimension = self.game.num_distinct_actions()
    self.trajectory_max = self.game.max_trajectory_length()
    #Chance nodes are not explicitly stored in the buffer, instead
    # they are skipped and only the next outcome sampled from it 
    # is stored.
    self.non_chance_trajectory_max = self.game.max_trajectory_lenght_no_chance()
    
    self.jax_seed = state["jax_seed"]
    


  @partial(jax.jit, static_argnums=(0, 1))
  def sample_batch_trajectories(self, batch_size:int, key):
    batch_keys = jax.random.split(key, batch_size)
    batch_sample_trajectories = jax.vmap(self.sample_trajectory, in_axes=(0), out_axes=(0))
    batch_trajectories = batch_sample_trajectories(batch_keys)
    return batch_trajectories

  @partial(jax.jit, static_argnums=0)
  def sample_trajectory(self, key) ->TimeStep:
    trajectory_key = jax.random.split(key, self.trajectory_max)
    actions = self.action_dimension
    
    game_state, legal_actions = self.game.initialize_structures()
    
    @chex.dataclass(frozen=True)
    class SampleTrajectoryCarry:
      game_state: GameState
      legal_actions: chex.Array
      reward: chex.Array
      terminal: bool
      valid: bool
      
    init_carry = SampleTrajectoryCarry(
      game_state = game_state,
      legal_actions = legal_actions,
      reward = jnp.array(0),
      terminal = jnp.array(False),
      valid = jnp.array(True)
    )
    
    
    @jax.jit
    def choice_wrapper(key, p):
      action = jax.random.choice(key, actions, p=p)
      action_oh = jax.nn.one_hot(action, actions)
      return action, action_oh

    
    vectorized_sample_action = jax.vmap(choice_wrapper, in_axes=(0, 0), out_axes=0)

    def _sample_trajectory(carry: SampleTrajectoryCarry, key) -> tuple[SampleTrajectoryCarry, chex.Array]:
      
      state, p1_iset, p2_iset, public_state = self.game.get_info(carry.game_state)
      obs = jnp.stack((p1_iset, p2_iset), axis=0)
      action_key, chance_key = jax.random.split(key)

      #For now we just use some very simple sampling policy
      # TODO: Change this to some better policy
      pi = get_reference_policy(carry.game_state, carry.legal_actions)
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
      new_carry = SampleTrajectoryCarry(
        game_state = next_game_state,
        legal_actions=jnp.where(next_terminal, self.example_timestep.legal, next_legal),
        reward = next_rewards,
        terminal = next_terminal,
        valid = next_valid
      )
        
      
      timestep = jax.tree.map(lambda t, f: jnp.where(carry.valid, t, f).astype(t.dtype), timestep, self.example_timestep)
      
      return new_carry, (timestep, is_chance)
    _, ys = jax.lax.scan(_sample_trajectory, init_carry, trajectory_key)
    timestep, is_chance = ys
    #This is used to remove the chance nodes from the trajectory
    non_chance = jnp.nonzero(~is_chance, size=self.non_chance_trajectory_max)[0]
    filtered_timestep = jax.tree_util.tree_map(lambda x: jnp.take_along_axis(x, jnp.expand_dims(non_chance, axis=range(1, x.ndim)), axis=0).astype(x.dtype), timestep)
    #[Trajectory, ...]
    return filtered_timestep
