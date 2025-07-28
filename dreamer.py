import chex
import os
import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
from functools import partial


from train_utils import DreamerConfig, get_reference_policy, TimeStep
from networks import create_networks_and_optimizers
from games.jax_game import JaxGame, GameState

  
@chex.dataclass(frozen=True)
class SampleTrajectoryCarry:
  game_state: GameState
  terminal: bool
  legal_actions: chex.Array



class Dreamer():
  """The actual model that handles the dreamer algorithm training.
  For now only the world model networks are used and actor/critic networks are not trained."""
  def __init__(self, config: DreamerConfig, game: JaxGame,  save_each=-1, print_each=-1, model_save_dir=""):
    self.config = config
    self.game = game
    self.init(save_each, print_each, model_save_dir)
    
    
  def init(self, save_each, print_each, model_save_dir):
    self.jax_rngs = jax.random.key(self.config.rng_seed)
    self.nnx_rngs = nnx.Rngs(self.generate_key())
    self.sample_rngs = nnx.Rngs(sampling =self.generate_key())
    
    
    self.networks_and_optimizers = create_networks_and_optimizers(self.config, self.game, self.nnx_rngs)
    self.learner_steps = 0

    self.trajectory_max = self.game.max_trajectory_length()
    self.action_dimension = self.game.num_distinct_actions()
    self.is_multi_agent = self.game.num_players() > 1
    self.sample_trajectory_func = self.sample_trajectory if self.is_multi_agent else self.sample_trajectory_single_agent

    self._get_example_timestep()
    self.prepare_checkpointer(save_each, print_each, model_save_dir)
    
  
  def _get_example_timestep(self):
        dummy_key = jax.random.key(0)
        example_state, example_legals = self.game.initialize_structures(dummy_key)
        if self.is_multi_agent:
          ex_state_tensor, ex_p1_iset, ex_p2_iset, ex_public_state = self.game.get_info(example_state)
          ex_obs = jnp.stack([ex_p1_iset, ex_p2_iset], axis=0)
        else:
          ex_state_tensor, ex_obs = self.game.get_info(example_state)
        valid = jnp.zeros(1, dtype=bool)
        legal = jnp.ones_like(example_legals)
        #the squeeze on axis 0 is for correct shape in a single agent environment
        action = jnp.squeeze(jnp.tile(nnx.one_hot(0, legal.shape[-1]), (self.game.num_players(), 1)), axis=0)
        policy = legal.astype(float) / jnp.sum(legal, axis=-1, keepdims=True)
        reward = jnp.zeros(1, dtype=float)
        terminal = jnp.zeros(1, dtype=bool)
        self.example_timestep = TimeStep(valid = valid,
                                        obs= ex_obs,
                                        action=action,
                                        legal=legal,
                                        policy = policy,
                                        reward = reward,
                                        terminal = terminal)
  
  def prepare_checkpointer(self, save_each, print_each, model_save_dir):
    """Initialize the checkpointer with given subdirectory and save_each configuration"""
    empty = ""
    game_params = self.game.params_dict()
    game_name = self.game.game_name()
    params_str = f'{empty.join(f"_{value}" for key, value in game_params.items())}'
    if not model_save_dir:
      model_save_dir = f"/trained_networks/{game_name}{params_str}/seed{self.config.seed}/network_seed{self.config.rng_seed}"
      model_save_dir = os.getcwd() + model_save_dir
    print(F"Saving at {model_save_dir}")
    options = ocp.CheckpointManagerOptions(
          save_interval_steps=save_each if save_each > 0 else None,
          create=True
      )
    self.print_each = print_each
    self.save_each = save_each
    self.checkpoint_manager = ocp.CheckpointManager(model_save_dir,
                                                    item_names = ("network_state", "config", "gameplay_key"),
                                                    options=options)  
  
  def generate_key(self):
    self.jax_rngs, key = jax.random.split(self.jax_rngs)
    return key

  def generate_keys(self, num_keys):
    split_key = self.generate_key()
    keys = jax.random.split(split_key, num_keys)
    return keys
    
  

  
  def training_step(self):
    gameplay_batch_key = self.generate_keys(self.config.batch_size)
    vectorized_sample_trajectories = jax.vmap(self.sample_trajectory_func, in_axes=(0), out_axes=(1))
    #[Trajectory, Batch, ...]
    batch_timestep = vectorized_sample_trajectories(gameplay_batch_key)

    loss = self.networks_and_optimizers.world_model_step(batch_timestep, self.sample_rngs)
    return loss



  def train_model(self, num_steps):
    for s in range(num_steps):
      step_loss = self.training_step()
      self.checkpoint_manager.save(self.learner_steps, args = ocp.args.Composite(
                                                        network_state = ocp.args.StandardSave(nnx.split(self.networks_and_optimizers)[1].to_pure_dict()),
                                                        config = ocp.args.StandardSave(self.config),
                                                        gameplay_key = ocp.args.ArraySave(self.gameplay_key)))
      if self.print_each > 0 and self.learner_steps % self.print_each == 0:
        print(f"Step {self.learner_steps}, loss: {step_loss}")
      self.learner_steps += 1
      self.checkpoint_manager.wait_until_finished()
    #Also update after training 
    
  def restore_latest_checkpoint(self, step: int=-1):
    """Restore the model from a given saved checkpoint.
    If step = -1 is provided, the last saved checkpoint is restored.
    Ensure this class has been initialized with the same game 
    and same parameters as the model checkpoint that you are attempting 
    to restore!!!""" 
    latest_step = self.checkpoint_manager.latest_step()
    all_steps = self.checkpoint_manager.all_steps()
    restore_step = step if step > -1 else latest_step
    assert latest_step is not None, "No network checkpoint was found!"
    assert restore_step in all_steps, f"Network checkpoint from step {restore_step} does not exist!"
    #We restore the saved config 
    # and network state. Then use the config to initialize self.networks_and_optimizers anew,
    # because the provided dummy config could cause shape mismatch. Then, finally
    restore_args = ocp.args.Composite(
                        #Restores as a general PyTree
                        network_state=ocp.args.StandardRestore(None),
                        config = ocp.args.StandardRestore(self.config),
                        gameplay_key = ocp.args.ArrayRestore(self.gameplay_key))
    print(f"Restoring model step {restore_step}")
    restored_items = self.checkpoint_manager.restore(restore_step, args=restore_args)
    self.config = restored_items["config"]
    #Also need to restore the last PRNG key that was used, 
    # to ensure that the trajectory sampling will not produce
    # identical data again
    self.gameplay_key = restored_items["gameplay_key"]
    #This will not work if a different game,
    # or the same game with different parameters is passed!!!
    new_networks_and_optimizers = create_networks_and_optimizers(self.config, self.game, self.nnx_rngs)
    graphdef, current_state = nnx.split(new_networks_and_optimizers)
    saved_state = restored_items["network_state"]
    current_state.replace_by_pure_dict(saved_state)
    self.networks_and_optimizers = nnx.merge(graphdef, current_state)
    #Remember where the training ended
    self.learner_steps = restore_step




  @partial(jax.jit, static_argnums=0)
  def sample_trajectory_single_agent(self, key) ->TimeStep:
    """Samples trajectories from the game for 
    game.max_trajectory_lenght turns (using valid to
    mask out invalid states that appear when playing post terminal).
    This is the single agent version."""
    init_key, trajectory_key, = jax.random.split(key)
    trajectory_key = jax.random.split(trajectory_key, self.trajectory_max)
    

    actions = self.action_dimension
          
    game_state, legal_actions = self.game.initialize_structures(init_key)
    init_carry = SampleTrajectoryCarry(
      game_state = game_state,
      terminal = False,
      legal_actions = legal_actions
    )
    
    @jax.jit
    def choice_wrapper(key, p):
      action = jax.random.choice(key, actions, p=p)
      action_oh = jax.nn.one_hot(action, actions)
      return action, action_oh
    
    
    def _sample_trajectory(carry: SampleTrajectoryCarry, xs) -> tuple[SampleTrajectoryCarry, chex.Array]:
      (key, turn) = xs
      
      state, obs = self.game.get_info(game_state)
      
      sample_key, action_key = jax.random.split(key)

      #For now we just use some very simple sampling policy
      # TODO: Change this to some better policy
      pi = get_reference_policy(carry.game_state, carry.legal_actions)
      # For each player samples a single action
      action, action_oh = choice_wrapper(key, pi)
      next_game_state, terminal, next_rewards, next_legal = self.game.apply_action(carry.game_state, action_key, turn, action)
      #Action in terminal state is not valid
      valid = (jnp.ones_like(next_rewards, dtype=int) - carry.terminal).astype(bool)
      terminal = jnp.where(valid, terminal, True)
      #We need state tensor to be defined in terminal state as well
      next_rewards = jnp.where(valid, next_rewards, jnp.zeros_like(next_rewards))
      obs = jnp.where(valid, obs, self.example_timestep.obs)   
      new_carry = SampleTrajectoryCarry(
        game_state = next_game_state,
        terminal = terminal,
        legal_actions=jnp.where(terminal, self.example_timestep.legal, next_legal)
      )
      timestep = TimeStep(
        valid = valid,
        obs = obs,
        legal = carry.legal_actions,
        action = action_oh,
        policy = pi,
        terminal = terminal[..., None],
        reward = next_rewards
      )
      return new_carry, timestep
    _, timestep = jax.lax.scan(_sample_trajectory,
             init=init_carry,
             xs=(trajectory_key, jnp.arange(self.trajectory_max)))
    #[Trajectory, ...]
    return timestep

  @partial(jax.jit, static_argnums=0)
  def sample_trajectory(self, key) ->TimeStep:
    """Samples trajectories from the game for 
    game.max_trajectory_lenght turns (using valid to
    mask out invalid states that appear when playing post terminal).
    This is the multi agent version."""
    init_key, trajectory_key, = jax.random.split(key)
    trajectory_key = jax.random.split(trajectory_key, self.trajectory_max)
    

    actions = self.action_dimension
          
    game_state, legal_actions = self.game.initialize_structures(init_key)
    init_carry = SampleTrajectoryCarry(
      game_state = game_state,
      terminal = False,
      legal_actions = legal_actions
    )
    
    @jax.jit
    def choice_wrapper(key, p):
      action = jax.random.choice(key, actions, p=p)
      action_oh = jax.nn.one_hot(action, actions)
      return action, action_oh

    
    vectorized_sample_action = jax.vmap(choice_wrapper, in_axes=(0, 0), out_axes=0)

    
    def _sample_trajectory(carry: SampleTrajectoryCarry, xs) -> tuple[SampleTrajectoryCarry, chex.Array]:
      (key, turn) = xs
      state, p1_iset, p2_iset, public_state = self.game.get_info(game_state)
      obs = jnp.stack((p1_iset, p2_iset), axis=0)
      
      sample_key, action_key = jax.random.split(key)

      #For now we just use some very simple sampling policy
      # TODO: Change this to some better policy
      pi = get_reference_policy(carry.game_state, carry.legal_actions)
      # For each player samples a single action
      sample_key = jax.random.split(sample_key, 2)
      action, action_oh = vectorized_sample_action(sample_key, pi)
      next_game_state, terminal, next_rewards, next_legal = self.game.apply_action(carry.game_state, action_key, turn, action)
      #Action in terminal state is not valid
      valid = (jnp.ones_like(next_rewards, dtype=int) - carry.terminal).astype(bool)
      terminal = jnp.where(valid, terminal, True)
      #We need state tensor to be defined in terminal state as well
      next_rewards = jnp.where(valid, next_rewards, jnp.zeros_like(next_rewards))
      obs = jnp.where(valid, obs, self.example_timestep.obs)   
      new_carry = SampleTrajectoryCarry(
        game_state = next_game_state,
        terminal = terminal,
        legal_actions=jnp.where(terminal, self.example_timestep.legal, next_legal)
      )
      timestep = TimeStep(
        valid = valid,
        obs = obs,
        legal = carry.legal_actions,
        action = action_oh,
        policy = pi,
        terminal = terminal[..., None],
        reward = next_rewards
      )
      return new_carry, timestep
    _, timestep = jax.lax.scan(_sample_trajectory,
             init=init_carry,
             xs=(trajectory_key, jnp.arange(self.trajectory_max)))
    #[Trajectory, ...]
    return timestep

  
  
  
  
  
  
  