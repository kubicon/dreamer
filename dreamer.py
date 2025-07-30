import chex
import os
import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
from functools import partial


from train_utils import DreamerConfig, get_reference_policy, TimeStep, get_loss_mean_with_mask, PredictionStep
from distributions import get_normal_log_prob, get_bin_log_prob, kl_divergence, sample_categorical
from networks import initialize_dreamer_optimizers, DreamerOptimizers, SequenceModel, Encoder, Decoder, DynamicsPredictor, Predictor
from games.jax_game import JaxGame, GameState


@chex.dataclass(frozen=True)
class SampleTrajectoryCarry:
  game_state: GameState
  terminal: bool
  legal_actions: chex.Array
  valid: bool


class Dreamer():
  """The actual model that handles the dreamer algorithm training.
  For now only the world model networks are used and actor/critic networks are not trained."""
  def __init__(self, config: DreamerConfig, game: JaxGame):
    self.config = config
    self.game = game
    self.init()
    
    
  def init(self):
    self.jax_rngs = jax.random.key(self.config.rng_seed)
    self.nnx_rngs = nnx.Rngs(self.generate_key())
    
    
    self.optimizers = initialize_dreamer_optimizers(self.config, self.game, self.nnx_rngs)
    self.learner_steps = 0

    #We want + 1. since we want the terminal state as well
    # as representation should be also trained on those
    self.trajectory_max = self.game.max_trajectory_length() + 1

    self.action_dimension = self.game.num_distinct_actions()
    self.is_multi_agent = self.game.num_players() > 1
    self.sample_trajectory_func = self.sample_trajectory if self.is_multi_agent else self.sample_trajectory_single_agent

    self._get_example_timestep()
    self.cached_train = nnx.cached_partial(self.world_model_train, self.optimizers)
    
  
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
    self.example_timestep = TimeStep(action_valid = valid,
                                    state_valid = valid,
                                    obs= ex_obs,
                                    action=action,
                                    legal=legal,
                                    policy = policy,
                                    reward = reward,
                                    terminal = terminal)
  
  def generate_key(self):
    self.jax_rngs, key = jax.random.split(self.jax_rngs)
    return key

  def generate_keys(self, num_keys):
    split_key = self.generate_key()
    keys = jax.random.split(split_key, num_keys)
    return keys
    

  @partial(nnx.jit, static_argnums=(0, 1))
  def sample_trajectories(self,  batch_size, key):
    keys = jax.random.split(key, batch_size)
    return nnx.vmap(self.sample_trajectory_func, in_axes=0, out_axes=1)(keys)
  

  @partial(jax.jit, static_argnums=0)
  def sample_trajectory_single_agent(self, key) ->TimeStep:
    """Samples trajectories from the game for 
    game.max_trajectory_lenght turns (using valid to
    mask out invalid states that appear when playing post terminal).
    This is the single agent version."""
    # TODO: Can we find some common interface so we do not repeat most of the stuff twice.
    init_key, trajectory_key, = jax.random.split(key)
    trajectory_key = jax.random.split(trajectory_key, self.trajectory_max)
    

    actions = self.action_dimension
          
    game_state, legal_actions = self.game.initialize_structures(init_key)
    init_carry = SampleTrajectoryCarry(
      game_state = game_state,
      terminal = False,
      valid = True,
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
      action_valid = (jnp.ones_like(next_rewards, dtype=int) - carry.terminal).astype(bool)
      state_valid = (jnp.ones_like(next_rewards, dtype=int) - carry.valid).astype(bool)
      terminal = jnp.where(action_valid, terminal, True)
      #We need state tensor to be defined in terminal state as well
      next_rewards = jnp.where(action_valid, next_rewards, jnp.zeros_like(next_rewards))
      obs = jnp.where(state_valid, obs, self.example_timestep.obs)   
      new_carry = SampleTrajectoryCarry(
        game_state = next_game_state,
        terminal = terminal,
        valid = action_valid,
        legal_actions=jnp.where(terminal, self.example_timestep.legal, next_legal)
      )
      timestep = TimeStep(
        action_valid = action_valid,
        state_valid = state_valid,
        obs = obs,
        legal = carry.legal_actions,
        action = action_oh,
        policy = pi,
        terminal = terminal[..., None],
        reward = next_rewards[..., None]
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
      valid = True,
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
      action_valid = (jnp.ones_like(next_rewards, dtype=int) - carry.terminal).astype(bool)
      state_valid = (jnp.ones_like(next_rewards, dtype=int) - carry.valid).astype(bool)
      terminal = jnp.where(action_valid, terminal, True)
      #We need state tensor to be defined in terminal state as well
      next_rewards = jnp.where(action_valid, next_rewards, jnp.zeros_like(next_rewards))
      obs = jnp.where(state_valid, obs, self.example_timestep.obs)   
      new_carry = SampleTrajectoryCarry(
        game_state = next_game_state,
        terminal = terminal,
        valid = action_valid,
        legal_actions=jnp.where(terminal, self.example_timestep.legal, next_legal)
      )
      timestep = TimeStep(
        action_valid = action_valid,
        state_valid = state_valid,
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
  
  def update_world_model(self, optimizers: DreamerOptimizers, timestep: TimeStep, rng_key):
    """Compound loss for the entire world model."""
    sample_keys = jax.random.split(rng_key, self.trajectory_max * self.config.batch_size)
    sample_keys = sample_keys.reshape((self.trajectory_max, self.config.batch_size))
    
    def world_model_loss(sequence_model: SequenceModel, encoder: Encoder, decoder: Decoder, dynamics_model: DynamicsPredictor, predictor: Predictor):
      l_pred, l_dyn, l_rep = 0, 0, 0
      #[Trajectory, Batch, ...]
      @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
      def _predict_over_timestep(carry, xs):
        sequence_model, encoder, decoder, dynamics_model, predictor, hidden_state = carry
        action, obs, cur_key = xs
        stochastic_state = encoder(hidden_state, obs)
        deterministic_state = sample_categorical(stochastic_state, cur_key)
        next_stochastic_state = dynamics_model(hidden_state)
        reward, done = predictor(hidden_state, deterministic_state)
        decoded_obs = decoder(hidden_state, deterministic_state)
        gru_input = jnp.concatenate([deterministic_state.reshape(*deterministic_state.shape[:-2], -1), action], axis=-1) 
        new_hidden = sequence_model(hidden_state, gru_input)
        preds = PredictionStep(repr_state = stochastic_state,
                                decoded_obs = decoded_obs,
                                reward_dist_logit = reward,
                                done_logit = done,
                                dynamics_state = next_stochastic_state)
        new_carry = (sequence_model, encoder, decoder, dynamics_model, predictor, new_hidden)
        return new_carry, preds
      
      xs = (timestep.action, timestep.obs, sample_keys)
      init_carry = jnp.zeros((self.config.batch_size, self.config.hidden_state_size)) 
      init_hidden = (sequence_model, encoder, decoder, dynamics_model, predictor, init_carry)
      vectorized_predict = nnx.vmap(_predict_over_timestep, in_axes=((None, None, None, None, None, 0), (1, 1, 1)), out_axes=((None, None, None, None, None, 0), 1))
      final_hidden, predictions = vectorized_predict(init_hidden, xs) 
       
      #[Trajectory, Batch, obs_size]
      reconstruction_loss = -get_normal_log_prob(predictions.decoded_obs, timestep.obs, use_symlog=True)
      l_pred += get_loss_mean_with_mask(reconstruction_loss, timestep.state_valid[..., None])
      #[Trajectory, Batch, 1]
      continuation_loss = -get_normal_log_prob(predictions.done_logit, timestep.terminal.astype(int))
      l_pred += get_loss_mean_with_mask(continuation_loss, timestep.action_valid[..., None])
      #[Trajectory, Batch, 2* bin_range + 1]
      bins = jnp.arange((2 * self.config.bin_range) + 1) - self.config.bin_range
      reward_loss = -get_bin_log_prob(predictions.reward_dist_logit, bins, timestep.reward, use_symlog=True)
      l_pred += get_loss_mean_with_mask(reward_loss, timestep.action_valid[..., None])

      #Using free bits to clip dynamics and representation losses
      # to 1, thus disabling their gradient when they are below 1
      #[Trajectory, Batch, encoded_categories, encoded_classes]

      posterior = nnx.softmax(predictions.repr_state, axis=-1)
      prior = nnx.softmax(predictions.dynamics_state, axis=-1)
      #TODO: Use state_valid or action_valid here?
      #[Trajectory, Batch]
      dynamics_loss = jnp.maximum(1, kl_divergence(jax.lax.stop_gradient(posterior), prior))
      l_dyn += get_loss_mean_with_mask(dynamics_loss, timestep.state_valid)
      #[Trajectory, Batch]
      repr_loss = jnp.maximum(1, kl_divergence(posterior, jax.lax.stop_gradient(prior)))
      l_rep += get_loss_mean_with_mask(repr_loss, timestep.state_valid)


      return self.config.beta_prediction * l_pred + self.config.beta_dynamics * l_dyn + self.config.beta_representation * l_rep
  
    loss, grad = nnx.value_and_grad(world_model_loss, argnums=(0, 1, 2, 3, 4))(optimizers.sequence_optimizer.model, optimizers.encoder_optimizer.model, optimizers.decoder_optimizer.model, optimizers.dynamics_optimizer.model, optimizers.predictor_optimizer.model)
    
    optimizers.sequence_optimizer.update(grad[0])
    optimizers.encoder_optimizer.update(grad[1])
    optimizers.decoder_optimizer.update(grad[2])
    optimizers.dynamics_optimizer.update(grad[3])
    optimizers.predictor_optimizer.update(grad[4])
    
    return loss
  
  # Unlike flax.linen, nnx.jit allows updating the model itself.
  @partial(nnx.jit, static_argnums=(0))
  def world_model_train(self, optimizers, rng_key):
    trajectory_key, train_key = jax.random.split(rng_key)
    timestep = self.sample_trajectories(self.config.batch_size, trajectory_key)
    loss = self.update_world_model(optimizers,timestep, train_key)
    return loss
    

  def world_model_train_step(self):
    rng_key = self.generate_key()
    return self.world_model_train(self.optimizers, rng_key)
    # return self.cached_train(rng_key)
   
  def __getstate__(self):
    return {
      "config": self.config,
      "game": self.game,
      "jax_rngs": self.jax_rngs,
      "nnx_rngs": nnx.state(self.nnx_rngs),
      "optimizers": nnx.state(self.optimizers)
    } 
    
    
  def __setstate__(self, state):
    self.config = state["config"]
    self.game = state["game"]
    
    self.init()
    
    def update_nnx(model_optimizer, load_optimizer):
      static_graph, _ = nnx.split(model_optimizer)
      model_optimizer = nnx.merge(static_graph, load_optimizer)
      return model_optimizer
    
    self.jax_rngs = state["jax_rngs"]
    self.nnx_rngs = update_nnx(self.nnx_rngs, state["nnx_rngs"])
    self.optimizers = update_nnx(self.optimizers, state["optimizers"])
    
    