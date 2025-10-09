import chex
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from functools import partial


from train_utils import DreamerMAConfig, get_reference_policy, TimeStep, get_loss_mean_with_mask, PredictionStepWithLegal, symexp, save_model
from distributions import get_normal_log_prob, get_bin_log_prob, kl_divergence, sample_categorical
from networks import initialize_ma_dreamer_optimizers, DreamerMAOptimizers, SequenceModel, JointIsetEncoder, IsetDecoder, DynamicsPredictor, Predictor, LegalActionsNetwork
from games.jax_game import JaxGame, GameState


@chex.dataclass(frozen=True)
class DreamerMAGradients():
  sequence: nnx.State
  encoder: nnx.State
  p1_decoder: nnx.State
  p2_decoder: nnx.State
  dynamics: nnx.State
  predictor: nnx.State
  legal_predictor: nnx.State


class DreamerMA():
  """The actual model that handles the dreamer algorithm training.
  For now only the world model networks are used and actor/critic networks are not trained."""
  def __init__(self, config: DreamerMAConfig, game: JaxGame):
    self.config = config
    self.game = game
    self.init()
    
    
  def init(self):
    self.jax_rngs = jax.random.key(self.config.rng_seed)
    self.nnx_rngs = nnx.Rngs(self.generate_key())
    
    
    self.optimizers = initialize_ma_dreamer_optimizers(self.config, self.game, self.nnx_rngs)
    self.learner_steps = 0
    
    #Assuming that this contains a terminal state as well
    self.trajectory_max = self.game.max_trajectory_length()
    self.non_chance_trajectory_max = self.game.max_trajectory_lenght_no_chance()

    self.action_dimension = self.game.num_distinct_actions()
    assert self.game.num_players() > 1, f"This implementation of Dreamer assumes a game with at least 2 players not {self.game.num_players()}"
    
    self._get_example_timestep()
    self.cached_train = nnx.cached_partial(self.world_model_train, self.optimizers)
    
  
  def _get_example_timestep(self):
    #This can produce a chance node, but that 
    # one by default produces invalid isets
    # and legals so it is not a problem 
    example_state, example_legals = self.game.initialize_structures()
    _, ex_p1_iset, ex_p2_iset, _ = self.game.get_info(example_state)
    ex_obs = jnp.stack([ex_p1_iset, ex_p2_iset], axis=0)
    legal = jnp.ones_like(example_legals)
    action = jax.nn.one_hot(jnp.argmax(legal, -1), legal.shape[-1]) 
    policy = legal.astype(float) / jnp.sum(legal, axis=-1, keepdims=True)
    self.example_timestep = TimeStep(
                                    obs= ex_obs,
                                    action=action,
                                    legal=legal,
                                    policy = policy,
                                    reward = 0.0,
                                    terminal = False,
                                    valid = False)
  
  def generate_key(self):
    self.jax_rngs, key = jax.random.split(self.jax_rngs)
    return key

  def generate_keys(self, num_keys):
    split_key = self.generate_key()
    keys = jax.random.split(split_key, num_keys)
    return keys
    

  @chex.assert_max_traces(n=1)
  @partial(nnx.jit, static_argnums=(0, 1))
  def sample_trajectories(self,  batch_size, key):
    keys = jax.random.split(key, batch_size)
    return nnx.vmap(self.sample_trajectory, in_axes=0, out_axes=1)(keys)
  

  @chex.assert_max_traces(n=1)
  @partial(nnx.jit, static_argnums=0)
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
    
    
    @nnx.jit
    def choice_wrapper(key, p):
      action = jax.random.choice(key, actions, p=p)
      action_oh = jax.nn.one_hot(action, actions)
      return action, action_oh

    
    vectorized_sample_action = nnx.vmap(choice_wrapper, in_axes=(0, 0), out_axes=0)

    @nnx.scan(in_axes = (nnx.Carry, 0), out_axes=(nnx.Carry, 0, 0))
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
        legal = carry.legal_actions,
        action = action_oh,
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
        
      
      timestep = jax.tree.map(lambda t, f: jnp.where(carry.valid, t, f), timestep, self.example_timestep)
      
      return new_carry, timestep, is_chance
    _, timestep, is_chance = _sample_trajectory(init_carry, trajectory_key)
    #This is used to remove the chance nodes from the trajectory
    non_chance = jnp.nonzero(~is_chance, size=self.non_chance_trajectory_max)[0]
    filtered_timestep = jax.tree_util.tree_map(lambda x: jnp.take_along_axis(x, jnp.expand_dims(non_chance, axis=range(1, x.ndim)), axis=0), timestep)
    #[Trajectory, ...]
    return filtered_timestep
  
  def update_world_model(self, optimizers: DreamerMAOptimizers, timestep: TimeStep, rng_key):
    """Compound loss for the entire world model."""
    sample_keys = jax.random.split(rng_key, self.non_chance_trajectory_max * self.config.batch_size)
    sample_keys = sample_keys.reshape((self.non_chance_trajectory_max, self.config.batch_size))
    
    def world_model_loss(sequence_model: SequenceModel, encoder: JointIsetEncoder, 
                         p1_decoder: IsetDecoder, p2_decoder: IsetDecoder, 
                         dynamics_model: DynamicsPredictor, predictor: Predictor,
                         legal_actions_network: LegalActionsNetwork):
      l_pred, l_dyn, l_rep = 0, 0, 0
      #[Trajectory, Batch, ...]
      @nnx.scan(in_axes=(nnx.Carry, 0, None, None, None, None, None, None, None), out_axes=(nnx.Carry, 0))
      def _predict_over_timestep(hidden_state, xs, sequence_model, encoder, p1_decoder, p2_decoder, dynamics_model, predictor, legal_network):
        
        action, obs, cur_key = xs
        stochastic_state = encoder(hidden_state, obs)
        deterministic_state = sample_categorical(stochastic_state, cur_key)
        prior_stochastic_state = dynamics_model(hidden_state)
        reward, done = predictor(hidden_state, deterministic_state)
        legal = legal_network(hidden_state, deterministic_state)
        decoded_p1_obs = p1_decoder(hidden_state, deterministic_state)
        decoded_p2_obs = p2_decoder(hidden_state, deterministic_state)
        decoded_obs = jnp.stack([decoded_p1_obs, decoded_p2_obs], axis=0)
        flattened_action = jnp.reshape(action, (*deterministic_state.shape[:-2], -1))
        gru_input = jnp.concatenate([deterministic_state.reshape(*deterministic_state.shape[:-2], -1), flattened_action], axis=-1) 
        new_hidden = sequence_model(hidden_state, gru_input)
        preds = PredictionStepWithLegal(repr_state = stochastic_state,
                                decoded_obs = decoded_obs,
                                reward_dist_logit = reward,
                                done_logit = done,
                                legal_logit = legal,
                                dynamics_state = prior_stochastic_state) 
        
        return new_hidden, preds
      
      xs = (timestep.action, timestep.obs, sample_keys)
      init_hidden = jnp.zeros((self.config.batch_size, self.config.hidden_state_size)) 
      vectorized_predict = nnx.vmap(_predict_over_timestep, in_axes=(0, 1, None, None, None, None, None, None, None), out_axes=(0, 1))
      _, predictions = vectorized_predict(init_hidden, xs, sequence_model, encoder, p1_decoder, p2_decoder, dynamics_model, predictor, legal_actions_network) 

      #[Trajectory, Batch, num_players, obs_size]
      reconstruction_loss = -get_normal_log_prob(predictions.decoded_obs, timestep.obs)
      l_pred += get_loss_mean_with_mask(reconstruction_loss, timestep.valid[..., None, None])
      #[Trajectory, Batch, 1]
      #continuation_loss = -get_normal_log_prob(predictions.done_logit, timestep.terminal.astype(jnp.int16))
      continuation_loss = optax.sigmoid_binary_cross_entropy(predictions.done_logit, timestep.terminal[..., None])
      l_pred += get_loss_mean_with_mask(continuation_loss, timestep.valid[..., None])
      #[Trajectory, Batch, players, action_dim]
      legal_loss = optax.sigmoid_binary_cross_entropy(predictions.legal_logit, timestep.legal)
      #Legal actions should not be trained in terminal states, as there are no legal actions there
      l_pred += get_loss_mean_with_mask(legal_loss, ~timestep.terminal[..., None, None])
      #[Trajectory, Batch, 2* bin_range + 1]
      bins = jnp.arange((2 * self.config.bin_range) + 1) - self.config.bin_range
      reward_loss = -get_bin_log_prob(predictions.reward_dist_logit, bins, timestep.reward, use_symlog=True)
      #[Trajectory, Batch, 1]
      #reward_loss = -get_normal_log_prob(timestep.reward, timestep.reward)
      l_pred += get_loss_mean_with_mask(reward_loss, timestep.valid[..., None])

      #Using free bits to clip dynamics and representation losses
      # thus disabling their gradient when they are below free_bits_clip_threshold
      #[Trajectory, Batch, encoded_categories, encoded_classes]

      posterior = nnx.softmax(predictions.repr_state, axis=-1)
      prior = nnx.softmax(predictions.dynamics_state, axis=-1)
      #[Trajectory, Batch]
      dynamics_loss = kl_divergence(jax.lax.stop_gradient(posterior), prior)
      l_dyn += jnp.maximum(self.config.free_bits_clip_threshold, get_loss_mean_with_mask(dynamics_loss, timestep.valid))
      #[Trajectory, Batch]
      repr_loss = kl_divergence(posterior, jax.lax.stop_gradient(prior))
      l_rep += jnp.maximum(self.config.free_bits_clip_threshold, get_loss_mean_with_mask(repr_loss, timestep.valid))

      return self.config.beta_prediction * l_pred + self.config.beta_dynamics * l_dyn + self.config.beta_representation * l_rep
  
    loss, grad = nnx.value_and_grad(world_model_loss, argnums=(0, 1, 2, 3, 4, 5, 6))(optimizers.sequence_optimizer.model, optimizers.encoder_optimizer.model, optimizers.p1_decoder_optimizer.model, 
                                                                                  optimizers.p2_decoder_optimizer.model, optimizers.dynamics_optimizer.model, optimizers.predictor_optimizer.model,
                                                                                  optimizers.legal_actions_optimizer.model)
    
    optimizers.sequence_optimizer.update(grad[0])
    optimizers.encoder_optimizer.update(grad[1])
    optimizers.p1_decoder_optimizer.update(grad[2])
    optimizers.p2_decoder_optimizer.update(grad[3])
    optimizers.dynamics_optimizer.update(grad[4])
    optimizers.predictor_optimizer.update(grad[5])
    optimizers.legal_actions_optimizer.update(grad[6])
    
    return loss
  

  @partial(nnx.jit, static_argnums=(0))
  def update_optimizers_with_grads(self, optimizers: DreamerMAOptimizers, grad: DreamerMAGradients):
    """Update the world model with the computed grad dictionary.
    """
    optimizers.sequence_optimizer.update(grad.sequence)
    optimizers.encoder_optimizer.update(grad.encoder)
    optimizers.p1_decoder_optimizer.update(grad.p1_decoder)
    optimizers.p2_decoder_optimizer.update(grad.p2_decoder)
    optimizers.dynamics_optimizer.update(grad.dynamics)
    optimizers.predictor_optimizer.update(grad.predictor)
    optimizers.legal_actions_optimizer.update(grad.legal_predictor) 
  
  
  # Unlike flax.linen, nnx.jit allows updating the model itself.
  @partial(nnx.jit, static_argnums=(0))
  def world_model_train(self, optimizers, rng_key):
    trajectory_key, train_key = jax.random.split(rng_key)
    timestep = self.sample_trajectories(self.config.batch_size, trajectory_key)
    loss = self.update_world_model(optimizers,timestep, train_key)
    return loss
    

  def world_model_train_step(self):
    rng_key = self.generate_key()
    #return self.world_model_train(self.optimizers, rng_key)
    loss = self.cached_train(rng_key)
    self.learner_steps += 1
    return loss

  def train_world_model(self, model_save_dir:str, num_steps:int, print_each: int = -1, save_each: int = -1):
     
    for i in range(num_steps):
      rng_key = self.generate_key() 
      loss = self.cached_train(rng_key)
      if print_each > 0 and i % print_each == 0:
        print(f"Step {i}, Loss: {loss}")
      if save_each > 0 and i % save_each == 0:
        model_file = model_save_dir + f"step_{i}.pkl"
        save_model(self, model_file)
      self.learner_steps += 1
   
  def __getstate__(self):
    return {
      "config": self.config,
      "game": self.game,
      "jax_rngs": self.jax_rngs,
      "optimizers": nnx.state(self.optimizers),
      "steps": self.learner_steps
    }
    

  def update_nnx(self, model_state, saved_state):
    static_graph, _ = nnx.split(model_state)
    new_model_state = nnx.merge(static_graph, saved_state)
    return new_model_state
  
  def __setstate__(self, state):
    self.config = state["config"]
    self.game = state["game"]
    
    self.init()
    
    self.jax_rngs = state["jax_rngs"]
    self.optimizers = self.update_nnx(self.optimizers, state["optimizers"])
    self.learner_steps = state["steps"]

  @partial(nnx.jit, static_argnums=(0, 5, 6))
  def get_predictor(self, predictor_model: Predictor, legal_model: LegalActionsNetwork, 
                    hidden_state: chex.Array, deterministic_state:chex.Array, terminal_threshold: float = 0.5, legal_threshold: float = 0.5):
    """Calls the predictor and legal actions networks and 
    passes the reward, done logits and legal action logits through
    appropriate transformations to return the actual values"""
    #[2* bin_range + 1], [1]
    reward_bin_logits, done_logit = predictor_model(hidden_state, deterministic_state)
    legal_logit = legal_model(hidden_state, deterministic_state)
    #Implementing the summation order suggestion
    # from https://arxiv.org/pdf/2301.04104 page 18
    bins = jnp.arange((2 * self.config.bin_range) + 1) - self.config.bin_range
    reward_probs = nnx.softmax(reward_bin_logits)
    pos_bins = bins * (bins >= 0)
    # flip the probs and bins for the negative
    # to ensure summation from small to large in magnitude 
    neg_bins = bins * (bins < 0)
    reward_pos_part = jnp.sum(reward_probs * pos_bins)
    reward_neg_part = jnp.sum(jnp.flip(reward_probs * neg_bins))
    reward = reward_pos_part + reward_neg_part
    #reward = jnp.sum(reward_probs * bins)
    reward = symexp(reward)
    #reward_untransformed, done_logit = predictor_model(hidden_state, deterministic_state)
    #reward = reward_untransformed
    done_prob = nnx.sigmoid(done_logit)
    terminal = done_prob >= terminal_threshold
    legal_prob = nnx.sigmoid(legal_logit)
    legal_actions = legal_prob >= legal_threshold
    return reward, terminal[0], legal_actions
  
  @partial(nnx.jit, static_argnums=(0))
  def get_decoder(self, decoder_model: IsetDecoder, hidden_state: chex.Array, deterministic_state: chex.Array):
    """Calls the decoder network and 
    applies the appropriate transformation to its output.
    Outputs either predicted real observation in single agent setting, or 
    predicted iset for a single player in a multi agent setting. """
    decoder_output_untransformed = decoder_model(hidden_state, deterministic_state)
    #decoder_output = symexp(decoder_output_untransformed)
    decoder_output = decoder_output_untransformed
    return decoder_output
  
  @partial(nnx.jit, static_argnums=(0))
  def get_dynamics(self, dynamics_model: DynamicsPredictor, hidden_state:chex.Array):
    return dynamics_model(hidden_state)
  
  @partial(nnx.jit, static_argnums=(0))
  def get_encoder(self, encoder_model:JointIsetEncoder, hidden_state:chex.Array, obs: chex.Array):
    return encoder_model(hidden_state, obs)
  
  @partial(nnx.jit, static_argnums=(0))
  def get_next_hidden(self, sequence_model:SequenceModel, hidden_state:chex.Array, deterministic_state:chex.Array, joint_action:chex.Array):
    flattened_action = jnp.reshape(joint_action, (*deterministic_state.shape[:-2], -1))
    gru_input = jnp.concatenate([deterministic_state.reshape(*deterministic_state.shape[:-2], -1), flattened_action], axis=-1) 
    return sequence_model(hidden_state, gru_input)
    
    