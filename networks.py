
import chex
import jax
from flax import nnx
import jax.numpy as jnp
from train_utils import DreamerConfig
from functools import partial
import optax
from games.jax_game import JaxGame

from train_utils import TimeStep, PredictionStep, get_loss_mean_with_mask
from distributions import sample_categorical, get_normal_log_prob, get_bin_log_prob, kl_divergence



class LinNormRelu(nnx.Module):
  def __init__(self, in_features, out_features,rngs: nnx.Rngs):
    self.linear = nnx.Linear(in_features, out_features, rngs=rngs)
    self.norm = nnx.LayerNorm(out_features, rngs=rngs)
    
  def __call__(self, x: chex.Array):
    x = self.linear(x)
    x = self.norm(x)
    return nnx.silu(x)

class HiddenMLP(nnx.Module):
  '''
    Multi-layered perceptron, which has first layer that rescales the input to the hidden_features, and then several layers with the same hidden_features.
  '''
  def __init__(self, hidden_features, num_layers, rngs: nnx.Rngs):
    
    # Taken from https://flax.readthedocs.io/en/latest/guides/linen_to_nnx.html. It should speed up the compilation, because it tells the model that each hidden layer is the same.
    @nnx.split_rngs(splits=num_layers)
    @nnx.vmap(in_axes=(0,), out_axes=0)
    def create_block(rngs: nnx.Rngs):
      return LinNormRelu(hidden_features, hidden_features, rngs)
    
    self.num_layers = num_layers 
    self.hidden_layers = create_block(rngs) 
    

  def __call__(self, x: chex.Array): 
    # Taken from https://flax.readthedocs.io/en/latest/guides/linen_to_nnx.html. It should speed up the compilation, because it tells the model that each hidden layer is the same
    @nnx.split_rngs(splits=self.num_layers)
    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
    def forward_hidden(x, model):
      x = model(x)
      return x
    
    return forward_hidden(x, self.hidden_layers) 


class SequenceModel(nnx.Module):
  '''
    Used to produce the next state of the game from the hidden state and the action
  '''
  def __init__(self, encoded_classes, encoded_categories, action_features, hidden_state_size, rngs: nnx.Rngs):
    self.gru_cell = nnx.GRUCell(encoded_classes * encoded_categories + action_features, hidden_state_size, gate_fn=nnx.silu, rngs=rngs) 
    #TODO: I think we actually do not want to use RNN here
    # as it should just wrap the GRU cells into a sequence, 
    # but as we require also to predict the deterministic state in the current sequence
    # to get the next one, I think we want to process the sequence manually with scan
    #self.rnn = nnx.RNN(self.gru_cell, return_carry=True, time_major=True, rngs=rngs)
    
  def __call__(self, hidden_state: chex.Array, gru_input: chex.Array):
    """gru_input is concatend deterministic state and action"""
    # TODO: Shall we first pass through MLP before the RNN?
    new_hidden_state, _ = self.gru_cell(hidden_state, gru_input)
    return new_hidden_state
    #return self.rnn(hidden_state,gru_input) 
  
  
class Encoder(nnx.Module):
  """Recieve an observation from the environment,
  return logits of current stochastic latent state."""
  def __init__(self, hidden_state_size, observation_features, encoded_classes, encoded_categories, hidden_features, num_layers, rngs: nnx.Rngs) -> None:
    self.encoded_classes = encoded_classes
    self.encoded_categories = encoded_categories
    self.init_layer = LinNormRelu(hidden_state_size + observation_features, hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    self.last_layer = nnx.Linear(hidden_features, encoded_classes * encoded_categories, rngs=rngs)
    
  def __call__(self, hidden_state: chex.Array, observation: chex.Array):
    x = jnp.concatenate([hidden_state, observation], axis=-1)
    x = self.init_layer(x)
    x = self.core_mlp(x)
    x = self.last_layer(x)
    encoded_state = x.reshape(*x.shape[:-1], self.encoded_classes, self.encoded_categories)
    return encoded_state

class Decoder(nnx.Module):
  """Receive a current deterministic latent state (eg. a encoded_categories-hot vector)
  and return a reconstruction of current observation."""
  def __init__(self, hidden_state_size, observation_features, encoded_classes, encoded_categories, hidden_features, num_layers, rngs: nnx.Rngs) -> None:
    self.encoded_classes = encoded_classes
    self.encoded_categories = encoded_categories
    self.init_layer = LinNormRelu(hidden_state_size + encoded_classes * encoded_categories, hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    self.last_layer = nnx.Linear(hidden_features, observation_features, rngs=rngs)
    
  def __call__(self, hidden_state: chex.Array, encoded_state: chex.Array):
    x = jnp.concatenate([hidden_state, encoded_state.reshape(*encoded_state.shape[:-2], -1)], axis=-1)
    x = self.init_layer(x)
    x = self.core_mlp(x)
    obs = self.last_layer(x)
    return obs
  
# TODO: Not sure whether the SequenceModel does not do the same things
class DynamicsPredictor(nnx.Module):
  """Receive a current deterministic latent state (eg. a encoded_categories-hot vector)
  and return the next stochastic environment state logits"""
  def __init__(self, hidden_state_size, encoded_classes, encoded_categories, hidden_features, num_layers, rngs: nnx.Rngs) -> None:
    self.encoded_classes = encoded_classes
    self.encoded_categories = encoded_categories
    self.init_layer = LinNormRelu(hidden_state_size, hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    self.last_layer = nnx.Linear(hidden_features, encoded_classes * encoded_categories, rngs=rngs)
    
  def __call__(self, hidden_state: chex.Array):
    x = self.init_layer(hidden_state)
    x = self.core_mlp(x)
    x = self.last_layer(x)
    encoded_state = x.reshape(*x.shape[:-1], self.encoded_classes, self.encoded_categories)
    return encoded_state
  
  
class Predictor(nnx.Module):
  """Receive a current deterministic latent state (eg. a encoded_categories-hot vector)
  and return logits of the predicted reward bin_distribution and done flag, ordered as such.
  Pass the done logits through sigmoid and compare against a threshold if you want
  to obtain an actual done flag. The reward are logits of a distribution over the exponentially
  spaced bins like symexp([-bin_range, bin_range])""" 
  def __init__(self, hidden_state_size, encoded_classes, encoded_categories, bin_range, hidden_features, num_layers, rngs: nnx.Rngs) -> None:
    self.init_layer = LinNormRelu(hidden_state_size + encoded_classes * encoded_categories, hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    self.reward_layer = nnx.Linear(hidden_features, (2 * bin_range) + 1, rngs=rngs)
    self.done_layer = nnx.Linear(hidden_features, 1, rngs=rngs)
    
  def __call__(self, hidden_state: chex.Array, encoded_state: chex.Array):
    x = jnp.concatenate([hidden_state, encoded_state.reshape(*encoded_state.shape[:-2], -1)], axis=-1)
    x = self.init_layer(x)
    x = self.core_mlp(x)
    reward = self.reward_layer(x)
    done = self.done_layer(x)
    return reward, done
  
  
@chex.dataclass(frozen=True)
class DreamerOptimizers():
  sequence_optimizer: nnx.Optimizer
  encoder_optimizer: nnx.Optimizer
  decoder_optimizer: nnx.Optimizer
  dynamics_optimizer: nnx.Optimizer
  predictor_optimizer: nnx.Optimizer


@chex.dataclass(frozen=True)
class WorldModelNetworks():
  sequence_network: SequenceModel
  encoder_network: Encoder
  decoder_network: Decoder
  dynamics_network: DynamicsPredictor
  predictor_network: Predictor

    
class DreamerModel(nnx.Module):
  """Container for all the Dreamer networks.
  The optimizers are also contained within this module, so that a single 
  split gives a state of all the networks for checkpointing"""
  def __init__(self, world_model_networks: WorldModelNetworks, optims: DreamerOptimizers, config:DreamerConfig, trajectory_max:int):
    self.sequence_network = world_model_networks.sequence_network
    self.encoder_network = world_model_networks.encoder_network
    self.decoder_network = world_model_networks.decoder_network
    self.dynamics_network = world_model_networks.dynamics_network
    self.predictor_network = world_model_networks.predictor_network

    self.sequence_optimizer = optims.sequence_optimizer
    self.encoder_optimizer = optims.encoder_optimizer
    self.decoder_optimizer = optims.decoder_optimizer
    self.dynamics_optimizer = optims.dynamics_optimizer
    self.predictor_optimizer = optims.predictor_optimizer

    self.beta_prediction = config.beta_prediction
    self.beta_dynamics = config.beta_dynamics
    self.beta_representation = config.beta_representation


    self.trajectory_max = trajectory_max
    self.batch_size = config.batch_size
    self.hidden_state_size = config.hidden_state_size

    #Create an array with elements from [-bin_range, ..., bin_range] bounds inclusive
    self.bins = jnp.arange((2 * config.bin_range) + 1) -config.bin_range
    
      

  @nnx.jit
  def __call__(self, timestep: TimeStep, sample_key:jax.random.PRNGKey):
    """Perform a world model prediction step over the whole batch and trajectory, given
    the collected trajectories timestep and a single PRNG key, to handle sampling from categoricals"""
    #initial_hidden = jnp.tile(self.sequence_network.init_hidden, (self.batch_size, 1))
    initial_hidden = jnp.zeros((self.batch_size, self.hidden_state_size))
    sample_keys = jax.random.split(sample_key, self.trajectory_max * self.batch_size)
    sample_keys = sample_keys.reshape((self.trajectory_max, self.batch_size))

    xs = (timestep.action, timestep.obs, sample_keys)
    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def _predict_over_timestep(carry, xs):
      model, hidden_state = carry
      action, obs, cur_key = xs
      stochastic_state = model.encoder_network(hidden_state, obs)
      deterministic_state = sample_categorical(stochastic_state, cur_key)
      next_stochastic_state = model.dynamics_network(hidden_state)
      reward, done = model.predictor_network(hidden_state, deterministic_state)
      decoded_obs = model.decoder_network(hidden_state, deterministic_state)
      gru_input = jnp.concatenate([deterministic_state.reshape(*deterministic_state.shape[:-2], -1), action], axis=-1) 
      new_hidden = model.sequence_network(hidden_state, gru_input)
      preds = PredictionStep(repr_state = stochastic_state,
                              decoded_obs = decoded_obs,
                              reward_dist_logit = reward,
                              done_logit = done,
                              dynamics_state = next_stochastic_state)
      new_carry = (model, new_hidden)
      return new_carry, preds
    init_carry = (self, initial_hidden)
    vectorized_predict = nnx.vmap(_predict_over_timestep, in_axes=((None, 0), (1, 1, 1)), out_axes=((None, 0), 1))
    #final_hidden, all_preds = _predict_over_timestep(init_carry, xs)
    final_hidden, all_preds = vectorized_predict(init_carry, xs)
    return all_preds

  def world_model_loss(self, timestep: TimeStep, key:jax.random.PRNGKey):
    """Compound loss for the entire world model."""
    l_pred, l_dyn, l_rep = 0, 0, 0
    #[Trajectory, Batch, ...]
    predictions = self(timestep, key)
    #jax.debug.breakpoint()
    #[Trajectory, Batch, obs_size]
    reconstruction_loss = -get_normal_log_prob(predictions.decoded_obs, timestep.obs, use_symlog=True)
    l_pred += get_loss_mean_with_mask(reconstruction_loss, timestep.valid[..., None])
    #[Trajectory, Batch, 1]
    continuation_loss = -get_normal_log_prob(predictions.done_logit, timestep.terminal.astype(int))
    l_pred += get_loss_mean_with_mask(continuation_loss, timestep.valid[..., None])
    #[Trajectory, Batch, 2* bin_range + 1]
    reward_loss = -get_bin_log_prob(predictions.reward_dist_logit, self.bins, timestep.reward, use_symlog=True)
    l_pred += get_loss_mean_with_mask(reward_loss, timestep.valid[..., None])

    #Using free bits to clip dynamics and representation losses
    # to 1, thus disabling their gradient when they are below 1
    #[Trajectory, Batch, encoded_categories, encoded_classes]
    posterior = nnx.softmax(predictions.repr_state, axis=-1)
    prior = nnx.softmax(predictions.dynamics_state, axis=-1)
    #[Trajectory, Batch]
    dynamics_loss = jnp.maximum(1, kl_divergence(jax.lax.stop_gradient(posterior), prior))
    l_dyn += get_loss_mean_with_mask(dynamics_loss, timestep.valid)
    #[Trajectory, Batch]
    repr_loss = jnp.maximum(1, kl_divergence(posterior, jax.lax.stop_gradient(prior)))
    l_rep += get_loss_mean_with_mask(repr_loss, timestep.valid)


    return self.beta_prediction * l_pred + self.beta_dynamics * l_dyn + self.beta_representation * l_rep


  @nnx.jit
  def world_model_step(self, timestep: TimeStep, key:jax.random.PRNGKey)->float:
    """Wrapper performing a single gradient step on the world model."""
    def _loss_fn(model):
      return model.world_model_loss(timestep, key)
    loss, all_grads = nnx.value_and_grad(_loss_fn)(self)
    #TODO: Do we want a separate optimizer for each network
    # or a single joint optimizer for the whole model?
    self.sequence_optimizer.update(all_grads.sequence_network)
    self.encoder_optimizer.update(all_grads.encoder_network)
    self.decoder_optimizer.update(all_grads.decoder_network)
    self.dynamics_optimizer.update(all_grads.dynamics_network)
    self.predictor_optimizer.update(all_grads.predictor_network)
    return loss



 
  
  
def create_networks_and_optimizers(config: DreamerConfig, game: JaxGame, rngs: nnx.Rngs) ->DreamerModel:
  """Initializes the model networks and optimizers. For now actor and critic networks are not used"""
  sequence_model =  SequenceModel(
        encoded_classes=config.encoded_classes,
        encoded_categories=config.encoded_categories,
        action_features=game.num_distinct_actions(),
        hidden_state_size=config.hidden_state_size,
        rngs=rngs
    )

  sequence_optimizer = nnx.Optimizer(
    model= sequence_model,
    tx=optax.adam(learning_rate=config.learning_rate),
  )
  encoder = Encoder(
        hidden_state_size=config.hidden_state_size,
        observation_features=game.observation_tensor_shape(),
        encoded_classes=config.encoded_classes,
        encoded_categories=config.encoded_categories,
        hidden_features=config.encoder_network_details[0],
        num_layers=config.encoder_network_details[1],
        rngs=rngs
    )

  encoder_optimizer = nnx.Optimizer(
    model= encoder,
    tx=optax.adam(learning_rate=config.learning_rate),
  )

  decoder = Decoder(
        hidden_state_size=config.hidden_state_size,
        observation_features=game.observation_tensor_shape(),
        encoded_classes=config.encoded_classes,
        encoded_categories=config.encoded_categories,
        hidden_features=config.decoder_network_details[0],
        num_layers=config.decoder_network_details[1],
        rngs=rngs
    )
  
  decoder_optimizer = nnx.Optimizer(
    model= decoder,
    tx=optax.adam(learning_rate=config.learning_rate),
  )
  dynamics = DynamicsPredictor(
        hidden_state_size=config.hidden_state_size,
        encoded_classes=config.encoded_classes,
        encoded_categories=config.encoded_categories,
        hidden_features=config.dynamics_network_details[0],
        num_layers=config.dynamics_network_details[1],
        rngs=rngs
    )

  dynamics_optimizer = nnx.Optimizer(
    model= dynamics,
    tx=optax.adam(learning_rate=config.learning_rate),
  )
  predictor = Predictor(
        hidden_state_size=config.hidden_state_size,
        encoded_classes=config.encoded_classes,
        encoded_categories=config.encoded_categories,
        bin_range= config.bin_range,
        hidden_features=config.predictor_network_details[0],
        num_layers=config.predictor_network_details[1],
        rngs=rngs
    )

  predictor_optimizer = nnx.Optimizer(
    model= predictor,
    tx=optax.adam(learning_rate=config.learning_rate),
  )
  world_model_networks = WorldModelNetworks(
    sequence_network=sequence_model,
    encoder_network=encoder,
    decoder_network=decoder,
    dynamics_network=dynamics,
    predictor_network=predictor
  )
  optims = DreamerOptimizers(
    sequence_optimizer=sequence_optimizer,
    encoder_optimizer=encoder_optimizer,
    decoder_optimizer=decoder_optimizer,
    dynamics_optimizer=dynamics_optimizer,
    predictor_optimizer=predictor_optimizer
  )

  return DreamerModel(world_model_networks, optims, config, game.max_trajectory_length())