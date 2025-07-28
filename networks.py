
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
    self.rnn = nnx.RNN(self.gru_cell, return_carry=True, time_major=True, rngs=rngs)
    #For now the initial hidden state is set to zero
    zero_initializer = nnx.initializers.constant(0, dtype=jnp.float32)
    #This is a constant initializer so any key will do
    self.init_hidden = zero_initializer(jax.random.key(0), hidden_state_size, dtype=jnp.float32)
    
  def __call__(self, hidden_state: chex.Array, gru_input: chex.Array):
    #gru_input is concatenated encoded state and action
    # TODO: Shall we first pass through MLP before the RNN?
    return self.rnn(hidden_state,gru_input) 
  
  
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
    self.reward_layer = nnx.Linear(hidden_features, 1, rngs=rngs)
    self.done_layer = nnx.Linear(hidden_features, (2 * bin_range) + 1, rngs=rngs)
    
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

    #Create an array with elements from [-bin_range, ..., bin_range] bounds inclusive
    self.bins = jnp.arange((2 * config.bin_range) + 1) -config.bin_range


  @staticmethod
  def _predict_one(carry, xs, encoder_network: Encoder, dynamics_network: DynamicsPredictor, 
                   predictor_network: Predictor, decoder_network: Decoder, sequence_network: SequenceModel):
      action, obs, cur_key = xs
      stochastic_state = encoder_network(carry, obs)
      deterministic_state = sample_categorical(stochastic_state, cur_key)
      next_stochastic_state = dynamics_network(carry)
      reward, done = predictor_network(carry, deterministic_state)
      decoded_obs = decoder_network(carry, deterministic_state)
      gru_input = jnp.concatenate([deterministic_state.reshape(*deterministic_state.shape[:-2], -1), action], axis=-1) 
      new_carry = sequence_network(carry, gru_input)
      preds = PredictionStep(repr_state = stochastic_state,
                             decoded_obs = decoded_obs,
                             reward_dist_logit = reward,
                             done_logit = done,
                             dynamics_state = next_stochastic_state)
      return new_carry, preds
  
  
  


  # @nnx.jit
  # def get_predictions(self, timestep:TimeStep, key) ->PredictionStep:
  #   """Process the model predictions over the entire trajectory.
  #   TODO: Due to the sequence model, I dont think it can be paralelized 
  #   completely so for now scan over the trajectory"""

  #   batch_size = key.shape[0]
  #   vectorized_split = jax.vmap(jax.random.split, in_axes=(0, None), out_axes=(1))

    
  #   @nnx.vmap(in_axes=(None, 0, (0, 0, 0)), out_axes=(0, 0))
  #   def _predict_worker(self, carry, xs):
  #     return self._predict_one(carry, xs)
  #   #Should produce an array of [Trajectory, Batch] PRNG keys
  #   trajectory_key = vectorized_split(key, self.trajectory_max)
  #   init_carry = jnp.tile(self.sequence_network.init_hidden, (batch_size, 1))

  #   worker_bound_self = partial(_predict_worker, self)
  #   final_hidden, predictions = jax.lax.scan(worker_bound_self, init_carry,
  #                                            xs = (timestep.action, timestep.obs, trajectory_key))
  #   return predictions
  
    # vectorized_sample = jax.vmap(sample_categorical, in_axes=(0, 0), out_axes=(0))
    
    # vectorized_next_hidden = jax.vmap(self.sequence_network, in_axes=(0, 0), out_axes=(0))
    # vectorized_predict_dynamics = jax.vmap(self.dynamics_network, in_axes=(0), out_axes=(0))
    # vectorized_predictor = jax.vmap(self.predictor_network, in_axes=(0, 0), out_axes=(0, 0,))
    # vectorized_encode = jax.vmap(self.encoder_network, in_axes=(0, 0), out_axes=(0))
    # vectorized_decode = jax.vmap(self.decoder_network, in_axes=(0, 0), out_axes=(0))
    # all_preds = []
    # carry = init_carry
    # for i in range(self.trajectory_max):
    #   action, obs, cur_key = (timestep.action[i], timestep.obs[i], trajectory_key[i])
    #   stochastic_state = vectorized_encode(carry, obs)
    #   #Should take [Batch] random keys and [Batch, num_categoricals, num_classes]
    #   # stochastic states and produce [Batch, num_categoricals, num_classes]
    #   # deterministic state
    #   deterministic_state = vectorized_sample(stochastic_state, cur_key)
    #   next_stochastic_state = vectorized_predict_dynamics(carry)
    #   reward, done = vectorized_predictor(carry, deterministic_state)
    #   decoded_obs = vectorized_decode(carry, deterministic_state)
    #   gru_input = jnp.concatenate([deterministic_state.reshape(*deterministic_state.shape[:-2], -1), action], axis=-1) 
    #   carry = vectorized_next_hidden(carry, gru_input)
    #   preds = PredictionStep(repr_state = stochastic_state,
    #                          decoded_obs = decoded_obs,
    #                          reward_dist_logit = reward,
    #                          done_logit = done,
    #                          dynamics_state = next_stochastic_state)
    #   all_preds.append(preds)
    # predictions = jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0), *all_preds)
    
    #return predictions

  @nnx.jit
  def __call__(self, timestep: TimeStep, rngs:nnx.Rngs):
    """FIXME: This should return predictions over the entire model
    using scan over trajectory. This just does not work for now, 
    the vmap and scan together keep causing conflicts."""
    initial_hidden = jnp.tile(self.sequence_network.init_hidden, (self.batch_size, 1))
    sample_key = rngs._get_stream("sampling", KeyError)()
    sample_keys = jax.random.split(sample_key, self.trajectory_max * self.batch_size)
    sample_keys = sample_keys.reshape((self.trajectory_max, self.batch_size))

    partial_predict_one_fn = jax.tree_util.Partial(
            DreamerModel._predict_one,
            encoder_network=self.encoder_network,
            dynamics_network=self.dynamics_network,
            predictor_network=self.predictor_network,
            decoder_network=self.decoder_network,
            sequence_network = self.sequence_network
        )
    
    xs = (timestep.action, timestep.obs, sample_keys)
    vectorized_predict = jax.vmap(partial_predict_one_fn, in_axes=(0, 0), out_axes=(0, 0))
    scanned_predict_step = nnx.scan(
            vectorized_predict,               
            length=self.trajectory_max            
        )

    final_hidden, all_preds = scanned_predict_step(
        initial_hidden,  
        xs,  
    )
    return all_preds

  def world_model_loss(self, timestep: TimeStep, rngs:nnx.Rngs):
    """Compound loss for the entire world model."""
    l_pred, l_dyn, l_rep = 0, 0, 0
    #[Trajectory, Batch, ...]
    predictions = self(timestep, rngs)
    jax.debug.breakpoint()
    #[Trajectory, Batch, obs_size]
    reconstruction_loss = -get_normal_log_prob(predictions.decoded_obs, timestep.obs, use_symlog=True)
    l_pred += get_loss_mean_with_mask(reconstruction_loss, timestep.valid[..., None])
    #[Trajectory, Batch, 1]
    continuation_loss = -get_normal_log_prob(predictions.done_logit, timestep.terminal)
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
  def world_model_step(self, timestep: TimeStep, rngs: nnx.Rngs)->float:
    """Wrapper performing a single gradient step on the world model."""
    def _loss_fn(model):
      return model.world_model_loss(timestep, rngs)
    loss, all_grads = nnx.value_and_grad(_loss_fn)(self)
    #TODO: Do we want a separate optimizer for each network
    # or a single joint optimizer for the whole model?
    self.sequence_optimizer.update(all_grads)
    self.encoder_optimizer.update(all_grads)
    self.decoder_optimizer.update(all_grads)
    self.dynamics_optimizer.update(all_grads)
    self.predictor_optimizer.update(all_grads)
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