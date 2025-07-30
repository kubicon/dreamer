
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

  
  
def initialize_dreamer_optimizers(config: DreamerConfig, game: JaxGame, rngs: nnx.Rngs) -> DreamerOptimizers:
  """Initializes the model networks and optimizers. For now actor and critic networks are not used""" 

  sequence_optimizer = nnx.Optimizer(
    model= SequenceModel(
        encoded_classes=config.encoded_classes,
        encoded_categories=config.encoded_categories,
        action_features=game.num_distinct_actions(),
        hidden_state_size=config.hidden_state_size,
        rngs=rngs
    ),
    tx=optax.adam(learning_rate=config.learning_rate),
  )
  

  encoder_optimizer = nnx.Optimizer(
    model= Encoder(
        hidden_state_size=config.hidden_state_size,
        observation_features=game.observation_tensor_shape(),
        encoded_classes=config.encoded_classes,
        encoded_categories=config.encoded_categories,
        hidden_features=config.encoder_network_details[0],
        num_layers=config.encoder_network_details[1],
        rngs=rngs
    ),
    tx=optax.adam(learning_rate=config.learning_rate),
  )

  decoder_optimizer = nnx.Optimizer(
    model= Decoder(
        hidden_state_size=config.hidden_state_size,
        observation_features=game.observation_tensor_shape(),
        encoded_classes=config.encoded_classes,
        encoded_categories=config.encoded_categories,
        hidden_features=config.decoder_network_details[0],
        num_layers=config.decoder_network_details[1],
        rngs=rngs
    ),
    tx=optax.adam(learning_rate=config.learning_rate),
  )

  dynamics_optimizer = nnx.Optimizer(
    model= DynamicsPredictor(
        hidden_state_size=config.hidden_state_size,
        encoded_classes=config.encoded_classes,
        encoded_categories=config.encoded_categories,
        hidden_features=config.dynamics_network_details[0],
        num_layers=config.dynamics_network_details[1],
        rngs=rngs
    ),
    tx=optax.adam(learning_rate=config.learning_rate),
  )

  predictor_optimizer = nnx.Optimizer(
    model= Predictor(
        hidden_state_size=config.hidden_state_size,
        encoded_classes=config.encoded_classes,
        encoded_categories=config.encoded_categories,
        bin_range= config.bin_range,
        hidden_features=config.predictor_network_details[0],
        num_layers=config.predictor_network_details[1],
        rngs=rngs
      ),
    tx=optax.adam(learning_rate=config.learning_rate),
  )
  
  optims = DreamerOptimizers(
    sequence_optimizer=sequence_optimizer,
    encoder_optimizer=encoder_optimizer,
    decoder_optimizer=decoder_optimizer,
    dynamics_optimizer=dynamics_optimizer,
    predictor_optimizer=predictor_optimizer
  )

  return optims