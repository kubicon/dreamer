import chex
import jax
import jax.numpy as jnp
from games.jax_game import GameState 
import os
import pickle

def symlog(x: jax.Array):
  return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)

def symexp(x: jax.Array):
  return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)

def get_loss_mean_with_mask(loss: jax.Array, mask: jax.Array) -> jax.Array:
  """Mask a loss using mask and compute its mean, 
  such that elements with 0 in the mask are correctly ignored.
    Make sure loss and mask are of broadcastable dimensions. """
  masked_loss = loss * mask
  normalization_factor = jnp.sum(mask)
  summed_loss = jnp.sum(masked_loss)
  return summed_loss / (normalization_factor + (normalization_factor == 0))



@chex.dataclass(frozen=True)
class PredictionStep():
  repr_state: chex.Array
  decoded_obs: chex.Array
  reward_dist_logit: chex.Array
  done_logit: chex.Array
  dynamics_state: chex.Array

@chex.dataclass(frozen=True)
class TimeStep():
  
  obs: chex.Array = () # [..., Player, iset_dim] for multi agent or [..., obs_dim] for single_agent
  legal: chex.Array = () # [..., Player, A] for multi agent or [..., A] for single_agent
  
  action: chex.Array = () # [..., Player, A] for multi agent or [..., A] for single agent
  policy: chex.Array = () # [..., Player, A] for multi agent or [..., A] for single agent
  
  reward: chex.Array = () # [...] Reward after playing an action
  valid: chex.Array = () # [...] Flag determining, whether we should train in this state
  terminal: chex.Array = () #[...] Whether state after playing an action was terminal




@chex.dataclass(frozen=True)
class DreamerConfig():
  batch_size: int
  seed: int

  hidden_state_size: int #Size of the RNN hidden state
  encoded_classes: int # Number of classes for each categorical distribution in state
  encoded_categories: int # Number of categorical distributions in state

  learning_rate: float
  rng_seed: int



  #Weights of the individual loss terms of the world model
  beta_prediction: float = 1
  beta_dynamics: float = 1
  beta_representation: float = 0.1

  free_bits_clip_threshold: float = 1 #Threshold for loss clip in free bits. 
  
  bin_range: int = 20 #Number of the exponentially spaced bins for certain predictions such as reward in one direction, bins will be spaced out as symexp([-bin_range, ..., bin_range])
  
  # Ordered as (hidden_layer_features, num_hidden_layers)
  encoder_network_details: tuple[int, int] = (256, 1)
  decoder_network_details: tuple[int, int] = (256, 1)
  dynamics_network_details: tuple[int, int] = (256, 1)
  predictor_network_details: tuple[int, int] = (256, 1)
  
def get_reference_policy(game_state: GameState, legal_actions: jax.Array):
  """Returns the reference sampling policy. For now returns just a uniform policy.
  TODO: This is just for the basic testing, change this function"""
  return legal_actions / legal_actions.sum(axis=-1, keepdims=True)


def save_model(model, path): 
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "wb") as f:
    pickle.dump(model, f)
    
def load_model(path):
  with open(path, "rb") as f:
    return pickle.load(f)
  