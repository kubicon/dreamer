"""Define the distribution utilities used by the various models.
Heavily inspired by https://github.com/symoon11/dreamerv3-flax"""

import jax
import jax.numpy as jnp
from train_utils import symlog


def sample_categorical(logits: jax.Array, key, uniform_mix: float = 0.01)-> jax.Array:
  """Given a PRNG key produced by split, sample from each
  of the categorical distributions logits and return the
  one-hot encoded outcome for each of the distributions.
  This function does NOT split internally,
  make sure the key passed to it is not reused.
  Uniform_mix creates a mixture between the actual logits induced distribution
  and uniform distribution, to prevent KL losses spike early
  as described in https://arxiv.org/pdf/2301.04104 page 5. """
  # Calculate the logits-induced probability.
  probs = jax.nn.softmax(logits, axis=-1)
  uniform = jnp.ones_like(probs) / probs.shape[-1]
  # Mix the probability with the uniform distribution.
  probs = (1.0 - uniform_mix) * probs + uniform_mix * uniform
  # Recalculate the logits
  logits_with_uniform = jnp.log(probs)
  logits = jnp.where(uniform_mix > 0, logits_with_uniform, logits)
  num_classes = logits.shape[-1]
  sampled_classes = jax.random.categorical(key, logits, axis=-1)
  oh_sampled_classes = jax.nn.one_hot(sampled_classes, num_classes, axis=-1)
  #Perform the STE
  output = jax.lax.stop_gradient(oh_sampled_classes) + (probs - jax.lax.stop_gradient(probs))
  return output

def get_normal_log_prob(mean_logits: jax.Array, value: jax.Array, use_symlog=False) ->jax.Array:
   """Get log prob of the normal distributions represented by the predictor outputs.
   Since the predictors output logits for mean and variance is assumed to be one, 
   the log prob reduces to -MSE. Can use the symlog transformation from https://arxiv.org/pdf/2301.04104
   page 7 to improve robustness. If using it, make sure to then convert the network prediction
   with the inverse symexp for inference."""
   assert mean_logits.shape == value.shape, f"Expected mean logits and value shape to match, given shapes are {mean_logits.shape} {value.shape}"
   value = jnp.where(use_symlog, symlog(value), value)
   log_prob = -(value - mean_logits) **2
   return log_prob

def get_bin_log_prob(dist_logits: jax.Array, bins: jax.Array,  value: jax.Array, use_symlog = True)->jax.Array:
   """Get log prob of the discrete distribution corresponding to the exponentially spaced bins.
   From https://arxiv.org/pdf/2301.04104  page 7. First two hot encodes value, and the 
   final log prob is twohot(value) * logsoftmax(dist_logits). 
   Expects dist_logits to be of shape [Trajectory, batch, 2*bin_range + 1],
   value to be of shape [Trajectory, batch, 1] and
   bins of shape [2 * bin_range + 1]"""
   assert dist_logits.shape[-1] == bins.shape[0], f"Expected dis_logits to match lenght of bins, instead got {dist_logits.shape[-1]} {bins.shape[0]}"
   assert value.shape[:-1] == dist_logits.shape[:-1], f"Expected value shape to match dist_logits shape everywhere except last dimension, got {value.shape[:-1]} {dist_logits.shape[:-1]}"
   #[Trajectory, Batch, 2 * bin_range + 1]
   val_two_hot = two_hot_encode(bins, value, use_symlog=use_symlog)
   return val_two_hot * jax.nn.log_softmax(dist_logits)
   

def two_hot_encode(bins: jax.Array, value:jax.Array, use_symlog= True) -> jax.Array:
   """Perform the two hot encoding of value (by default transformed by symlog)
   in the range of bins. There will be two nonzero values of the two closest bins, 
   with values proportional to the bin closeness."""
   value = jnp.where(use_symlog, symlog(value), value)
   promoted_bins = bins[None, None, ...]
   below = value >= promoted_bins
   above = value <= promoted_bins
   #Making use of argmax/argmin returning the first occurence
   int_start_idx = jnp.where(jnp.sum(below, axis=-1) == bins.shape[0] - 1, bins.shape[0] - 1, jnp.maximum(jnp.argmin(below, axis=-1).astype(jnp.int32) - 1, 0))
   int_end_idx = jnp.argmax(above, axis= -1).astype(jnp.int32)

   equal = int_start_idx == int_end_idx
   equal_oh = jax.nn.one_hot(int_start_idx, bins.shape[0], axis=-1)
   
   #[Trajectory, Batch, 1]
   start_bins = jnp.take_along_axis(promoted_bins, int_start_idx[..., None], axis=-1)
   end_bins = jnp.take_along_axis(promoted_bins, int_end_idx[..., None], axis=-1)
   start_dist = jnp.abs(value - start_bins)
   end_dist = jnp.abs(end_bins - value) 
   total = start_dist + end_dist
   weight_start = start_dist / total
   weight_end = end_dist / total

   start_oh = jax.nn.one_hot(int_start_idx, bins.shape[0], axis=-1) * weight_start
   end_oh = jax.nn.one_hot(int_end_idx, bins.shape[0], axis=-1) * weight_end
   
   #[Trajectory, Batch, 2 * bin_range + 1]
   two_hot = jnp.where(equal[..., None], equal_oh, start_oh + end_oh)

   return two_hot

def kl_divergence(orig: jax.Array, other: jax.Array) ->jax.Array:
   """Computes KL divergence. Expects both orig and other to already be softmaxed
   into probability distributions. Returning kl_divergence is summed over the last two dimensions
   [categoricals, classes]."""
   div_term = jnp.log(orig) - jnp.log(other)
   divergence_unmasked = jnp.sum(orig * div_term, axis=(-1, -2))
   return divergence_unmasked
   


   
   
