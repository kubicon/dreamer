import chex
import jax
from flax import nnx
from train_utils import DreamerConfig
from networks import create_optimizers


class Dreamer():
  def __init__(self, config: DreamerConfig, game):
    self.config = config
    self.game = game
    self.init()
    
    
  def init(self):
    self.jax_rngs = jax.random.key(self.config.rng_seed)
    self.nnx_rngs = nnx.Rngs(self.generate_key())
    
    
    self.optimizers = create_optimizers(self.config, self.game, self.nnx_rngs)
    self.learner_steps = 0
    
     
  
  def generate_key(self):
    self.jax_rngs, key = jax.random.split(self.jax_rngs)
    return key

  def generate_keys(self, num_keys):
    self.jax_rngs, keys = jax.random.split(self.jax_rngs, num_keys + 1)
    return keys
  
  
  
  
  
  
  