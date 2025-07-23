import chex

@chex.dataclass(frozen=True)
class DreamerConfig():
  batch_size: int
  seed: int 
  
  hidden_state_size: int
  encoded_classes: int
  encoded_categories: int
  
  encoder_network_details: tuple[int, int] = (256, 1)
  decoder_network_details: tuple[int, int] = (256, 1)
  dynamics_network_details: tuple[int, int] = (256, 1)
  predictor_network_details: tuple[int, int] = (256, 1)
  
  learning_rate: float
  rng_seed: int
  
 