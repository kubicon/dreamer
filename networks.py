
import chex
from flax import nnx
import jax.numpy as jnp



from train_utils import legal_policy, legal_log_policy


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


class RNaDNetwork(nnx.Module):
  """The RNaD algorithm network, with policy and value heads.
  Receive current iset and legal actions mask and return
  policy, value, log policy and policy logit.
  Only the iset is sent to the network.
  The value is parametrized as logits for a categorical distribution
  over the exponentially spaced bins like symexp([-bin_range, bin_range])"""

  def __init__(self, iset_features, action_features, bin_range, hidden_features, num_layers, rngs:nnx.Rngs):
    self.init_layer = LinNormRelu(iset_features, hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    self.policy_head = nnx.Linear(hidden_features, action_features, rngs=rngs)
    #Initialize the value output layer to all zeros, as per
    # https://arxiv.org/pdf/2301.04104 page 6
    self.value_head = nnx.Linear(hidden_features, (2 * bin_range) + 1, rngs=rngs, kernel_init=nnx.initializers.zeros_init(), bias_init=nnx.initializers.zeros_init())
  
  def __call__(self, iset, legal):
    x = self.init_layer(iset)
    x = self.core_mlp(x)
    logit = self.policy_head(x)
    v_dist_logits = self.value_head(x)
    
    pi = legal_policy(logit, legal)
    log_pi = legal_log_policy(logit, legal)
    
    return pi, v_dist_logits, log_pi, logit
  
class ActorNetwork(nnx.Module):
  """Actor network used for Reinforce in standard Dreamer.
  The input to the network is infoset for IIGs, or
  [sampled_model_state, recurrent_state] for PIGs or POMPDPs.
  Thus, input features are either the dimension of infoset
  or (num_classes * num_categoricals) + recurrent_state_size"""
  def __init__(self, input_features, action_features, hidden_features, num_layers, rngs:nnx.Rngs):
    self.init_layer = LinNormRelu(input_features, hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    self.policy_head = nnx.Linear(hidden_features, action_features, rngs=rngs)
  
  def __call__(self, input, legal):
    x = self.init_layer(input)
    x = self.core_mlp(x)
    logit = self.policy_head(x)
    
    pi = legal_policy(logit, legal)
    log_pi = legal_log_policy(logit, legal)
    
    return pi, log_pi, logit
  
class CriticNetwork(nnx.Module):
  """Critic network used in standard Dreamer.
  The input to the network is infoset for IIGs, or
  [sampled_model_state, recurrent_state] for PIGs or POMPDPs.
  Thus, input features are either the dimension of infoset
  or (num_classes * num_categoricals) + recurrent_state_size
  Return
  the logits of categorical distribution of value of the 
  current infoset/state, that is defined over the 
  exponentially spaced bins like symexp([-bin_range, bin_range])."""
  def __init__(self, input_features, bin_range, hidden_features, num_layers, rngs:nnx.Rngs):
    self.init_layer = LinNormRelu(input_features, hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    #Initialize the value output layer to all zeros, as per
    # https://arxiv.org/pdf/2301.04104 page 6
    self.value_head = nnx.Linear(hidden_features, (2 * bin_range + 1), rngs=rngs, kernel_init=nnx.initializers.zeros_init(), bias_init=nnx.initializers.zeros_init())
  
  def __call__(self, input):
    x = self.init_layer(input)
    x = self.core_mlp(x)
    v_dist_logit = self.value_head(x)
    
    return v_dist_logit

class SequenceModel(nnx.Module):
  '''
    Used to produce the next state of the game from the hidden state and the joint action.
    It is implemented in the same way as in the DreamerV3 reference implementation,
    except that instead of using their custom BlockLinear uses just standard MLP.
  '''
  def __init__(self, encoded_classes, encoded_categories, action_features, num_players,
               linear_hidden_features: int, linear_hidden_layers: int,
               recurrent_state_size, rngs: nnx.Rngs):
    self.multi_player = int(num_players > 1)
    self.hidden_init = LinNormRelu(recurrent_state_size, linear_hidden_features, rngs=rngs)
    self.action_init = LinNormRelu((action_features * num_players), linear_hidden_features, rngs=rngs)
    self.deter_init = LinNormRelu(encoded_classes * encoded_categories, linear_hidden_features, rngs=rngs)
    # Concatenation of the actual recurrent state, with
    # the embeddings of the recurrent state, action and stochastic state
    core_input_size = recurrent_state_size + 3 * linear_hidden_features
    self.core_mlp = HiddenMLP(core_input_size, num_layers= linear_hidden_layers, rngs=rngs)
    #This is the projection to the reset, cand and update gates
    self.gate_head = nnx.Linear(core_input_size, 3 * recurrent_state_size, rngs=rngs)
    
  def __call__(self, recurrent_state: chex.Array, deter_state: chex.Array, action:chex.Array):
    """Ensure that action is already one hot encoded. Deter state is the 
    already sampled state out of stochastic state"""
    flat_deter = jnp.reshape(deter_state, (*deter_state.shape[:-2], -1))
    #If we have multi-player, we need to flatten the last two dimensions
    stop_at = -1 -self.multi_player
    flat_action = jnp.reshape(action, (*action.shape[:stop_at], -1))
    x0 = self.hidden_init(recurrent_state)
    x1 = self.deter_init(flat_deter)
    x2 = self.action_init(flat_action)
    x = jnp.concatenate([recurrent_state, x0, x1, x2], axis=-1)
    x = self.core_mlp(x)
    gates = self.gate_head(x)
    reset, cand, update = jnp.split(gates, 3, axis=-1)
    reset = nnx.sigmoid(reset)
    cand = nnx.tanh(reset * cand)
    #The -1 makes the update naturally smaller
    # making the network more biased towards
    # keeping the old recurrent_state
    update = nnx.sigmoid(update - 1)
    new_recurrent_state = update * cand + (1 - update) * recurrent_state
    
    return new_recurrent_state
  

class Encoder(nnx.Module):
  """Recieve an observation from the environment,
  return a latent feature vector that, along with the current
  recurrent state, will be used to produce current deterministic state"""
  def __init__(self, observation_features, tokens_features, hidden_features, num_layers, rngs: nnx.Rngs) -> None:
    self.init_layer = LinNormRelu(observation_features, hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    self.last_layer = nnx.Linear(hidden_features, tokens_features, rngs=rngs)
    
  def __call__(self, observation: chex.Array):
    x = self.init_layer(observation)
    x = self.core_mlp(x)
    tokens = self.last_layer(x)
    return tokens

class JointIsetEncoder(nnx.Module):
  """Recieve a joint infoset from the environment,
  return a latent feature vector that, along with the current
  recurrent state, will be used to produce current deterministic state."""
  def __init__(self, iset_features, num_players, tokens_features, hidden_features, num_layers, rngs: nnx.Rngs) -> None:

    self.init_layer = LinNormRelu(iset_features * num_players, hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    self.last_layer = nnx.Linear(hidden_features, tokens_features, rngs=rngs)
    
  def __call__(self, joint_iset: chex.Array):
    x = jnp.reshape(joint_iset, (*joint_iset.shape[:-2], -1))
    x = self.init_layer(x)
    x = self.core_mlp(x)
    tokens = self.last_layer(x)
    return tokens
  
class ObservedPredictor(nnx.Module):
  """Receive a current recurrent state and observation
  latent tokens produced by some encoder and return current stochastic
  state logits."""

  def __init__(self, recurrent_state_size, token_features, encoded_classes, encoded_categories, hidden_features, num_layers, rngs:nnx.Rngs) ->None:
    self.encoded_classes = encoded_classes
    self.encoded_categories = encoded_categories
    self.init_layer = LinNormRelu(recurrent_state_size + token_features, hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    self.last_layer = nnx.Linear(hidden_features, (encoded_classes * encoded_categories), rngs=rngs)

  def __call__(self, recurrent_state: chex.Array, obs_tokens: chex.Array):
    x = jnp.concatenate([recurrent_state, obs_tokens], axis=-1)
    x = self.init_layer(x)
    x = self.core_mlp(x)
    stoch_logits = self.last_layer(x)
    stoch_logits = jnp.reshape(stoch_logits, (*stoch_logits.shape[:-1], self.encoded_classes, self.encoded_categories))
    return stoch_logits

class Decoder(nnx.Module):
  """Receive a current deterministic latent state (eg. a encoded_categories-hot vector)
  and recurrent state
  and return a reconstruction of current observation."""
  def __init__(self, recurrent_state_size, observation_features, encoded_classes, encoded_categories, hidden_features, num_layers, rngs: nnx.Rngs) -> None:
    self.encoded_classes = encoded_classes
    self.encoded_categories = encoded_categories
    self.init_layer = LinNormRelu(recurrent_state_size + encoded_classes * encoded_categories, hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    self.last_layer = nnx.Linear(hidden_features, observation_features, rngs=rngs)
    
  def __call__(self, recurrent_state: chex.Array, encoded_state: chex.Array):
    x = jnp.concatenate([recurrent_state, encoded_state.reshape(*encoded_state.shape[:-2], -1)], axis=-1)
    x = self.init_layer(x)
    x = self.core_mlp(x)
    obs = self.last_layer(x)
    return obs
  
class DynamicsPredictor(nnx.Module):
  """Recieve a current hidden state and return the current stochastic state logits.
  Acts as a prior to the encoders posterior."""
  def __init__(self, recurrent_state_size, encoded_classes, encoded_categories, hidden_features, num_layers, rngs: nnx.Rngs) -> None:
    self.encoded_classes = encoded_classes
    self.encoded_categories = encoded_categories
    self.init_layer = LinNormRelu(recurrent_state_size, hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    self.last_layer = nnx.Linear(hidden_features, encoded_classes * encoded_categories, rngs=rngs)
    
  def __call__(self, recurrent_state: chex.Array):
    x = self.init_layer(recurrent_state)
    x = self.core_mlp(x)
    x = self.last_layer(x)
    encoded_state = x.reshape(*x.shape[:-1], self.encoded_classes, self.encoded_categories)
    return encoded_state
  
  
class Predictor(nnx.Module):
  """Receive a current deterministic latent state (eg. a encoded_categories-hot vector)
  and return logits of the predicted reward bin_distribution and done flag, ordered as such.
  Pass the done logits through sigmoid and compare against a threshold if you want
  to obtain an actual done flag. The reward are logits of a distribution over the exponentially
  spaced bins like symexp([-bin_range, bin_range])."""
  def __init__(self, recurrent_state_size, encoded_classes, encoded_categories, bin_range, hidden_features, num_layers, rngs: nnx.Rngs) -> None:
    self.init_layer = LinNormRelu(recurrent_state_size + encoded_classes * encoded_categories, hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    #Initialize the reward output layer to all zeros, as per
    # https://arxiv.org/pdf/2301.04104 page 6
    self.reward_layer = nnx.Linear(hidden_features, (2 * bin_range) + 1, rngs=rngs, kernel_init= nnx.initializers.zeros_init(), bias_init=nnx.initializers.zeros_init())
    self.done_layer = nnx.Linear(hidden_features, 1, rngs=rngs)
    
  def __call__(self, recurrent_state: chex.Array, encoded_state: chex.Array):
    x = jnp.concatenate([recurrent_state, encoded_state.reshape(*encoded_state.shape[:-2], -1)], axis=-1)
    x = self.init_layer(x)
    x = self.core_mlp(x)
    reward = self.reward_layer(x)
    done = self.done_layer(x)
    return reward, done

class LegalActionsNetwork(nnx.Module):
  """Receive a current hidden state and deterministic state and return the legal action logits."""
  def __init__(self, num_players, action_dimension, encoded_classes, encoded_categories, recurrent_state_size, hidden_features, num_layers, rngs: nnx.Rngs) -> None:
    self.init_layer = LinNormRelu(recurrent_state_size + (encoded_classes * encoded_categories), hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    self.legal_layer = nnx.Linear(hidden_features, num_players * action_dimension, rngs=rngs)

    self.num_players = num_players
    self.action_dimension = action_dimension

  def __call__(self, recurrent_state: chex.Array, encoded_state: chex.Array):
    x = jnp.concatenate([recurrent_state, encoded_state.reshape(*encoded_state.shape[:-2], -1)], axis=-1)
    x = self.init_layer(x)
    x = self.core_mlp(x)
    legal = self.legal_layer(x)
    return jnp.reshape(legal, (*legal.shape[:-1], self.num_players, self.action_dimension))

class RewardPredictor(nnx.Module):
  """Receive a current hidden state and deterministic state and 
  return the logits of a the reward categorical
  distribution over exponentially spaced bins such as
  symexp([-bin_range, bin_range])"""
  def __init__(self, bin_range, encoded_classes, encoded_categories, recurrent_state_size, hidden_features, num_layers, rngs: nnx.Rngs) -> None:
    self.init_layer = LinNormRelu(recurrent_state_size + (encoded_categories * encoded_classes), hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    self.reward_layer = nnx.Linear(hidden_features, (2 * bin_range) + 1, rngs=rngs,kernel_init= nnx.initializers.zeros_init(), bias_init=nnx.initializers.zeros_init())


  def __call__(self, recurrent_state: chex.Array, encoded_state: chex.Array):
    x = jnp.concatenate([recurrent_state, encoded_state.reshape(*encoded_state.shape[:-2], -1)], axis=-1)
    x = self.init_layer(x)
    x = self.core_mlp(x)
    reward_dist_logits = self.reward_layer(x)
    return reward_dist_logits
  
class DonePredictor(nnx.Module):
  """Receive a current hidden state and deterministic state and
  return the logits of the done/terminal flag"""
  def __init__(self,encoded_classes, encoded_categories, recurrent_state_size, hidden_features, num_layers, rngs: nnx.Rngs) -> None:
    self.init_layer = LinNormRelu(recurrent_state_size + (encoded_classes * encoded_categories), hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    self.done_layer = nnx.Linear(hidden_features, 1, rngs=rngs)


  def __call__(self, recurrent_state: chex.Array, encoded_state: chex.Array):
    x = jnp.concatenate([recurrent_state, encoded_state.reshape(*encoded_state.shape[:-2], -1)], axis=-1)
    x = self.init_layer(x)
    x = self.core_mlp(x)
    done_logit = self.done_layer(x)
    return done_logit
  

# class PredictorWithLegal(nnx.Module):
#   """Has reward and done heads the same way as standard predictor,
#   but also predicts legal action mask for both players."""
#   def __init__(self, num_players, action_dimension, recurrent_state_size, encoded_classes, encoded_categories, bin_range, hidden_features, num_layers, rngs: nnx.Rngs) -> None:
#     self.init_layer = LinNormRelu(recurrent_state_size + encoded_classes * encoded_categories, hidden_features, rngs)
#     self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
#     self.reward_layer = nnx.Linear(hidden_features, (2 * bin_range) + 1, rngs=rngs)
#     #self.reward_layer = nnx.Linear(hidden_features, 1, rngs=rngs)
#     self.done_layer = nnx.Linear(hidden_features, 1, rngs=rngs)
#     self.legal_layer = nnx.Linear(hidden_features, num_players * action_dimension, rngs=rngs)

#     self.num_players = num_players
#     self.action_dimension = action_dimension
    
#   def __call__(self, recurrent_state: chex.Array, encoded_state: chex.Array):
#     x = jnp.concatenate([recurrent_state, encoded_state.reshape(*encoded_state.shape[:-2], -1)], axis=-1)
#     x = self.init_layer(x)
#     x = self.core_mlp(x)
#     reward = self.reward_layer(x)
#     done = self.done_layer(x)
#     flat_legal = self.legal_layer(x)
#     legal = jnp.reshape(flat_legal, (*flat_legal.shape[:-1], self.num_players, self.action_dimension))
#     return reward, done, legal

