
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

  
class ActorNetwork(nnx.Module):
  """Actor network used for Reinforce in standard Dreamer.
  The input to the network is latent or real infoset."""
  def __init__(self, input_features, action_features, hidden_features, num_layers, rngs:nnx.Rngs):
    self.init_layer = LinNormRelu(input_features, hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    #Initialize to uniform policy logits
    #self.policy_head = nnx.Linear(hidden_features, action_features, rngs=rngs, kernel_init=nnx.initializers.zeros_init(), bias_init=nnx.initializers.zeros_init())
    self.policy_head = nnx.Linear(hidden_features, action_features, rngs=rngs)
    
  def __call__(self, input, legal):
    x = self.init_layer(input)
    x = self.core_mlp(x)
    logit = self.policy_head(x)
    
    pi = legal_policy(logit, legal)
    log_pi = legal_log_policy(logit, legal)
    
    return pi, log_pi, logit
  
class CriticNetwork(nnx.Module):
  """Critic network of the over the history value function.
  The input to the network is joint_latent_infoset.
  Thus, input features are  2 * latent_infoset_size.
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
    Centralized: accepts a joint action of shape (..., num_players, action_features).
  '''
  def __init__(self, encoded_classes, encoded_categories, num_players, action_features,
               hidden_features: int, linear_hidden_layers: int,
               recurrent_state_size, rngs: nnx.Rngs):
    self.hidden_init = LinNormRelu(recurrent_state_size, hidden_features, rngs=rngs)
    self.action_init = LinNormRelu(num_players * action_features, hidden_features, rngs=rngs)
    self.deter_init = LinNormRelu(encoded_classes * encoded_categories, hidden_features, rngs=rngs)
    # Concatenation of the actual recurrent state, with
    # the embeddings of the recurrent state, action and stochastic state
    core_input_size = recurrent_state_size + 3 * hidden_features
    self.core_mlp = HiddenMLP(core_input_size, num_layers= linear_hidden_layers, rngs=rngs)
    #This is the projection to the reset, cand and update gates
    self.gate_head = nnx.Linear(core_input_size, 3 * recurrent_state_size, rngs=rngs)

  def __call__(self, recurrent_state: chex.Array, deter_state: chex.Array, action: chex.Array):
    """Ensure that actions are already one hot encoded.
    Deter state is the already sampled state out of stochastic state.
    Action has shape (..., num_players, action_features)."""
    flat_deter = jnp.reshape(deter_state, (*deter_state.shape[:-2], -1))
    flat_action = jnp.reshape(action, (*action.shape[:-2], -1))
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


class InfosetModel(nnx.Module):
  '''
    Maintains a recurrent latent infoset for a single player.
    Receives the current latent_infoset, an observation and an action for that player,
    and outputs an updated latent_infoset of fixed size infoset_dim.
    Structure is identical to SequenceModel.
  '''
  def __init__(self, observation_features, action_features, infoset_dim: int,
               hidden_features: int, hidden_layers: int, rngs: nnx.Rngs):
    self.latent_init = LinNormRelu(infoset_dim, hidden_features, rngs=rngs)
    self.observation_init = LinNormRelu(observation_features, hidden_features, rngs=rngs)
    self.action_init = LinNormRelu(action_features, hidden_features, rngs=rngs)
    # Concatenation of the actual latent_infoset with the three embeddings
    core_input_size = infoset_dim + 3 * hidden_features
    self.core_mlp = HiddenMLP(core_input_size, num_layers=hidden_layers, rngs=rngs)
    # Projection to reset, cand and update gates
    self.gate_head = nnx.Linear(core_input_size, 3 * infoset_dim, rngs=rngs)

  def __call__(self, latent_infoset: chex.Array, observation: chex.Array, action: chex.Array):
    """Ensure action is already one hot encoded."""
    x0 = self.latent_init(latent_infoset)
    x1 = self.observation_init(observation)
    x2 = self.action_init(action)
    x = jnp.concatenate([latent_infoset, x0, x1, x2], axis=-1)
    x = self.core_mlp(x)
    gates = self.gate_head(x)
    reset, cand, update = jnp.split(gates, 3, axis=-1)
    reset = nnx.sigmoid(reset)
    cand = nnx.tanh(reset * cand)
    # The -1 makes the update naturally smaller,
    # making the network more biased towards keeping the old latent_infoset
    update = nnx.sigmoid(update - 1)
    new_latent_infoset = update * cand + (1 - update) * latent_infoset

    return new_latent_infoset


class InfosetDecoder(nnx.Module):
  """Receive the latent infoset produced by InfosetModel and return
  reconstructions of the real observation and real previous action for that player."""
  def __init__(self, observation_features, action_features, infoset_dim,
               hidden_features, num_layers, rngs: nnx.Rngs) -> None:
    self.init_layer = LinNormRelu(infoset_dim, hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    self.observation_head = nnx.Linear(hidden_features, observation_features, rngs=rngs)
    self.action_head = nnx.Linear(hidden_features, action_features, rngs=rngs)

  def __call__(self, latent_infoset: chex.Array):
    x = self.init_layer(latent_infoset)
    x = self.core_mlp(x)
    obs = self.observation_head(x)
    action = self.action_head(x)
    return obs, action
  
class InfosetPredictor(nnx.Module):
  """Receive the joint latent infoset among all players 
  and return the reconstruction of the latent model state.
  Uses the assumption of no-hidden chance outcomes, to assume
  that union of information contained in infosets is enough to obtain
  a perfect information state. Used to force the latent infosets 
  to contain sufficient context instead of just predicting observations
  """
  def __init__(self, num_players, infoset_dim, recurrent_state_size, num_classes, num_categories,
               hidden_features, num_layers, rngs: nnx.Rngs):
    self.num_classes = num_classes
    self.num_categories = num_categories
    self.joint_infoset_init = LinNormRelu(num_players * infoset_dim, hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    self.rec_state_head = LinNormRelu(hidden_features, recurrent_state_size, rngs)
    self.deter_state_head = LinNormRelu(hidden_features, num_classes * num_categories, rngs)

  def __call__(self, joint_latent_infoset: chex.Array):
    #Flatten the joint latent infoset
    flat_latent_infoset = jnp.reshape(joint_latent_infoset, (*joint_latent_infoset.shape[:-2], -1))
    x = self.joint_infoset_init(flat_latent_infoset)
    x = self.core_mlp(x)
    recurrent = self.rec_state_head(x)
    flat_deter = self.deter_state_head(x)
    #Reshape into the usual Classes x Categories shape
    deter = jnp.reshape(flat_deter, (*flat_deter.shape[:-1], self.num_classes, self.num_categories))
    return recurrent, deter
    


class Encoder(nnx.Module):
  """Receive joint observations from all players,
  return a latent feature vector that, along with the current
  recurrent state, will be used to produce current stochastic state logits.
  Centralized: accepts joint observations of shape (..., num_players, observation_features)."""
  def __init__(self, num_players, observation_features, tokens_features, hidden_features, num_layers, rngs: nnx.Rngs) -> None:

    self.tokens_features = tokens_features
    self.init_layer = LinNormRelu(num_players * observation_features, hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    self.last_layer = nnx.Linear(hidden_features, tokens_features, rngs=rngs)

  def __call__(self, observations: chex.Array):
    flat_obs = jnp.reshape(observations, (*observations.shape[:-2], -1))
    x = self.init_layer(flat_obs)
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
  and recurrent state, and return reconstructions of the observations for all players
  as a tensor of shape (..., num_players, observation_features)."""
  def __init__(self, num_players, recurrent_state_size, observation_features, encoded_classes, encoded_categories, hidden_features, num_layers, rngs: nnx.Rngs) -> None:
    self.num_players = num_players
    self.observation_features = observation_features
    self.init_layer = LinNormRelu(recurrent_state_size + encoded_classes * encoded_categories, hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    self.last_layer = nnx.Linear(hidden_features, num_players * observation_features, rngs=rngs)

  def __call__(self, recurrent_state: chex.Array, encoded_state: chex.Array):
    x = jnp.concatenate([recurrent_state, encoded_state.reshape(*encoded_state.shape[:-2], -1)], axis=-1)
    x = self.init_layer(x)
    x = self.core_mlp(x)
    obs = self.last_layer(x)
    return jnp.reshape(obs, (*obs.shape[:-1], self.num_players, self.observation_features))
  
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
    self.init_layer = LinNormRelu((recurrent_state_size + encoded_classes * encoded_categories), hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    #Initialize the reward output layer to all zeros, as per
    # https://arxiv.org/pdf/2301.04104 page 6
    self.reward_layer = nnx.Linear(hidden_features, (2 * bin_range) + 1, rngs=rngs, kernel_init= nnx.initializers.zeros_init(), bias_init=nnx.initializers.zeros_init())
    self.done_layer = nnx.Linear(hidden_features, 1, rngs=rngs)
    
  def __call__(self, recurrent_state: chex.Array, encoded_state: chex.Array):
    #Flatten the dimensions
    # Flatten [K, C]
    flat_encoded_state = encoded_state.reshape(*encoded_state.shape[:-2], -1)                        
    x = jnp.concatenate([recurrent_state, flat_encoded_state], axis=-1)
    x = self.init_layer(x)
    x = self.core_mlp(x)
    reward = self.reward_layer(x)
    done = self.done_layer(x)
    return reward, done

class LegalActionsNetwork(nnx.Module):
  """Receive a current hidden state and deterministic state and return the legal action logits.
  CRUCIAL! Centralized legals are only sound where knowledge of legal actions does not reveal any information. 
  In our domains it holds, but in Fog of War games, 
  where we can try to move into an unobservable fog, knowing that the action is illegal would reveal information. 
  Just something to keep in mind."""
  def __init__(self, num_players, action_dimension, encoded_classes, encoded_categories, recurrent_state_size, hidden_features, num_layers, rngs: nnx.Rngs) -> None:
    self.init_layer = LinNormRelu((recurrent_state_size + (encoded_classes * encoded_categories)), hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    self.legal_layer = nnx.Linear(hidden_features, num_players * action_dimension, rngs=rngs)

    self.num_players = num_players
    self.action_dimension = action_dimension

  def __call__(self, recurrent_state: chex.Array, encoded_state: chex.Array):
    #Flatten the dimensions
    # Flatten [K, C]
    flat_encoded_state = encoded_state.reshape(*encoded_state.shape[:-2], -1)                        
    x = jnp.concatenate([recurrent_state, flat_encoded_state], axis=-1)
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
    self.init_layer = LinNormRelu((recurrent_state_size + (encoded_categories * encoded_classes)), hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    self.reward_layer = nnx.Linear(hidden_features, (2 * bin_range) + 1, rngs=rngs,kernel_init= nnx.initializers.zeros_init(), bias_init=nnx.initializers.zeros_init())


  def __call__(self, recurrent_state: chex.Array, encoded_state: chex.Array):
    #Flatten the dimensions
    # Flatten [K, C]
    flat_encoded_state = encoded_state.reshape(*encoded_state.shape[:-2], -1)                        
    x = jnp.concatenate([recurrent_state, flat_encoded_state], axis=-1)
    x = self.init_layer(x)
    x = self.core_mlp(x)
    reward_dist_logits = self.reward_layer(x)
    return reward_dist_logits
  
class DonePredictor(nnx.Module):
  """Receive a current hidden state and deterministic state and
  return the logits of the done/terminal flag"""
  def __init__(self, encoded_classes, encoded_categories, recurrent_state_size, hidden_features, num_layers, rngs: nnx.Rngs) -> None:
    self.init_layer = LinNormRelu((recurrent_state_size + (encoded_classes * encoded_categories)), hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    self.done_layer = nnx.Linear(hidden_features, 1, rngs=rngs)


  def __call__(self, recurrent_state: chex.Array, encoded_state: chex.Array):
    #Flatten the dimensions
    # Flatten [K, C]
    flat_encoded_state = encoded_state.reshape(*encoded_state.shape[:-2], -1)                        
    x = jnp.concatenate([recurrent_state, flat_encoded_state], axis=-1)
    x = self.init_layer(x)
    x = self.core_mlp(x)
    done_logit = self.done_layer(x)
    return done_logit

