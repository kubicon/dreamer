import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import chex

from functools import partial

from dreamer_ma import DreamerMA
from games.jax_game import JaxGame
from experiments.eval_utils import cartesian_product, unroll_chance_node
from games.jax_game import GameState
#from ma_rssm import MARSSM

@chex.dataclass
class ModelGameState:
  game_state: GameState
  legals: chex.Array
  reward: chex.Array
  terminal: chex.Array
  turn: int
  recurrent_state: chex.Array # The recurrent network state for all players
  deter_state: chex.Array # The one-hot sampled outcome of the categoricals for all players
  stoch_state: chex.Array # The softmaxed distribution over the latent deter states for all players
                          # with the invalid outcomes already filtered out.
  prev_action:chex.Array #Previous action needed for the latent infoset
  joint_latent_infoset: chex.Array # The latent infoset

u8 = jnp.uint8
f32 = jnp.float32

def zero_pad_x(x, to_pad:int):
  padding_tup = ((0, to_pad, ), *[(0, 0) for _ in range(x.ndim - 1)])
  padded_x = jnp.pad(x, padding_tup, mode='constant', constant_values=0)
  return padded_x

class DreamerModelGame(JaxGame):
  def __init__(self, model: DreamerMA, probability_threshold = 0.05):
    """A game which has structure corresponding to
    a JAX game, but it is instead built from the learned Dreamer
    world model. It operates like this: It walks the real game
    and at the same time updates the context of the Dreamer model
    with the actions and observations (infosets in this case) collected
    from the environment.
    To traverse the whole model state space we differ between two nodes.
    1. Decision nodes: These get original environment action and use it to
    update both the model context and the state of the underlying game.
    2. Chance nodes: Crucially every decision node will be followed by a chance
    node. These pick samples from the stochastic component of the model. Additionally
    to fulfill the constraint of only one consecutive chance node, they also jointly
    unroll original game chance nodes where necessary. After a chance node, we will
    have access to both the actual deterministic model context and the underlying
    game state corresponding to it.


    IMPORTANT! Due to padding the chance outcomes and unrolling simultaneously
    the original environment ones and the latent ones, the
    amount of chance outcomes is max_orig_game_chance_outcomes * (categories ** (classes * num_players)), this will only reasonably work
    for small latent stochastic states and low stochasticity."""
    self.game = model.game
    self.recurrent_state_size = model.recurrent_state_size
    self.ma_rssm = model.optimizer.model
    self.num_classes = model.wm_config.encoded_classes
    self.num_categories = model.wm_config.encoded_categories

    self.deter_state_size = self.num_classes * self.num_categories
    self.latent_infoset_size = model.latent_infoset_size
    self.players = model.game.num_players()

    self.actions = model.game.num_distinct_actions()

    self.max_chance_outcomes = self.game.max_chance_outcomes() * self.num_categories ** (self.num_classes)
    #Every step will have a corresponding chance node
    self.trajectory_max = 2 * model.trajectory_max

    self.probabilty_threshold = probability_threshold

    
    class_indices = jnp.tile(jnp.arange(self.num_categories), (self.num_classes, 1))
    #Precomputing the indices of the individual outcomes. These will be the same
    # only the probabilities will differ. 
    # shape [self.num_categories ** (self.num_classes), self.num_classes]
    self.outcome_indices = cartesian_product(*class_indices)
    self.init_recurrent = self.ma_rssm.get_init_recurrent()
    self.use_real_infoset = model.use_real_infoset

    self.cache_model_calls()

  
  @partial(jax.jit, static_argnums=0)
  def get_real_game_obs(self, game_state: GameState):
    _, p1_infoset, p2_infoset, _ = self.game.get_info(game_state)
    return jnp.stack([p1_infoset, p2_infoset], axis=0)


  def get_chance_outcomes(self, game_state: ModelGameState):
    """This method unrolls jointly the original game chance node
    and the samples from the model stochastic state."""
    
    def unroll_game_chance():
      next_chance_outcomes, next_chance_probs = self.game.get_outcomes_and_probs(game_state.game_state)
      #Do not vmap over the current state, that will be the same.
      vectorized_apply_action = jax.vmap(self.game.apply_action, in_axes=(None, 0))
      next_game_states, next_terminal, next_rewards, next_legal = vectorized_apply_action(game_state.game_state, next_chance_outcomes)
      return next_game_states, next_terminal, next_rewards, next_legal.astype(u8), next_chance_probs
    def unroll_no_chance():
      states = jax.tree.map(lambda x: x[None, ...], game_state.game_state)
      legal = game_state.legals[None, ...]
      terminal = game_state.terminal[None, ...]
      rewards = game_state.reward[None, ...]
      return states, terminal, rewards, legal, jnp.ones(1)
    get_stochs = jax.vmap(self.cached_encoder, in_axes=(None, 0), out_axes=0)
    vectorized_obs = jax.vmap(self.get_real_game_obs, in_axes=0)
    states, terminals, rewards, legals, probs = jax.lax.cond(self.game.is_chance(game_state.game_state), unroll_game_chance, unroll_no_chance)
    observations = vectorized_obs(states)
    stochs = get_stochs(game_state.recurrent_state, observations)
    num_outcomes = terminals.shape[0]
    #Threshold and renormalize the stochs
    # Shape [Next observations, Classes, Categories]
    stochs = stochs * (stochs >= self.probabilty_threshold)
    normalization = jnp.sum(stochs, axis=-1, keepdims=True)
    stochs = stochs / (normalization + (normalization == 0))
    # Shape [Categories ** Classes, Classes, Categories]
    all_deters = jax.nn.one_hot(self.outcome_indices, self.num_categories, axis=-1, dtype=u8)
    #We want to get the probabilities of every deterministic
    # state conditioned on receiving the observation. We can do this
    # by broadcasting the deters along observation dimension
    # and the stoch probabilities along the outcome dimension
    #Shape [Next_observations, Categories ** Classes]
    deter_probs = jnp.prod(jnp.sum(stochs[:, None, ...] * all_deters[None, ...], axis=-1), axis=-1)
    #Also multiply by the chance outcome probabilities and flatten
    outcome_probs = (deter_probs * probs[:, None]).ravel()

    #Get the new latent infosets.
    # We need to vmap over the real game outcome dimension,
    # but only for the observation. The rest will stay constant for
    # all chance outcomes
    vectorized_latent_infosets = jax.vmap(self.cached_infoset_net, in_axes=(None, 0, None), out_axes=(0))

    new_latent_infosets = vectorized_latent_infosets(game_state.joint_latent_infoset, observations, game_state.prev_action)

    #Now create the outcomes We can repeat most of the data
    # for each deterministic state, when fixing a given observation
    outcomes = ModelGameState(game_state= jax.tree.map(lambda x: x[:, None, ...], states),
                              legals = legals[:, None, ...],
                              reward = rewards[:, None, ...],
                              terminal=terminals[:, None, ...],
                              turn = jnp.full((num_outcomes, 1), game_state.turn + 1),
                              recurrent_state = game_state.recurrent_state[None, None, ...],
                              deter_state=all_deters[None, ...],
                              stoch_state=stochs[:, None, ...],
                              prev_action=game_state.prev_action[None, None, ...],
                              joint_latent_infoset=new_latent_infosets[:, None, ...])
    
    def repeat_and_flatten(x: chex.Array):
      """Helper function, repeat as needed over the first two axes and flatten them into
      a single axis"""
      first_reps = max(1, num_outcomes - x.shape[0])
      second_reps = max(1, all_deters.shape[0] - x.shape[1])
      x = jnp.repeat(x, first_reps, axis=0)
      x = jnp.repeat(x, second_reps, axis=1)
      return x.reshape((-1, *x.shape[2:]))
    #Now do the tiling and also flatten the first two dimensions
    outcomes = jax.tree.map(lambda x: repeat_and_flatten(x), outcomes)
    return outcomes, outcome_probs


  def params_dict(self) ->dict:
    return self.game.params_dict()

  def game_name(self)->str:
    return f"model_{self.game.game_name()}"
  
  def num_distinct_actions(self):
    return self.actions
  
  def num_players(self):
    return self.players
  
  def information_type(self):
    return self.game.information_type()
  
  def max_trajectory_length(self):
    return self.trajectory_max
  
  def information_state_tensor_shape(self):
    if self.use_real_infoset:
      return self.game.information_state_tensor_shape()
    return self.recurrent_state_size + (self.num_categories ** self.num_classes)
  
  def observation_tensor_shape(self):
    return self.information_state_tensor_shape()
  

  def cache_model_calls(self):
    """Cache calls to the model networks for the given model.
    This is called on init automatically and should be called again when
    the model networks update"""
    
    def next_recur_wrapper(graphdef: nnx.GraphDef, state: nnx.State, recurrent_state:chex.Array,
                        deter_state: chex.Array, action: chex.Array):
      ma_rssm = nnx.merge(graphdef, state)
      next_recurrent = ma_rssm.get_next_recurrent_no_jit(recurrent_state, deter_state, action)
      return next_recurrent
    
    def next_infosets_wrapper(graphdef: nnx.GraphDef, state: nnx.State, joint_latent_infosets: chex.Array,
                           obs: chex.Array, prev_action: chex.Array):
      ma_rssm = nnx.merge(graphdef, state)
      next_infosets = ma_rssm.get_next_infoset_all_no_jit(joint_latent_infosets, obs, prev_action)
      return next_infosets
    
    def encoder_wrapper(graphdef: nnx.GraphDef, state: nnx.State, recurrent_state: chex.Array, obs: chex.Array):
      ma_rssm = nnx.merge(graphdef, state)
      stoch_logits = ma_rssm.get_encoder_no_jit(recurrent_state, obs)
      stoch_unfiltered = nnx.softmax(stoch_logits, axis=-1)
      stoch_unnormalized = stoch_unfiltered * (stoch_unfiltered >= self.probabilty_threshold)
      normalization = jnp.sum(stoch_unnormalized, axis=-1, keepdims=True)
      normalization = normalization + (normalization == 0)
      stoch = stoch_unnormalized / normalization
      return stoch

    ma_rssm_graphdef, ma_rssm_state = nnx.split(self.ma_rssm)
    #self.cached_predictor = partial(predictors_wrapper, ma_rssm_graphdef, ma_rssm_state)
    #self.cached_decoder = partial(decoders_wrapper, ma_rssm_graphdef, ma_rssm_state)
    self.cached_encoder = partial(encoder_wrapper, ma_rssm_graphdef, ma_rssm_state)
    self.cached_sequential = partial(next_recur_wrapper, ma_rssm_graphdef, ma_rssm_state)
    self.cached_infoset_net = partial(next_infosets_wrapper, ma_rssm_graphdef, ma_rssm_state)

    #self.get_init_chance_outcomes()


  @partial(nnx.jit, static_argnums=0)
  def initialize_structures(self):
    """return init game state (chance node), init legals (dummy)"""
    init_game_state, _ = self.game.initialize_structures()
    #Assuming the original game produces some output even in chance node
    init_obs = self.get_real_game_obs(init_game_state)
    init_deter = jnp.zeros((self.players, self.num_classes, self.num_categories), dtype=u8)
    #Get the actual stochastic state, if the
    # original game starts with a chance node
    init_stoch = self.cached_encoder(self.init_recurrent, init_obs)
    init_legals = jnp.ones((self.players, self.actions), dtype=u8)
    init_state = ModelGameState(turn=0, game_state=init_game_state,
                                terminal=jnp.array(False, dtype=bool),
                                reward=jnp.array(0, dtype=f32),
                                legals=init_legals, recurrent_state=self.init_recurrent,
                                 deter_state=init_deter,
                                 stoch_state=init_stoch)
    return init_state, init_legals
  
  @partial(jax.jit, static_argnums=0)
  def get_info(self, game_state: ModelGameState):
    """Follows the jax game  state_tensor, p1_infoset, p2_infoset, public_state convention
    in our case, defining the  player model infoset latent_infoset of pl
    and the model state as [recurrent_state, deterministic_categorical_state],
    it is model_state, p1_model_infoset, p2_model_infoset, model_state.
    IMPORTANT: As it is unclear how to recover the public state
    from the model infosets, we just return the perfect information
    model state for consistency of the number of returned arguments.
    However, this is just invalid output and it does NOT represent
    the actual public state."""
    #Return the real environment info, if the config is set up that way
    if self.use_real_infoset:
      return self.game.get_info(game_state.game_state)
    
    p1_model_infoset, p2_model_infoset = game_state.joint_latent_infoset[0], game_state.joint_latent_infoset[1]

    model_state = jnp.concatenate([game_state.recurrent_state, game_state.deter_state.ravel()], axis=0)

    return model_state, p1_model_infoset, p2_model_infoset, model_state
  

  def state_tensor_shape(self)->int:
    return self.recurrent_state_size + self.deter_state_size if not self.use_real_infoset else self.game.state_tensor_shape()
  
  def public_state_tensor_shape(self)->int:
    return self.state_tensor_shape() if not self.use_real_infoset else self.game.public_state_tensor_shape()
  
  def information_state_tensor_shape(self)->int:
    return self.latent_infoset_size if not self.use_real_infoset else self.game.information_state_tensor_shape()

  @partial(jax.jit, static_argnums=0)
  def apply_action(self, game_state: ModelGameState, action: chex.Array):
    """Return new_game_state, terminal, reward, new_legals"""
    #return self.apply_action_chance(game_state, action)
    new_state, terminal, reward, new_legal =  jax.lax.cond(self.is_chance(game_state), self.apply_action_chance
                        , self.apply_action_no_chance, game_state, action)
    return new_state, terminal, reward, new_legal


  @partial(jax.jit, static_argnums=0)
  def apply_action_chance(self, game_state: ModelGameState, action: chex.Array):
    #TODO: Can we get rid of this duplicate call somehow?
    outcomes, _ = self.get_chance_outcomes(game_state)
    #Following the convention that player 2 has the actions in chance nodes
    outcome_idx = action[1]
    next_state = jax.tree.map(lambda x: x[outcome_idx], outcomes)
    next_legal = next_state.legals
    next_terminal = next_state.terminal
    next_reward = next_state.reward
    
    return next_state, next_terminal, next_reward, next_legal



  @partial(jax.jit, static_argnums=(0))
  def apply_action_no_chance(self, game_state: ModelGameState, action: chex.Array):
    action_oh = jax.nn.one_hot(action, self.actions, axis=-1)
    next_recurrent = self.cached_sequential(game_state.recurrent_state, game_state.deter_state, action_oh)
    #The deterministic states are sampled at chance nodes
    next_deter = jnp.zeros_like(game_state.deter_state)
    obs = self.get_real_game_obs(game_state.game_state)
    next_stoch = self.cached_encoder(next_recurrent, obs)

    next_game_state, game_term, game_rew, game_legals = self.game.apply_action(game_state.game_state, action)

    #Terminal, reward and legals are decided from predictors
    # after chance node outcomes
    next_terminal = jnp.array(False)
    next_reward = jnp.array(0, dtype=f32)
    next_legals = jnp.ones((self.players, self.actions), dtype=u8)

    next_state = ModelGameState(turn = game_state.turn + 1,
                                game_state=next_game_state,
                                legals = game_legals.astype(u8),
                                reward=game_rew,
                                terminal=game_term,
                                recurrent_state= next_recurrent,
                                deter_state= next_deter,
                                stoch_state=next_stoch)
    
    return next_state, next_terminal, next_reward, next_legals

  def is_chance(self, game_state: ModelGameState) -> chex.Array:
    return game_state.turn % 2 == 0
  
  
  @partial(jax.jit, static_argnums=0)
  def get_outcomes_and_probs(self, game_state: ModelGameState) -> tuple[chex.Array, chex.Array]:
    """return array of [outcomes, probs], return invalid data when this is
    not a chance node. Pad it to the lenght of max_chance_outcomes"""
    outcomes = jnp.stack([jax.nn.one_hot(0, self.max_chance_outcomes), jnp.arange(self.max_chance_outcomes)], axis=-1)
    def invalid_probs(game_state):
      return jnp.zeros(self.max_chance_outcomes)
    def deter_probs(game_state: ModelGameState):
      #[max chance outcomes, num_classes]
      _,  outcome_probs = self.get_chance_outcomes(game_state)
      return outcome_probs
    probs = jax.lax.cond(self.is_chance(game_state), deter_probs, invalid_probs, game_state)
    return outcomes, probs

  def depth_chance_outcomes(self, depth) ->int:
    if depth % 2 == 0:
      return self.max_chance_outcomes
    return 1

  @partial(jax.jit, static_argnums=0)
  def state_valid_chance_outcomes(self, state: ModelGameState) ->int:
    """Return the actual number of valid chance outcomes. Here I intentionally
    deviate from the per depth of a standard JaxGame, since I do not
    have a guarantee that at given depth all states have learned
    the same amount of outcomes"""
    #this gives us the number of valid outcomes for each categorical
    valid_outcomes = jnp.sum(state.stoch_state > 0, axis=-1)
    num_total_outcomes = jnp.prod(valid_outcomes, axis=-1)
    num_total_outcomes = jnp.where(state.turn == 0, self.init_valid_outcomes, num_total_outcomes)
    return num_total_outcomes.astype(int)
  

# if __name__ == "__main__":
#   class_indices = jnp.tile(jnp.arange(3), (3, 1))
#   outcome_indices = cartesian_product(*class_indices)
#   outcomes_oh = nnx.one_hot(outcome_indices, 3, axis=-1)
#   jax.debug.breakpoint()

  