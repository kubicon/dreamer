import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import chex

from functools import partial

from dreamer_ma import DreamerMA
from experiments.eval_utils import cartesian_product, unroll_chance_node
from games.jax_game import GameState
from ma_rssm import MARSSM

@chex.dataclass
class ModelGameState:
  game_state: GameState
  legals: chex.Array
  reward: chex.Array
  terminal: chex.Array
  turn: int
  recurrent_state: chex.Array # The recurrent network state
  deter_state: chex.Array # The one-hot sampled outcome of the categoricals
  stoch_state: chex.Array # The softmaxed distribution over the latent deter states
                          # with the invalid outcomes already filtered out

u8 = jnp.uint8
f32 = jnp.float32

def zero_pad_x(x, to_pad:int):
  padding_tup = ((0, to_pad, ), *[(0, 0) for _ in range(x.ndim - 1)])
  padded_x = jnp.pad(x, padding_tup, mode='constant', constant_values=0)
  return padded_x

class DreamerModelGame():
  def __init__(self, model: DreamerMA, probability_threshold = 0.05):
    """A game which has structure corresponding to
    a JAX game, but it is instead built from the learned Dreamer
    world model. It operates like this: It begins with a
    chance node that has outcomes corresponding to the learned categoricals.
    Similarly, every action is followed by such a chance node.
    It is used just for the actor-critic evaluation, so that
    we can seamlessly transition between original games and model games.
    This game is currently defined only for perfect information games.

    It also walks in the real game, and the returned reward, terminal
     and legal information are from the real environment. For the initial chance
     node, we know which outcome the sampled categoricals correspond to.
     For other chance nodes, for each possible outcome the closest 
     chance outcome in the real game is selected, where the closeness
     is defined as distance between real game observation and model decoded observation.


    IMPORTANT! Due to padding the chance outcomes to the maximum
    amount of chance outcomes (categories ** classes), this will only reasonably work
    for small latent stochastic states."""
    self.game = model.game
    self.recurrent_state_size = model.recurrent_state_size
    self.ma_rssm = model.optimizer.model
    self.num_classes = model.wm_config.encoded_classes
    self.num_categories = model.wm_config.encoded_categories

    self.deter_state_size = self.num_classes * self.num_categories
    self.players = model.game.num_players()

    self.actions = model.game.num_distinct_actions()

    self.max_chance_outcomes = (self.num_categories) ** self.num_classes
    #Every step will have a corresponding chance node
    self.trajectory_max = 2 * model.trajectory_max

    self.probabilty_threshold = probability_threshold

    
    class_indices = jnp.tile(jnp.arange(self.num_categories), (self.num_classes, 1))
    #Precomputing the indices of the individual outcomes. These will be the same
    # only the probabilities will differ. 
    # shape [max_chance_outcomes, encoded_classes]
    self.outcome_indices = cartesian_product(*class_indices)
    self.init_recurrent = self.ma_rssm.get_init_recurrent()

    self.cache_model_calls()

  
  @partial(nnx.jit, static_argnums=0)
  def get_init_stoch(self, ma_rssm: MARSSM, obs):
    stoch_logits = ma_rssm.get_encoder(self.init_recurrent, obs)
    stoch_unfiltered = nnx.softmax(stoch_logits, axis=-1)
    stoch_unnormalized = stoch_unfiltered * (stoch_unfiltered >= self.probabilty_threshold)
    normalization = jnp.sum(stoch_unnormalized, axis=-1, keepdims=True)
    normalization = normalization + (normalization == 0)
    stoch = stoch_unnormalized / normalization
    return stoch


  def get_init_chance_outcomes(self):
    """The first stochastic state is special, since that 
     one requires the observation from the environment. Unroll for
     all possible observations (if the original game begins with a chance node)"""
    orig_game_init_state, init_legal = self.game.initialize_structures()
    if self.game.is_chance(orig_game_init_state):
      init_chance_valid_outcomes = self.game.depth_chance_valid_outcomes(0)
      init_states, init_terminal, init_rewards, init_legal, _ = unroll_chance_node(self.game, orig_game_init_state, init_chance_valid_outcomes)
      vectorized_get_info = jax.vmap(self.game.get_info, in_axes=(0), out_axes=(0, 0, 0, 0))
      _, p1_isets, p2_isets, _ = vectorized_get_info(init_states)
      observations = jnp.stack([p1_isets, p2_isets], axis=1)
    else:
      _, p1_iset, p2_iset, _ = self.game.get_info(orig_game_init_state)
      init_states = jax.tree.map(lambda x: x[None, ...], orig_game_init_state)
      observations = jnp.stack([p1_iset, p2_iset], axis=0)
      observations = observations[None, ...]
      init_legal = init_legal[None, ...]
      init_terminal = jnp.zeros(1, dtype=bool)
      init_rewards = jnp.zeros(1, dtype=f32)
    get_init_stochs = nnx.vmap(self.get_init_stoch, in_axes=(None, 0), out_axes=0)
    stochs = get_init_stochs(self.ma_rssm, observations)
    
    init_deters = []
    states = []
    legals = []
    rewards = []
    terminal = []
    probs = []
    class_arange = np.arange(self.num_classes)
    def get_deter_idx(deter: np.ndarray):
      for i, d in enumerate(init_deters):
        if np.sum((deter - d) ** 2) <= 1e-8:
          return i
      return -1
    for i, stoch in enumerate(stochs):
      state = jax.tree.map(lambda x: x[i], init_states)
      legal = init_legal[i]
      rew = init_rewards[i]
      term = init_terminal[i]
      outcome_arrays = [np.nonzero(c)[0] for c in stoch]
      possible_deter_indices = cartesian_product(*outcome_arrays)
      for indices in possible_deter_indices:
        deter = np.asarray(jax.nn.one_hot(indices, self.num_categories, axis=-1), dtype=np.int8)
        per_class_prob = stoch[class_arange, indices]
        prob = np.prod(per_class_prob)
        #This deter was already visited
        deter_idx = get_deter_idx(deter)
        if deter_idx >= 0:
          probs[deter_idx] += prob
          continue
        init_deters.append(deter)
        legals.append(legal)
        rewards.append(rew)
        terminal.append(term)
        states.append(state)
        probs.append(prob)
    states = jax.tree.map(lambda *x: jnp.stack(x, axis=0), *states)
    legals = jnp.asarray(legals, dtype=u8)
    terminal = jnp.asarray(terminal, dtype=bool)
    rewards = jnp.asarray(rewards)
    init_deters = jnp.asarray(init_deters, dtype=u8)
    init_probs = jnp.asarray(probs)

    self.init_valid_outcomes = len(init_deters)

    to_pad = self.max_chance_outcomes - self.init_valid_outcomes

    self.init_deters = zero_pad_x(init_deters, to_pad)
    self.init_legals = zero_pad_x(legals, to_pad)
    self.init_rewards = zero_pad_x(rewards, to_pad)
    self.init_terminal = zero_pad_x(terminal, to_pad)
    self.init_states = jax.tree.map(lambda x: zero_pad_x(x, to_pad), states)
    self.init_probs = zero_pad_x(init_probs, to_pad)

  
  def params_dict(self) ->dict:
    return self.game.params_dict()

  def game_name(self)->str:
    return f"model_{self.game.game_name()}"
  
  def num_distinct_actions(self):
    return self.actions
  
  def num_players(self):
    return self.players
  

  def cache_model_calls(self):
    """Cache calls to the model networks for the given model.
    This is called on init automatically and should be called again when
    the model networks update"""
    # def predictors_wrapper(graphdef: nnx.GraphDef, state: nnx.State, recurrent_state:chex.Array,
    #                     deter_state: chex.Array):
    #   ma_rssm = nnx.merge(graphdef, state)
    #   reward, terminal, legal = ma_rssm.get_predictor_no_jit(recurrent_state, deter_state)
    #   return reward, terminal, legal

    def decoders_wrapper(graphdef: nnx.GraphDef, state: nnx.State, recurrent_state: chex.Array,
                         deter_state: chex.Array):
      ma_rssm = nnx.merge(graphdef, state)
      p1_decoded = ma_rssm.get_decoder(recurrent_state, deter_state, player=0)
      p2_decoded = ma_rssm.get_decoder(recurrent_state, deter_state, player=1)
      return jnp.stack([p1_decoded, p2_decoded], axis=0)
    
    def next_recur_wrapper(graphdef: nnx.GraphDef, state: nnx.State, recurrent_state:chex.Array,
                        deter_state: chex.Array, action: chex.Array):
      ma_rssm = nnx.merge(graphdef, state)
      next_recurrent = ma_rssm.get_next_recurrent_no_jit(recurrent_state, deter_state, action)
      return next_recurrent
    
    def dynamics_wrapper(graphdef: nnx.GraphDef, state: nnx.State, recurrent_state: chex.Array):
      ma_rssm = nnx.merge(graphdef, state)
      stoch_logits = ma_rssm.dyn(recurrent_state)
      stoch_unfiltered = nnx.softmax(stoch_logits, axis=-1)
      stoch_unnormalized = stoch_unfiltered * (stoch_unfiltered >= self.probabilty_threshold)
      normalization = jnp.sum(stoch_unnormalized, axis=-1, keepdims=True)
      normalization = normalization + (normalization == 0)
      stoch = stoch_unnormalized / normalization
      return stoch

    ma_rssm_graphdef, ma_rssm_state = nnx.split(self.ma_rssm)
    #self.cached_predictor = partial(predictors_wrapper, ma_rssm_graphdef, ma_rssm_state)
    self.cached_decoders = partial(decoders_wrapper, ma_rssm_graphdef, ma_rssm_state)
    self.cached_dynamics = partial(dynamics_wrapper, ma_rssm_graphdef, ma_rssm_state)
    self.cached_sequential = partial(next_recur_wrapper, ma_rssm_graphdef, ma_rssm_state)

    self.get_init_chance_outcomes()


  @partial(nnx.jit, static_argnums=0)
  def initialize_structures(self):
    """return init game state (chance node), init legals (dummy)"""
    init_game_state, _ = self.game.initialize_structures()
    init_deter = jnp.zeros((self.num_classes, self.num_categories), dtype=u8)
    #Just a dummy, the outcomes are actually precomputed,
    # because the first node is special and we take it
    # posterior (with the observation)
    init_stoch = self.cached_dynamics(self.init_recurrent)
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
    """return state_tensor, p1_iset, p2_iset, public_state
    in our case, defining the model state as [recurrent_state, deter_state]
    it is model_state, [model_state, one_hot(p1)], [model_state, one_hot(p2)], model_state.
    TODO: For IIG also put some sort of flag that instead returns the decoded
    isets as observations. Public state tensor from those two
    could not be gotten in a straightforward way though."""

    flat_deter = jnp.reshape(game_state.deter_state, (*game_state.deter_state.shape[:-2], -1))
    model_state = jnp.concatenate([game_state.recurrent_state, flat_deter], axis=0)
    
    p1_iset = jnp.concatenate([model_state, jax.nn.one_hot(0, self.players)], axis=0)
    p2_iset = jnp.concatenate([model_state, jax.nn.one_hot(1, self.players)], axis=0)

    return model_state, p1_iset, p2_iset, model_state
  

  def state_tensor_shape(self)->int:
    return self.recurrent_state_size + self.deter_state_size
  
  def public_state_tensor_shape(self)->int:
    return self.state_tensor_shape()
  
  def information_state_tensor_shape(self)->int:
    return self.state_tensor_shape() + self.players

  @partial(jax.jit, static_argnums=0)
  def apply_action(self, game_state: ModelGameState, action: chex.Array):
    """Return new_game_state, terminal, reward, new_legals"""
    #return self.apply_action_chance(game_state, action)
    new_state, terminal, reward, new_legal =  jax.lax.cond(self.is_chance(game_state), self.apply_action_chance
                        , self.apply_action_no_chance, game_state, action)
    return new_state, terminal, reward, new_legal

  def get_closest_next_idx(self, recurrent_state: chex.Array, deter: chex.Array, 
                           next_states: GameState, next_state_probs: chex.Array):
    dec_output = self.cached_decoders(recurrent_state, deter)
    vectorized_get_obs = jax.vmap(self.game.get_info, in_axes=(0), out_axes=0)
    _, p1_obs, p2_obs, _= vectorized_get_obs(next_states)
    obs = jnp.stack([p1_obs, p2_obs], axis=1)
    next_dist = jnp.sum((dec_output[None, ...] - obs) ** 2, axis=(-1, -2))
    max_dist = jnp.max(next_dist)
    #Mask out the states that cannot happen
    valid = next_state_probs >= 1e-8
    next_dist = valid * (next_dist) + (1 - valid) * (max_dist + 1)
    return jnp.argmin(next_dist)




  @partial(jax.jit, static_argnums=0)
  def expand_states(self, game_state: ModelGameState, next_deters: chex.Array):
    is_chance = self.game.is_chance(game_state.game_state)
    num_outcomes = next_deters.shape[0]
    def no_chance_next():
      def tile_x(x):
        return jnp.repeat(x[None, ...], num_outcomes, axis=0)
      game_states = jax.tree.map(lambda x: tile_x(x), game_state.game_state)
      rewards = tile_x(game_state.reward)
      terminals = tile_x(game_state.terminal)
      legals = tile_x(game_state.legals)
      return game_states, terminals, rewards, legals
    def chance_next():
      # Unroll the chance node (including
      # not reacheable outcomes as well, to be jittable.)
       # and then select the closest state based 
       # on decoder for each deter. This will
       # be VERY expensive and work only for small instances.
       next_chance_outcomes, next_chance_probs = self.game.get_outcomes_and_probs(game_state.game_state)
       #Do not vmap over the current state, that will be the same.
       vectorized_apply_action = jax.vmap(self.game.apply_action, in_axes=(None, 0))
       next_game_states, next_terminal, next_rewards, next_legal = vectorized_apply_action(game_state.game_state, next_chance_outcomes)
       #vmap only over the deter states
       vectorized_closest_idx = jax.vmap(self.get_closest_next_idx, in_axes=(None, 0, None, None), out_axes=0)
       closest_indices = vectorized_closest_idx(game_state.recurrent_state, next_deters, next_game_states, next_chance_probs)
       game_states = jax.tree.map(lambda x : x[closest_indices], next_game_states)
       rewards, terminals, legals = next_rewards[closest_indices], next_terminal[closest_indices], next_legal[closest_indices]
       return game_states, terminals, rewards, legals.astype(u8)
    return jax.lax.cond(is_chance, chance_next, no_chance_next)

  @partial(jax.jit, static_argnums=0)
  def apply_action_chance(self, game_state: ModelGameState, action: chex.Array):
    #[max_chance_outcomes, num_classes, num_categories]
    next_deters = jax.nn.one_hot(self.outcome_indices, self.num_categories, axis=-1, dtype=game_state.deter_state.dtype)
    is_first = game_state.turn == 0
    next_deters = jnp.where(is_first, self.init_deters, next_deters)
    #Following the convention, that the player 2 component chooses the outcome
    selected_outcome = action[1].astype(jnp.int32)
    #in the chance node, we just choose the
    # deterministic state that was sampled.
    # We do not alter the hidden or stochastic states
    next_deter = next_deters[selected_outcome]

    def first_turn_states():
      return self.init_states, self.init_terminal, self.init_rewards, self.init_legals
    def other_turn_states():
      return self.expand_states(game_state, next_deters)

    next_states, next_terminals, next_rewards , next_legals = jax.lax.cond(is_first, first_turn_states, other_turn_states)
    next_state = jax.tree.map(lambda x: x[selected_outcome], next_states) 
    next_reward, next_terminal, next_legal = next_rewards[selected_outcome], next_terminals[selected_outcome], next_legals[selected_outcome]

    max_depth_reached = game_state.turn == self.trajectory_max
    next_terminal = jnp.logical_or(next_terminal, max_depth_reached)

    next_state = ModelGameState(turn=game_state.turn + 1,
                                game_state = next_state,
                                reward=next_reward,
                                terminal = next_terminal,
                                legals=next_legal,
                                recurrent_state=game_state.recurrent_state,
                                deter_state=next_deter,
                                stoch_state=game_state.stoch_state)
    
    return next_state, next_terminal, next_reward, next_legal



  @partial(jax.jit, static_argnums=(0))
  def apply_action_no_chance(self, game_state: ModelGameState, action: chex.Array):
    action_oh = jax.nn.one_hot(action, self.actions, axis=-1)
    next_hidden = self.cached_sequential(game_state.recurrent_state, game_state.deter_state, action_oh)
    #The deterministic states are sampled at chance nodes
    next_deter = jnp.zeros_like(game_state.deter_state)
    next_stoch = self.cached_dynamics(next_hidden)

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
                                recurrent_state= next_hidden,
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
      outcome_per_class_probs = game_state.stoch_state[jnp.arange(self.num_classes), self.outcome_indices]
      outcome_probs = jnp.prod(outcome_per_class_probs, axis=-1)
      return outcome_probs
    probs = jax.lax.cond(self.is_chance(game_state), deter_probs, invalid_probs, game_state)
    probs = jnp.where(game_state.turn == 0, self.init_probs, probs)
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

  