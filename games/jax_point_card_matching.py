import jax
import jax.numpy as jnp

import chex
import functools

from games.jax_game import JaxGame, GameState, InformationType

"""A single player goofspiel inspired game.
Basically just a match the point card game.
This version is actually treated as a 2p0s
perfect information game with a dummy player 2 """

@chex.dataclass(frozen=True)
class PointCardMatchingState(GameState):
  #history of one hot played cards
  played_cards: chex.Array
  #history of one hot point cards
  points: chex.Array
  point_card: chex.Array
  terminal: chex.Array
  turn: chex.Array


class PointCardMatching(JaxGame):
  def __init__(self, num_cards):
    assert num_cards >= 3, f"The point card matching game is only defined for num card >= 3. Given was {num_cards}"
    self.num_cards = num_cards
    self.max_turns = num_cards
    #get a reward 1 whenever a card is matched
    self.max_points = num_cards

  

  def initialize_structures(self):
    init_played_cards = jnp.zeros((self.max_turns, self.num_cards))
    init_points = jnp.zeros(1)
    init_point_card = jax.nn.one_hot(self.num_cards - 1, self.num_cards)
    init_state = PointCardMatchingState(played_cards = init_played_cards,
                                        points = init_points,
                                        point_card = init_point_card,
                                        terminal = jnp.array(False),
                                        turn = jnp.array(0))
    init_legals = jnp.stack([jnp.ones((self.num_cards)), jax.nn.one_hot(0, self.num_cards)], axis=0)
    return init_state, init_legals
  
  @functools.partial(jax.jit, static_argnums=(0))
  def get_info(self, state: PointCardMatchingState):
    #starting at 0 points hence the + 1
    points_oh = jax.nn.one_hot(state.points, self.max_points + 1)
    state_tensor = jnp.concatenate([state.played_cards.ravel(), state.point_card.ravel(), points_oh.ravel()])
    p1_infoset = jnp.concatenate([jax.nn.one_hot(0, 2), state_tensor], axis=0)
    p2_infoset = jnp.concatenate([jax.nn.one_hot(1, 2), state_tensor], axis=0)
    return state_tensor, p1_infoset, p2_infoset, state_tensor
  
  def num_distinct_actions(self):
    return self.num_cards
  
  def max_trajectory_length(self):
    return self.max_turns
  
  def game_name(self):
    return "pcm"
  
  def params_dict(self):
    return {"num_cards": self.num_cards}
  
  def information_type(self):
    return InformationType.PIG

  def num_players(self):
    return 2
  
  def observation_tensor_shape(self):
    return self.information_state_tensor_shape()
  
  def public_state_tensor_shape(self):
    return self.information_state_tensor_shape()
  
  def state_tensor_shape(self):
    return (self.max_turns * self.num_cards) + self.num_cards + self.max_points + 1
  
  def information_state_tensor_shape(self):
    return 2 + self.state_tensor_shape()
  

  @functools.partial(jax.jit, static_argnums=(0))
  def apply_action(self, state: PointCardMatchingState, action):
    turn_oh = jax.nn.one_hot(state.turn, self.max_turns)
    #The second player is a dummy player
    action = action[0]
    action_oh = jax.nn.one_hot(action, self.num_cards)

    new_played_cards = state.played_cards + (action_oh[None, ...] * turn_oh[..., None])
    already_played = jnp.sum(new_played_cards, axis=0)
    new_legals = jnp.ones(self.num_cards) - already_played

    #descending order
    point_card = self.num_cards - state.turn - 2
    point_card_oh = jax.nn.one_hot(point_card, self.num_cards)
    #Match the action on the PREVIOUS point card
    new_points = state.points + (point_card + 1 == action)

    terminal = state.turn == (self.num_cards - 2)
    terminal = state.terminal + terminal
    #new_point_cards = jnp.where(turn ==(self.max_turns - 1), state.point_cards, new_point_cards)
    #checking if we can still match the
    #last point card, which will be the first card
    # because of descending order.
    last_card_matched = jnp.sum(new_legals * jax.nn.one_hot(0, self.num_cards))
    reward = jnp.where(terminal, new_points + last_card_matched, jnp.zeros_like(new_points + last_card_matched))
    new_legals = jnp.stack([new_legals, jax.nn.one_hot(0, self.num_cards)], axis=0)

    new_state = PointCardMatchingState(played_cards=new_played_cards,
                                       point_card = point_card_oh,
                                       points= new_points,
                                       terminal = terminal,
                                       turn = state.turn + 1)
    return new_state, terminal, reward[0], new_legals
  

@chex.dataclass(frozen=True)
class PointCardMatchingStochasticState(GameState):
  #history of one hot played cards
  played_cards: chex.Array
  #history of one hot point cards
  points: chex.Array
  point_cards: chex.Array
  terminal: chex.Array
  turn: int
  is_chance: chex.Array

class PointCardMatchingStochastic(JaxGame):
  """A point card matching variant with a single
  chance node at the end. Cards are revealed in a descending order,
  except the chance node level, when the revealed card is chosen at random
  and then the game continues in descending order."""
  def __init__(self, num_cards: int, chance_turn_before_terminal: int = 1):
    """chance_turn_before_terminal specifies how many turns before a terminal
    turn willl the chance node happen. For example when chance_turn_before_terminal == 1,
    then the chance node happens on turn num_cards - 2, when there is a choice
    between only point cards 1 or 2.
    Assure that num_cards >= 3"""
    assert num_cards >= 3, f"The point card matching game is only defined for num card >= 3. Given was {num_cards}"
    self.num_cards = num_cards
    self.max_turns = num_cards
    #get a reward 1 whenever a card is matched
    self.max_points = num_cards
    self.chance_turn = self.num_cards - 1 - chance_turn_before_terminal
    assert self.chance_turn >= 0, f"Invalid config with {self.num_cards} and {chance_turn_before_terminal}, the chance node is set to happen at invalid turn {self.chance_turn}."

    self.chance_outcomes = self.num_cards - self.chance_turn

  def initialize_structures(self):
    init_chance = jnp.array(self.chance_turn == 0)
    init_played_cards = jnp.zeros((self.max_turns, self.num_cards))
    init_points = jnp.zeros(1)
    init_point_cards = jnp.concatenate([jax.nn.one_hot(self.num_cards - 1, self.num_cards)[None, ...], jnp.zeros((self.num_cards - 1, self.num_cards))], axis=0)
    no_point_card_dealt = jnp.zeros((self.num_cards, self.num_cards))
    init_state = PointCardMatchingStochasticState(played_cards = init_played_cards,
                                        points = init_points,
                                        point_cards = jnp.where(init_chance, no_point_card_dealt, init_point_cards),
                                        terminal = jnp.array(False),
                                        turn= jnp.array(0),
                                        is_chance = init_chance)
    init_legals = jnp.stack([jnp.ones((self.num_cards)), jax.nn.one_hot(0, self.num_cards)], axis=0)
    return init_state, init_legals
  
  @functools.partial(jax.jit, static_argnums=(0))
  def get_info(self, state: PointCardMatchingStochasticState):
    #starting at 0 points hence the + 1
    points_oh = jax.nn.one_hot(state.points, self.max_points + 1)
    state_tensor = jnp.concatenate([state.played_cards.ravel(), state.point_cards.ravel(), points_oh.ravel()])

    #Return invalid data on chance turn
    state_tensor = jnp.where(state.is_chance, jnp.zeros_like(state_tensor), state_tensor)
    p1_infoset = jnp.concatenate([jax.nn.one_hot(0, 2), state_tensor], axis=0)
    p2_infoset = jnp.concatenate([jax.nn.one_hot(1, 2), state_tensor], axis=0)
    return state_tensor, p1_infoset, p2_infoset, state_tensor
  
  def num_distinct_actions(self):
    return self.num_cards
  
  def max_trajectory_length(self):
    # Normally would be self.num_cards, but a chance node is also added to the trajectory
    return self.num_cards + 1
  
  def max_trajectory_lenght_no_chance(self):
    return self.num_cards
  
  def game_name(self):
    return "pcm_stochastic"
  
  def params_dict(self):
    return {"num_cards": self.num_cards, "chance_turn": self.chance_turn}
  
  def information_type(self):
    return InformationType.PIG

  def num_players(self):
    return 2
  
  def observation_tensor_shape(self):
    return self.information_state_tensor_shape()
  
  def public_state_tensor_shape(self):
    return self.information_state_tensor_shape()
  
  def state_tensor_shape(self):
    return 2 * (self.max_turns * self.num_cards) + self.max_points + 1
  
  def information_state_tensor_shape(self):
    return 2 + self.state_tensor_shape()
  
  @functools.partial(jax.jit, static_argnums=(0))
  def generate_all_chance_outcomes(self, chance_state: PointCardMatchingStochasticState) ->tuple[PointCardMatchingStochasticState, chex.Array, chex.Array]:
    """Generate all chance outcomes for a chance node 
    state, which happens on self.chance_turn
    . There are num_cards - 1 - self.chance_turn choices for the chance node.
    These outcomes will be returned as a list, sorted in ascending order by the point card."""
    #chance node chooses uniformly from the available point cards
    played_point_cards = jnp.sum(chance_state.point_cards, axis=0)
    legal_point_cards  = 1 - played_point_cards

    chance_turn_oh = jax.nn.one_hot(self.chance_turn, self.max_turns)
    
    valid_outcomes = jnp.nonzero(legal_point_cards, size=self.chance_outcomes)[0]
    valid_outcomes_oh = jax.nn.one_hot(valid_outcomes, self.num_cards)
    outcome_probs = jnp.ones(self.chance_outcomes) / self.chance_outcomes

    already_played = jnp.sum(chance_state.played_cards, axis=0)
    outcome_legals = jnp.ones((self.chance_outcomes, self.num_cards)) - already_played[None, ...]

    stacked_state = jax.tree.map(lambda x: jnp.tile(x[None, ...], (self.chance_outcomes,) + (1,) * len(x.shape)).astype(x.dtype), chance_state)
    #This returns an array of shape 
    #[self.chance_outcomes, self.max_turns, self.num_cards]
    # using broadcasting to mask the oh_card into the proper turn as well as 
    # broadcasting over the outcomes. It has the one-hot encoded outcomes in the proper turn
    outcome_point_cards = valid_outcomes_oh[:, None, ...] * chance_turn_oh[None, ..., None]
    #Add it to the history of point cards
    outcome_point_cards = stacked_state.point_cards + outcome_point_cards

    outcome_states = PointCardMatchingStochasticState(played_cards = stacked_state.played_cards,
                                              points = stacked_state.points,
                                              terminal = stacked_state.terminal,
                                              point_cards = outcome_point_cards,
                                              turn = stacked_state.turn,
                                              is_chance = jnp.zeros_like(stacked_state.is_chance, dtype=bool))
    
    return outcome_states, outcome_legals, outcome_probs
  
  def max_chance_outcomes(self):
    return self.chance_outcomes
  
  def is_chance(self, game_state: PointCardMatchingStochasticState):
    return game_state.is_chance
  
  def depth_chance_outcomes(self, depth: int):
    if depth == self.chance_turn:
      return self.num_cards
    return 1
  
  def depth_chance_valid_outcomes(self, depth: int):
    if depth == self.chance_turn:
      return self.chance_outcomes
    return 1
  
  @functools.partial(jax.jit, static_argnums=(0))
  def get_outcomes_and_probs(self, game_state:PointCardMatchingStochasticState) -> tuple[PointCardMatchingStochasticState, chex.Array, chex.Array]:
    outcomes = jnp.stack([jax.nn.one_hot(0, self.num_cards), jnp.arange(self.num_cards)], axis=-1)
    def invalid_probs(game_state):
      return jnp.zeros(self.num_cards)
    def chance_probs(game_state: PointCardMatchingStochasticState):
      played_point_cards = jnp.sum(game_state.point_cards, axis=0)
      legal_point_cards  = 1 - played_point_cards
      return legal_point_cards / jnp.sum(legal_point_cards)
    probs = jax.lax.cond(game_state.is_chance,
                                           chance_probs,
                                           invalid_probs, game_state)
    return outcomes, probs

    

  
  @functools.partial(jax.jit, static_argnums=(0))
  def apply_action(self, state: PointCardMatchingStochasticState, actions):
    #print(f"Actions shape : {actions.shape}")
    #jax.debug.breakpoint()
    return jax.lax.cond(state.is_chance, self.apply_action_chance, self.apply_action_no_chance, state, actions)


  @functools.partial(jax.jit, static_argnums=(0))
  def apply_action_chance(self, state: PointCardMatchingStochasticState, action):
    #The convention is that in chance nodes player 2 
    # action contains the outcome in 2 player games
    action = action[1]
    turn_oh = jax.nn.one_hot(state.turn, self.max_turns)
    action_oh = jax.nn.one_hot(action, self.chance_outcomes)
    played_point_cards = jnp.sum(state.point_cards, axis=0)
    prev_point_card = jnp.argmax(jnp.sum(state.point_cards * turn_oh[..., None], axis=0)) 

    outcomes, legals, probs = self.generate_all_chance_outcomes(state)
    outcome = jax.tree.map(lambda x: jnp.sum(x * jnp.reshape(action_oh, (action_oh.shape[0], ) + (1,) * len(x.shape[1:])), axis=0).astype(x.dtype),outcomes)
    legals = jnp.sum(action_oh[..., None] * legals, axis=0)

    num_played = jnp.sum(state.played_cards)
    #The state is terminal, if there is only one more 
    # action left to play, since there is no decision making anymore
    terminal = num_played == self.num_cards - 1

    new_points = state.points + (prev_point_card == action)
    not_played = jnp.argmin(played_point_cards)
    last_card_matched = jnp.sum(legals * jax.nn.one_hot(not_played, self.num_cards))
    legals = jnp.stack([legals, jax.nn.one_hot(0, self.num_cards)], axis=0)
    reward = jnp.where(terminal, new_points + last_card_matched, jnp.zeros_like(new_points + last_card_matched))

    return outcome, terminal, reward[0], legals

  @functools.partial(jax.jit, static_argnums=(0))
  def apply_action_no_chance(self, state: PointCardMatchingStochasticState, action):
    #Dummy first player
    action = action[0]

    turn_oh = jax.nn.one_hot(state.turn, self.max_turns)
    point_card_turn_oh = jax.nn.one_hot(state.turn + 1, self.max_turns)
    action_oh = jax.nn.one_hot(action, self.num_cards)

    new_played_cards = state.played_cards + (action_oh[None, ...] * turn_oh[..., None])
    already_played = jnp.sum(new_played_cards, axis=0)
    new_legals = jnp.ones(self.num_cards) - already_played

    prev_point_card = jnp.argmax(jnp.sum(state.point_cards * turn_oh[..., None], axis=0)) 
    played_point_cards = jnp.sum(state.point_cards, axis=0)
    legal_point_cards  = jnp.arange(self.num_cards) * (1 - played_point_cards)
    #Play the highest available card by the descending order
    descending_point_card = jnp.argmax(legal_point_cards)
    #Match the action on the PREVIOUS point card
    new_points = state.points + (prev_point_card == action)

    point_card_oh = jax.nn.one_hot(descending_point_card, self.num_cards)
    
    next_chance = (state.turn + 1) == self.chance_turn
    # Dont pick a new point card when the next state will be a chance node.
    new_point_cards = jnp.where(next_chance, state.point_cards, state.point_cards + (point_card_oh[None, ...] * point_card_turn_oh[..., None]))

    
    #The state is terminal, if there is only one more 
    # action left to play, since there is no decision making anymore
    terminal = jnp.sum(new_played_cards) == self.num_cards - 1
    terminal = state.terminal + terminal
    terminal = jnp.where(next_chance, jnp.zeros_like(terminal), terminal)
    #checking if we can still match the
    #last point card
    not_played = jnp.argmin(played_point_cards)
    last_card_matched = jnp.sum(new_legals * jax.nn.one_hot(not_played, self.num_cards))
    reward = jnp.where(terminal, new_points + last_card_matched, jnp.zeros_like(new_points + last_card_matched))

    new_legals = jnp.stack([new_legals, jax.nn.one_hot(0, self.num_cards)], axis=0)

    new_state = PointCardMatchingStochasticState(played_cards=new_played_cards,
                                      point_cards = new_point_cards,
                                      points= new_points,
                                      terminal = terminal,
                                      turn = state.turn + 1,
                                      is_chance = next_chance)
    return new_state, terminal, reward[0], new_legals