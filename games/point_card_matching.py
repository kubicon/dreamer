import jax
import jax.numpy as jnp

import chex
import functools

from games.jax_game import JaxGame, GameState

"""A single player goofspiel inspired game.
To mantain consistency with other JAX games,
it actually is treated as a simultaneous move
game, but the second player always has only
one legal action.
Basically just a match the point card game.
Just a sanity check for the VQ-VAE algorithm"""

@chex.dataclass(frozen=True)
class PointCardMatchingState(GameState):
  #history of one hot played cards
  played_cards: chex.Array
  #history of one hot point cards
  points: chex.Array
  point_cards: chex.Array
  terminal: chex.Array


class PointCardMatching(JaxGame):
  def __init__(self, num_cards):
    self.num_cards = num_cards
    self.max_turns = num_cards
    #get a reward 1 whenever a card is matched
    self.max_points = num_cards

  

  def initialize_structures(self, key):
    init_played_cards = jnp.zeros((self.max_turns, self.num_cards))
    init_points = jnp.zeros(1)
    init_point_cards = jnp.concatenate([jax.nn.one_hot(self.num_cards - 1, self.num_cards)[None, ...], jnp.zeros((self.max_turns - 1, self.num_cards))], axis=0)
    init_state = PointCardMatchingState(played_cards = init_played_cards,
                                        points = init_points,
                                        point_cards = init_point_cards,
                                        terminal = jnp.array(False))
    init_legals = jnp.ones(self.num_cards)
    return init_state, init_legals
  
  @functools.partial(jax.jit, static_argnums=(0))
  def get_info(self, state: PointCardMatchingState):
    #starting at 0 points hence the + 1
    points_oh = jax.nn.one_hot(state.points, self.max_points + 1)
    state_tensor = jnp.concatenate([state.played_cards.ravel(), state.point_cards.ravel(), points_oh.ravel()])
    #Return just a state tensor as both state tensor and observation
    # this is a perfect information game
    return state_tensor, state_tensor
  
  def num_distinct_actions(self):
    return self.num_cards
  
  def max_trajectory_length(self):
    return self.num_cards - 1
  
  def game_name(self):
    return "point_card_matching"
  
  def params_dict(self):
    return {"num_cards": self.num_cards}

  def num_players(self):
    return 1
  
  def observation_tensor_shape(self):
    return self.information_state_tensor_shape()
  
  def public_state_tensor_shape(self):
    return self.information_state_tensor_shape()
  
  def state_tensor_shape(self):
    return 2 * (self.max_turns * self.num_cards) + self.max_points + 1
  
  def information_state_tensor_shape(self):
    return 2 * (self.max_turns * self.num_cards) + self.max_points + 1
  

  @functools.partial(jax.jit, static_argnums=(0))
  def apply_action(self, state: PointCardMatchingState, key, turn, action):
    turn_oh = jax.nn.one_hot(turn, self.max_turns)
    point_card_turn_oh = jax.nn.one_hot(turn + 1, self.max_turns)
    action_oh = jax.nn.one_hot(action, self.num_cards)

    new_played_cards = state.played_cards + (action_oh[None, ...] * turn_oh[..., None])
    already_played = jnp.sum(new_played_cards, axis=0)
    new_legals = jnp.ones(self.num_cards) - already_played

    #descending order
    point_card = self.max_turns - turn - 2
    #Match the action on the PREVIOUS point card
    new_points = state.points + (point_card + 1 == action)

    point_card_oh = jax.nn.one_hot(point_card, self.num_cards)
    new_point_cards = state.point_cards + (point_card_oh[None, ...] * point_card_turn_oh[..., None])

    terminal = turn == (self.max_turns - 2)
    terminal = state.terminal + terminal
    #new_point_cards = jnp.where(turn ==(self.max_turns - 1), state.point_cards, new_point_cards)
    #checking if we can still match the
    #last point card, which will be the first card
    # because of descending order.
    last_card_matched = jnp.sum(new_legals * jax.nn.one_hot(0, self.num_cards))
    reward = jnp.where(terminal, new_points + last_card_matched, jnp.zeros_like(new_points + last_card_matched))

    new_state = PointCardMatchingState(played_cards=new_played_cards,
                                       point_cards = new_point_cards,
                                       points= new_points,
                                       terminal = terminal)
    return new_state, terminal, reward[0], new_legals
  

class PointCardMatchingStochastic(JaxGame):
  """A point card matching variant with a single
  chance node at the end. Cards are revealed in a descending order,
  except the chance node level, when the revealed card is chosen at random
  and then the game continues in descending order."""
  def __init__(self, num_cards: int, chance_turn_before_terminal: int = 1):
    """chance_turn_before_terminal specifies how many turns before a terminal
    turn willl the chance node happen. For example when chance_turn_before_terminal == 1,
    then the chance node happens on turn num_cards - 3, when there is a choice
    between only point cards 1 or 2. The current implementation does not allow a chance node for the
    very first point card.
    Assure that num_cards >= 3"""
    self.num_cards = num_cards
    self.max_turns = num_cards
    #get a reward 1 whenever a card is matched
    self.max_points = num_cards
    self.chance_turn = self.num_cards - 2 - chance_turn_before_terminal
    assert self.chance_turn >= 0, f"Invalid config with {self.num_cards} and {chance_turn_before_terminal}, the chance node is set to happen at invalid turn {self.chance_turn}."

    self.chance_outcomes = self.num_cards - 1 - self.chance_turn

  def initialize_structures(self, key):
    init_played_cards = jnp.zeros((self.max_turns, self.num_cards))
    init_points = jnp.zeros(1)
    init_point_cards = jnp.concatenate([jax.nn.one_hot(self.num_cards - 1, self.num_cards)[None, ...], jnp.zeros((self.max_turns - 1, self.num_cards))], axis=0)
    init_state = PointCardMatchingState(played_cards = init_played_cards,
                                        points = init_points,
                                        point_cards = init_point_cards,
                                        terminal = jnp.array(False))
    init_legals = jnp.ones(self.num_cards)
    return init_state, init_legals
  
  @functools.partial(jax.jit, static_argnums=(0))
  def get_info(self, state: PointCardMatchingState):
    #starting at 0 points hence the + 1
    points_oh = jax.nn.one_hot(state.points, self.max_points + 1)
    state_tensor = jnp.concatenate([state.played_cards.ravel(), state.point_cards.ravel(), points_oh.ravel()])
    #Return just a state tensor as both state tensor and observation
    # this is a perfect information game
    return state_tensor, state_tensor
  
  def num_distinct_actions(self):
    return self.num_cards
  
  def max_trajectory_length(self):
    return self.num_cards - 1
  
  def game_name(self):
    return "point_card_matching_stochastic"
  
  def params_dict(self):
    return {"num_cards": self.num_cards, "chance_turn": self.chance_turn}

  def num_players(self):
    return 1
  
  def observation_tensor_shape(self):
    return self.information_state_tensor_shape()
  
  def public_state_tensor_shape(self):
    return self.information_state_tensor_shape()
  
  def state_tensor_shape(self):
    return 2 * (self.max_turns * self.num_cards) + self.max_points + 1
  
  def information_state_tensor_shape(self):
    return self.state_tensor_shape()
  
  def generate_all_chance_outcomes(self, after_chance_state: PointCardMatchingState):
    """Generate all chance outcomes (2 in this case) for a state that already happened
    after chance node, which happens when acting on self.chance_turn
    . There are num_cards - 1 - self.chance_turn choices for the chance node.
    These outcomes will be returned as a list, sorted in ascending order by the point card."""
    after_chance_turn = self.chance_turn + 1
    after_chance_turn_oh = jax.nn.one_hot(after_chance_turn, self.max_turns)
    chosen_card_oh = jnp.sum(after_chance_state.point_cards * after_chance_turn_oh[..., None], axis=0)
    choice_blanked_point_cards = after_chance_state.point_cards - (chosen_card_oh[None, ...] * after_chance_turn_oh[..., None])

    card_choices = [i for i in range(self.chance_outcomes)]
    outcome_states = []
    for c in card_choices:
      outcome_card_oh = jax.nn.one_hot(c, self.num_cards)

      outcome_point_cards = choice_blanked_point_cards +  (outcome_card_oh[None, ...] * after_chance_turn_oh[..., None])
      outcome_state = PointCardMatchingState(played_cards = after_chance_state.played_cards,
                                                points = after_chance_state.points,
                                                terminal = after_chance_state.terminal,
                                                point_cards = outcome_point_cards)
      outcome_states.append(outcome_state)
    
    return outcome_states

    

  
  @functools.partial(jax.jit, static_argnums=(0))
  def apply_action(self, state: PointCardMatchingState, key, turn, action):
    turn_oh = jax.nn.one_hot(turn, self.max_turns)
    point_card_turn_oh = jax.nn.one_hot(turn + 1, self.max_turns)
    action_oh = jax.nn.one_hot(action, self.num_cards)

    new_played_cards = state.played_cards + (action_oh[None, ...] * turn_oh[..., None])
    already_played = jnp.sum(new_played_cards, axis=0)
    new_legals = jnp.ones(self.num_cards) - already_played

    prev_point_card = jnp.argmax(jnp.sum(state.point_cards * turn_oh[..., None], axis=0)) 
    #descending order, just need to check for the case after
    #chance node to properly match
    is_chance = turn == self.chance_turn
    played_point_cards = jnp.sum(state.point_cards, axis=0)
    legal_point_cards  = jnp.arange(self.num_cards) * (1 - played_point_cards)
    #Play the highest available card by the descending order
    descending_point_card = jnp.argmax(legal_point_cards)
    #Chance node chooses uniformly from the available point cards
    chance_point_card = jax.random.choice(key, legal_point_cards)
    #If only two cards left, choose at random
    point_card = jnp.where(is_chance, chance_point_card, descending_point_card)
    #Match the action on the PREVIOUS point card
    new_points = state.points + (prev_point_card == action)

    point_card_oh = jax.nn.one_hot(point_card, self.num_cards)
    new_point_cards = state.point_cards + (point_card_oh[None, ...] * point_card_turn_oh[..., None])

    terminal = turn == (self.max_turns - 2)
    terminal = state.terminal + terminal
    #checking if we can still match the
    #last point card
    not_played = jnp.argmin(played_point_cards)
    last_card_matched = jnp.sum(new_legals * jax.nn.one_hot(not_played, self.num_cards))
    reward = jnp.where(terminal, new_points + last_card_matched, jnp.zeros_like(new_points + last_card_matched))

    new_state = PointCardMatchingState(played_cards=new_played_cards,
                                      point_cards = new_point_cards,
                                      points= new_points,
                                      terminal = terminal)
    return new_state, terminal, reward[0], new_legals