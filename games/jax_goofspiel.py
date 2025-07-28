import jax
import chex
import jax.numpy as jnp

import functools
from games.jax_game import JaxGame, GameState


@chex.dataclass(frozen=True)
class GoofspielGameState(GameState):
    point_cards: chex.Array
    played_cards: chex.Array
    p1_points: chex.Array


class JaxGoofspiel(JaxGame):
  def __init__(self, cards, points_order="descending", turns=-1, reward_type: str = "clip") -> None:
    self.cards = cards
    self.max_turns = turns
    if turns <= 0:
      self.max_turns = cards
    self.points_order = points_order 
    self.reward_type = 0 if reward_type == "clip" else 1
    
  
  def game_name(self):
    return "goofspiel"
  
  def params_dict(self):
    #Possible to also add other relevant information
    # for now just the cards will do to organize into subdirs
    return {self.cards}
  
  def num_players(self):
    return 2

  def state_tensor_shape(self):
    return self.max_turns * self.cards + self.max_turns * 2 + self.max_turns * self.cards * 3
   
  def information_state_tensor_shape(self):
    return self.max_turns * self.cards + self.max_turns * 2 + self.max_turns * self.cards * 2 + 2
  
  def public_state_tensor_shape(self):
    return self.max_turns * self.cards + self.max_turns * 2 + self.max_turns * self.cards
  
  def num_distinct_actions(self):
    return self.cards
  
  def max_trajectory_length(self):
    return self.max_turns - 1
  
  
  @functools.partial(jax.jit, static_argnums=(0))
  def initialize_structures(self, key):
    if self.points_order == "descending":
      point_cards = jnp.arange(self.cards, self.cards - self.max_turns, -1)
    if self.points_order == "ascending":
      point_cards = jnp.arange(1, 1 + self.cards - self.max_turns)
    played_cards = jnp.zeros((2, self.max_turns, self.cards))
    p1_points = jnp.zeros(self.max_turns)
    game_state= GoofspielGameState(point_cards=point_cards, played_cards=played_cards, p1_points=p1_points)
    return game_state, jnp.ones((2, self.cards))

  
  # State Tensor -> Point card [Turn, Card], Winner [Turn, Player], Tie Cards [Turn, Card], Played Cards [Player, Turn, Card], 
  # Iset tensor -> Observing Player, Point card [Turn, Card], Winner [Turn, Player], Tie Cards [Turn, Card], Played Cards [Turn, Card],
  # Public tensor -> Point card [Turn, Card], Winner [Turn, Player], Tie Cards [Turn, Card]
  @functools.partial(jax.jit, static_argnums=(0,))
  def get_info(self, game_state:GoofspielGameState):
    played_turns_mask = jnp.sum(game_state.played_cards[0], -1)
    # To set the first to 
    played_turns_mask = jnp.roll(played_turns_mask, 1, axis=0) + jax.nn.one_hot(0, self.max_turns)
    # Every card that is played have value >= 1, non-played has 0. So we just subtract 1 to make sure everything works with one-hot (-1 is all zeros)
    point_cards_masked = game_state.point_cards * played_turns_mask - 1  
    oh_point_cards = jax.nn.one_hot(point_cards_masked, self.cards)
    
    # Tie -1, P1 win 0, P2 win 1
    p2_winned = jnp.where(game_state.p1_points < 0, 1, 0) - (game_state.p1_points == 0)
    winner = jax.nn.one_hot(p2_winned, 2)
    
    tie_cards = jnp.expand_dims(((game_state.p1_points == 0) * played_turns_mask), -1) * game_state.played_cards[0]
    
    public_state_tensor = jnp.concatenate([jnp.ravel(oh_point_cards), jnp.ravel(winner), jnp.ravel(tie_cards)], axis=0)
    
    p1_player = jax.nn.one_hot(0, 2)
    
    p1_iset_tensor = jnp.concatenate([p1_player, public_state_tensor, jnp.ravel(game_state.played_cards[0])], axis=0)
    p2_iset_tensor = jnp.concatenate([1 - p1_player, public_state_tensor, jnp.ravel(game_state.played_cards[1])], axis=0)
    
    state_tensor = jnp.concatenate([public_state_tensor, jnp.ravel(game_state.played_cards)], axis=0)
    
    
    return state_tensor, p1_iset_tensor, p2_iset_tensor, public_state_tensor
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def apply_action(self, game_state:GoofspielGameState, key, turn, actions):
    # This is not working
    # turn = jnp.argmax(jnp.arange(self.max_turns) * jnp.sum(played_cards[0], -1))
    oh_actions = jax.nn.one_hot(actions, self.cards) 
    oh_turn = jax.nn.one_hot(turn, self.max_turns)
    
    winner = jnp.argmax(actions, axis=-1)
    loser = jnp.argmin(actions, axis=-1)
    tie = winner == loser
    
    # Point cards are from 0 to N-1, but points should be from 1 to N
    point = game_state.point_cards[..., turn] * oh_turn
    
    this_turn_played = oh_actions[..., None, :] * oh_turn[None, :, None]
    
    played_cards = game_state.played_cards + this_turn_played
    
    p1_points = jnp.where(tie, game_state.p1_points, jnp.where(winner == 0, game_state.p1_points + point, game_state.p1_points - point))
    
    legal_actions = 1 - jnp.sum(played_cards, 1)
    
    next_action = jnp.argmax(legal_actions, -1)
    next_winner = jnp.argmax(next_action)
    next_loser = jnp.argmin(next_action)
    next_tie = next_winner == next_loser
    next_point = game_state.point_cards[..., turn+1] * jax.nn.one_hot(turn+1, self.max_turns)
    
    p1_points = jnp.where(turn != self.cards - 2, p1_points, jnp.where(next_tie, p1_points, jnp.where(next_winner == 0, p1_points + next_point, p1_points - next_point)))
    
    rewards = jnp.where(self.reward_type == 0, jnp.clip(jnp.sum(p1_points), -1, 1), jnp.sum(p1_points))
    
    rewards = jnp.where(turn != self.cards - 2, 0, rewards)
    terminal = turn >= self.cards - 2
    
    # rewards = jnp.sum(p1_points)
    # if turn == self.cards-1:
    #   actions = jnp.argmax(legal_actions, -1)
    #   return self.apply_action(point_cards, played_cards, p1_points, turn+1, actions)
    game_state= GoofspielGameState(point_cards=game_state.point_cards, played_cards=played_cards, p1_points=p1_points)

    
    return game_state, terminal, rewards, legal_actions
    
