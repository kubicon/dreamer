import jax
import chex
import jax.numpy as jnp

import functools
from games.jax_game import JaxGame, GameState


@chex.dataclass(frozen=True)
class RPSSTate(GameState):
  p1_points: chex.Array
  terminal: chex.Array


class JaxRPS(JaxGame):
  def __init__(self) -> None:
    self.max_turns = 2 # Including the terminal state
    self.moves = {"r" : 0, "p": 1, "s": 2}
    # p1 wins: pr: 1, sp, 1, rs -2
    # p1 loses : rp: -1, ps: -1, sr: 2
    self.actions = 3
    
  
  def game_name(self):
    return "rps"
  
  def params_dict(self):
    return {}
  
  def num_players(self):
    return 2

  def state_tensor_shape(self):
    return self.information_state_tensor_shape()
   
  def information_state_tensor_shape(self):
    return 2 + 1 # the 2 is made to differentiate the winner. 0 for player 1, 1 for player 2, -1 for tie,the 1 is just one bit for the terminal flag
    # this is just the observation tensor of the current state, but because the game is only a single turn, it corresponds
    # to the perfect recall iset 

  def observation_tensor_shape(self):
    return self.information_state_tensor_shape()
  
  def public_state_tensor_shape(self):
    return self.information_state_tensor_shape()
  
  def num_distinct_actions(self):
    return self.actions
  
  def max_trajectory_length(self):
    return self.max_turns - 1
  
  
  @functools.partial(jax.jit, static_argnums=(0))
  def initialize_structures(self):
    game_state = RPSSTate(terminal = jnp.array(False, dtype=bool),
                          p1_points = jnp.array(0))
    return game_state, jnp.ones((2, self.actions))

  
  # This just returns an observation tensor for the current 
  # state, however, since the game is just one turn 
  # this is actually a perfect recall representation, since the no information
  # of the initial state will not play a role
  @functools.partial(jax.jit, static_argnums=(0,))
  def get_info(self, game_state:RPSSTate):
    # Taking advantage of -1 being encoded as all zeros
    terminal_oh = jax.nn.one_hot(game_state.terminal - 1, 1) 
    p1_points_oh = jax.nn.one_hot(game_state.p1_points, 2)
    p1_iset_tensor = jnp.concatenate([terminal_oh.ravel(), p1_points_oh.ravel()], axis=0)
    #We use just p1 points for state_tensor, public state tensor and p2_iset_tensor as well, since they uniquely define p2 points as well
    return p1_iset_tensor, p1_iset_tensor, p1_iset_tensor, p1_iset_tensor
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def apply_action(self, game_state:RPSSTate, actions):
    #A single action. The state after applying will always be terminal
    terminal = jnp.array(True, dtype=bool)
    action_difference = actions[0]- actions[1]
    # R: 0, P: 1, S: 2, with these, p1 wins at values -2, 1, loses at -1, 2 and its a tie at 0
    # sign(x) * (3 - 2 * abs(x)) produces these values, assuming sign(0) = 0, which it is in jnp
    p1_points = jnp.sign(action_difference) * (3 - (2 * jnp.abs(action_difference)))

    legal_actions = jnp.ones((2, self.actions))
    new_game_state = RPSSTate(terminal = terminal,
                              p1_points = p1_points)
    
    return new_game_state, terminal, p1_points, legal_actions
    
