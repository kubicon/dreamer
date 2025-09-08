from abc import ABC, abstractmethod
import chex
import jax
import jax.numpy as jnp
    
class GameState(ABC):
  pass


#Abstract parent class for any jax game
class JaxGame(ABC):
  
  def new_initial_state(self, key):
    init_state, legals = self.initialize_structures(key)
    return self.get_info(init_state)
  
  # Do not use this method if you plan to use any other tensor
  def state_tensor(self, game_state):
    return self.get_info(game_state)[0]
  
  
  # Do not use this method if you plan to use any other tensor
  def information_state_tensor(self, game_state, player: int):
    if player == 0:
      return self.get_info(game_state)[1]
    else:
      return self.get_info(game_state)[2]
  
  
  # Do not use this method if you plan to use any other tensor
  def public_state_tensor(self, game_state):
    return self.get_info(game_state)[3]
  
  def __str__(self):
    params_dict = self.params_dict()
    empty = ""
    params_str = f'{empty.join(f"Param: {key} value: {value}" for key, value in params_dict.items())}' 
    return str(self.game_name() + " " + params_str)
  
  def __repr__(self):
    return self.__str__()


  def is_chance(self, game_state: GameState)-> chex.Array:
    """Return whether the given game state is a chance node."""
    return jnp.array(False)
  
  def get_outcomes_and_probs(self, game_state:GameState) ->tuple[GameState, chex.Array, chex.Array]:
    """Gets the chance outcomes and probabilities of them for a game state at given turn.
    Returns invalid output if the state is not a chance node."""
    expanded_state = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), game_state)
    return expanded_state, jnp.ones((1, self.num_players(), self.num_distinct_actions())), jnp.zeros(1)

  def get_outcomes_and_probs_with_check(self, game_state:GameState) ->tuple[GameState, chex.Array, chex.Array]:
    """Gets all chance outcome states and their respective probabilities
    for a chance node. This version checks whether the state is 
    actually a chance node first, and will not allow being called on none chance nodes
    (where the behavior is undefined). Uses standard Python assert and as such will not work under vmap"""
    assert self.is_chance(game_state), "Tried getting the chance outcomes and their probabilities of a non chance node!"
    return self.get_outcomes_and_probs(game_state)
  
  def max_chance_outcomes(self):
    """Returns the maximum chance outcomes of any chance
    node in the game."""
    return 1
  
  def depth_chance_outcomes(self, depth:int):
    """Return how many of the first indices of probs 
    gotten by get_outcomes_and_probs can contain valid chance outcomes 
    at the given tree depth"""
    return 1
  
  def max_trajectory_lenght_no_chance(self):
    """Returns the maximum lenght of the game trajectory
    excluding chance nodes. Useful when we want to filter out
    the chance nodes under jit."""
    return self.max_trajectory_length()

  @abstractmethod
  def game_name(self) ->str:
    pass
  
  @abstractmethod
  def params_dict(self)->dict:
    pass

  @abstractmethod
  def num_players(self)->int:
    pass

  @abstractmethod
  def state_tensor_shape(self):
    pass
  
  @abstractmethod
  def observation_tensor_shape(self):
    pass
  
  @abstractmethod
  def information_state_tensor_shape(self):
    pass
    
  @abstractmethod
  def public_state_tensor_shape(self):
    pass
  
  @abstractmethod
  def num_distinct_actions(self):
    pass
  
  @abstractmethod
  def max_trajectory_length(self):
    pass
  
  #returns game_state, legal_actions
  @abstractmethod
  def initialize_structures(self):
    pass

  #returns state_tensor, p1_iset_tensor, p2_iset_tensor, public_state_tensor
  @abstractmethod
  def get_info(self, game_state):
    pass

  #returns new_game_state, terminal, rewards, new_legals
  @abstractmethod
  def apply_action(self, game_state, actions):
    pass