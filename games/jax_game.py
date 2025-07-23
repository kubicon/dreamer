from abc import ABC, abstractmethod
    
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
  
  #returns game_state, key, legal_actions
  @abstractmethod
  def initialize_structures(self, key):
    pass

  #returns state_tensor, p1_iset_tensor, p2_iset_tensor, public_state_tensor
  @abstractmethod
  def get_info(self, game_state):
    pass

  #returns new_game_state, key, terminal, rewards, new_legals
  @abstractmethod
  def apply_action(self, game_state, key, turn, actions):
    pass