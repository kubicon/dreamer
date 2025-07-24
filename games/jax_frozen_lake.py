import jax
import jax.numpy as jnp
import chex
import functools
from games.jax_game import JaxGame, GameState


@chex.dataclass(frozen=True)
class FrozenLakeGameState(GameState):
    board: chex.Array  # HxW board state (0=empty, 1=gold, 2=hole)
    player_pos: chex.Array  # [x, y] player position
    timestep: chex.Array  # current timestep
    gold_collected: chex.Array  # total gold collected
    terminal: chex.Array  # whether game is over


class FrozenLake(JaxGame):
  def __init__(self, board: jnp.ndarray, init_position: tuple[int, int] = (0, 0), max_timesteps: int = 50, eps: float = 0.1):
    padded_board = jnp.pad(board, ((1, 1), (1, 1)), mode='constant', constant_values=3)  # HxW, where 0 is empty tile, 1 is gold, 2 is hole, 3 is wall
    self.init_board = jax.nn.one_hot(padded_board, 4)
    self.height, self.width = board.shape
    self.max_gold = jnp.sum(board == 1)
    self.max_timesteps = max_timesteps
    self.eps = eps  # probability of random movement
    
    self.init_player_pos = jnp.array(init_position) + 1 # Move to padding

  def state_tensor_shape(self):
    # board (H*W) + player_pos + timestep  + gold_collected  + terminal 
    return self.height * self.width * 4 + self.height + self.width + self.max_timesteps + self.max_gold + 2
  
  def observation_tensor_shape(self): 
    
    return self.information_state_tensor_shape()
  
  def information_state_tensor_shape(self):
    # 4 neighrbohood + timestep + gold_collected + terminal
    return 4 * 4 + self.max_timesteps + self.max_gold + 2
  
  def public_state_tensor_shape(self): 
    return self.information_state_tensor_shape()
  
  def num_distinct_actions(self):
    return 4  # up, down, left, right
  
  def max_trajectory_length(self):
      return self.max_timesteps
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def initialize_structures(self, key):
    game_state = FrozenLakeGameState(
        board=self.init_board,
        player_pos=self.init_player_pos,
        timestep=0,
        gold_collected=0,
        terminal=False
    )
    # All actions are legal initially
    legal_actions = jnp.ones(4)
    return game_state, legal_actions

  @functools.partial(jax.jit, static_argnums=(0,))
  def get_info(self, game_state: FrozenLakeGameState):
    # Flatten board and concatenate with other state info
    player_y = jax.nn.one_hot(game_state.player_pos[0], self.height)
    player_x = jax.nn.one_hot(game_state.player_pos[1], self.width) 
    timestep = jax.nn.one_hot(game_state.timestep, self.max_timesteps)
    gold_collected = jax.nn.one_hot(game_state.gold_collected, self.max_gold)
    terminal = jax.nn.one_hot(game_state.terminal, 2)
    state_tensor = jnp.concatenate([
        jnp.ravel(game_state.board),
        player_y,
        player_x,
        timestep,
        gold_collected,
        terminal
    ])
    observed_y = jnp.array([game_state.player_pos[0] - 1, game_state.player_pos[0], game_state.player_pos[0] + 1, game_state.player_pos[0]])
    observed_x = jnp.array([game_state.player_pos[1], game_state.player_pos[1] + 1, game_state.player_pos[1], game_state.player_pos[1] - 1])
    observed_tiles = game_state.board[observed_y, observed_x]
    observation_tensor = jnp.concatenate([
        jnp.ravel(observed_tiles), 
        timestep,
        gold_collected,
        terminal
    ])
    return state_tensor, observation_tensor
  
  @functools.partial(jax.jit, static_argnums=(0,))
  def apply_action(self, game_state: FrozenLakeGameState, key, turn, actions):
    # actions is a single integer action (0=up, 1=right, 2=down, 3=left)
    action = actions
    
    # With probability eps, choose random action instead
    key, roll_key, action_key = jax.random.split(key, 3)
    random_roll = jax.random.uniform(roll_key)
    should_randomize = random_roll < self.eps
     
    random_action = jax.random.randint(action_key, (), 0, 4)
    actual_action = jnp.where(should_randomize, random_action, action)
    
    # Define movement directions: up, down, left, right
    directions = jnp.array([
        [-1, 0],  # up
        [0, 1],   # right
        [1, 0],  # down
        [0, -1]    # left
    ])
    
    # Calculate new position
    direction = directions[actual_action]
    
    new_pos = jnp.where(game_state.board[game_state.player_pos[0], game_state.player_pos[1], 3] == 1, game_state.player_pos, game_state.player_pos + direction) # Either you move or you do not
    
    
    # Get the tile at new position
    # tile_value = game_state.board[new_pos[0], new_pos[0]]
    
    # Check if game ends (hole or max timesteps)
    hit_hole = game_state.board[new_pos[0], new_pos[1], 2] == 1
    max_time_reached = game_state.timestep >= self.max_timesteps - 1
    collected_all_gold = game_state.gold_collected >= self.max_gold
    terminal = hit_hole | max_time_reached | collected_all_gold
    
    # Update gold collected if we're on a gold tile
    gold_gained = jnp.where(game_state.board[new_pos[0], new_pos[1], 1] == 1, 1, 0)
    new_gold_collected = game_state.gold_collected + gold_gained
    # Update board - remove gold from collected tile
    new_board = jnp.where(game_state.board[new_pos[0], new_pos[1], 1] == 1, game_state.board.at[new_pos[0], new_pos[1], 1].set(0).at[new_pos[0], new_pos[1], 0].set(1), game_state.board)
    
    # Create new game state
    new_game_state = FrozenLakeGameState(
        board=new_board,
        player_pos=new_pos,
        timestep=game_state.timestep + 1,
        gold_collected=new_gold_collected,
        terminal=terminal
    )
    
    reward = jnp.where(hit_hole, -10, gold_gained)
    
    # Legal actions - all actions are always legal (clipping handles boundaries)
    legal_actions = jnp.ones(4)
    
    return new_game_state, key, terminal, reward, legal_actions 