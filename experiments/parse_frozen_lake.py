from games.jax_frozen_lake import FrozenLake
import numpy as np
import os

def parse_game(filepath:str) -> FrozenLake:
  """Parse the frozen lake game config and return the game.
  The config is of form:
  <height> <width>
  <board specification as a string of 0s, 1s and 2s, where 0 is empty, 1 is gold, 2 is hole. One row per line>
  <init_player_pos_x> <init_player_pos_y>
  <max_timesteps>
  <epsilon>"""
  if not filepath.startswith("/"):
    filepath = os.getcwd() + "/" + filepath
  if not os.path.exists(filepath):
    raise FileNotFoundError(f"Game config file {filepath} does not exist.")
  with open(filepath, 'r') as f:
    lines = f.readlines()
  height, width = map(int, lines[0].strip().split())
  board = []
  for line in lines[1: 1+height]:
    board.append([])
    for char in line.strip().split():
      if char not in '012':
        raise ValueError(f"Invalid character {char} in board specification. Expected 0, 1 or 2.")
      board[-1].append(int(char))
  if len(board) != height or any(len(row) != width for row in board):
    raise ValueError(f"Board dimensions do not match specified height {height} and width {width}.")
  board = np.asarray(board, dtype=np.int32)
  init_player_pos_x, init_player_pos_y = map(int, lines[1+height].strip().split())
  max_timesteps = int(lines[2+height].strip())
  eps = float(lines[3+height].strip())
  
  return FrozenLake(
      board=board,
      init_position=(init_player_pos_x, init_player_pos_y),
      max_timesteps=max_timesteps,
      eps=eps
  )