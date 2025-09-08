import numpy as np
import os


from games.jax_game import JaxGame
from dreamer import DreamerConfig, Dreamer
from dreamer_ma import DreamerMAConfig, DreamerMA
from argparse import ArgumentParser

def create_dreamer_parser(multi_agent: bool = True) ->ArgumentParser:
  """Initializes parser with all the dreamer required parameters.
  Use the multi-agent bool to control which version of Dreamer to 
  initialize for."""
  parser = ArgumentParser()
  ##Model parameters 
  parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
  parser.add_argument("--encoded_categories", type=int, default=32, help="Number of categorical distributions in the latent state.")
  parser.add_argument("--encoded_classes", type=int, default=32, help="Number of classes for each categorical distribution in the latent state.")
  parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for the optimizer")
  parser.add_argument("--network_seed", type=int, default=-1, help="Random seed for network initialization")
  parser.add_argument("--trajectory_seed", type=int, default=-1, help="Random seed for trajectory generation")
  parser.add_argument("--hidden_state_size", type=int, default =256, help="Size of the RNN hidden state")
  parser.add_argument("--bin_range", type=int, default=20, help="Number of the exponentially spaced bins for certain predictions such as reward in one direction, bins will be spaced out as symexp([-bin_range, ..., bin_range])")

  ## Loss function coefficients
  parser.add_argument("--beta_prediction", type=float, default=1, help="The beta coefficient for the prediction loss")
  parser.add_argument("--beta_dynamics", type=float, default=1, help="The beta coefficient for the dynamics loss")
  parser.add_argument("--beta_representation", type=float, default=0.1, help="The beta coefficient for the representation loss")
  parser.add_argument("--free_bits_threshold", type=float, default=1, help="Clipping threshold for the dynamics and representation losses in free bits.")

  ##Network layer parameters
  parser.add_argument("--encoder_hidden_size", type=int, default=256, help="Size of the hidden layer in the encoder network")
  parser.add_argument("--dynamics_hidden_size", type=int, default=256, help="Size of the hidden layer in the dynamics network")
  parser.add_argument("--decoder_hidden_size", type=int, default=256, help="Size of the hidden layer in the decoder network")
  parser.add_argument("--predictor_hidden_size", type=int, default=256, help="Size of the hidden layer in the predictor network")
  parser.add_argument("--encoder_hidden_layers", type=int, default=1, help="Number of hidden layers in the encoder network")
  parser.add_argument("--dynamics_hidden_layers", type=int, default=1, help="Number of hidden layers in the dynamics network")
  parser.add_argument("--decoder_hidden_layers", type=int, default=1, help="Number of hidden layers in the decoder network")
  parser.add_argument("--predictor_hidden_layers", type=int, default=1, help="Number of hidden layers in the predictor network")
  if multi_agent:
    parser.add_argument("--legal_hidden_size", type=int, default=256, help="Size of the hidden layer in the legal actions network")
    parser.add_argument("--legal_hidden_layers", type=int, default=1, help="Number of hidden layers in the legal actions network")
  # Training parameters
  parser.add_argument("--num_steps", type=int, default=1001, help="Number of training steps")
  parser.add_argument("--save_each", type=int, default=100, help="Save model every N steps")
  parser.add_argument("--print_each", type=int, default=100, help="Print loss every N steps")
  parser.add_argument("--model_save_dir", type=str, default="", help="Directory to save the trained model")
  return parser

def train_model_ma(args, game:JaxGame):
  """Trains the multi-agent version of Dreamer on a given
  two player zero sum game. Args must contain all necessary parameters
  as created by create_dreamer_parser(multi_agent=True)."""
  assert game.num_players() == 2, f"This version of dreamer needs a 2 player zero-sum game, instead got {game.num_players()} player game."
  network_seed = args.network_seed
  trajectory_seed = args.trajectory_seed
  if network_seed == -1:
    network_seed = np.random.randint(0, 2**32 - 1)
  if trajectory_seed == -1:
    trajectory_seed = np.random.randint(0, 2**32 - 1)
  print(f"Using network seed: {network_seed}, trajectory seed: {trajectory_seed}")
  config = DreamerMAConfig(
      batch_size=args.batch_size,
      seed=args.trajectory_seed,


      #Weights of the individual loss terms of the world model
      beta_prediction = args.beta_prediction,
      beta_dynamics = args.beta_dynamics,
      beta_representation = args.beta_representation,
      
      hidden_state_size = args.hidden_state_size,
      encoded_classes = args.encoded_classes,
      encoded_categories = args.encoded_categories,
      bin_range = args.bin_range,

      # Ordered as (hidden_layer_features, num_hidden_layers)
      encoder_network_details = (args.encoder_hidden_size, args.encoder_hidden_layers),
      decoder_network_details = (args.decoder_hidden_size, args.decoder_hidden_layers),
      dynamics_network_details = (args.dynamics_hidden_size, args.dynamics_hidden_layers),
      predictor_network_details = (args.predictor_hidden_size, args.predictor_hidden_layers),
      legal_actions_network_details = (args.legal_hidden_size, args.legal_hidden_layers),

      learning_rate = args.learning_rate,
      rng_seed = args.network_seed
  )
  model_save_dir = args.model_save_dir
  game_name = game.game_name()
  empty = ""
  game_params = game.params_dict()
  params_str = f'{empty.join(f"_{value}" for key, value in game_params.items())}' 

  if not model_save_dir:
      model_save_dir = f"/trained_networks/dreamer/{game_name}{params_str}/seed{trajectory_seed}/network_seed{network_seed}/"
      model_save_dir = os.getcwd() + model_save_dir
  model = DreamerMA(
      config=config,
      game = game,
  )
  model.train_world_model(model_save_dir, args.num_steps, args.print_each, args.save_each)

def train_model(args, game:JaxGame):
  """Train a single agent Dreamer model on a given
  single agent environment. Args must contain all necessary parameters
  as created by create_dreamer_parser(multi_agent=False) """
  network_seed = args.network_seed
  trajectory_seed = args.trajectory_seed
  if network_seed == -1:
    network_seed = np.random.randint(0, 2**32 - 1)
  if trajectory_seed == -1:
    trajectory_seed = np.random.randint(0, 2**32 - 1)
  print(f"Using network seed: {network_seed}, trajectory seed: {trajectory_seed}")
  config = DreamerConfig(
      batch_size=args.batch_size,
      seed=args.trajectory_seed,


      #Weights of the individual loss terms of the world model
      beta_prediction = args.beta_prediction,
      beta_dynamics = args.beta_dynamics,
      beta_representation = args.beta_representation,
      free_bits_clip_threshold = args.free_bits_threshold,
      
      hidden_state_size = args.hidden_state_size,
      encoded_classes = args.encoded_classes,
      encoded_categories = args.encoded_categories,
      bin_range = args.bin_range,

      # Ordered as (hidden_layer_features, num_hidden_layers)
      encoder_network_details = (args.encoder_hidden_size, args.encoder_hidden_layers),
      decoder_network_details = (args.decoder_hidden_size, args.decoder_hidden_layers),
      dynamics_network_details = (args.dynamics_hidden_size, args.dynamics_hidden_layers),
      predictor_network_details = (args.predictor_hidden_size, args.predictor_hidden_layers),

      learning_rate = args.learning_rate,
      rng_seed = args.network_seed
  )
  
  model_save_dir = args.model_save_dir
  game_name = game.game_name()
  empty = ""
  game_params = game.params_dict()
  params_str = f'{empty.join(f"_{value}" for key, value in game_params.items())}'
  if not model_save_dir:
      model_save_dir = f"/trained_networks/dreamer/{game_name}{params_str}/seed{trajectory_seed}/network_seed{network_seed}/"
      model_save_dir = os.getcwd() + model_save_dir
  model = Dreamer(
      config=config,
      game = game,
  )
  model.train_world_model(model_save_dir, args.num_steps, args.print_each, args.save_each)