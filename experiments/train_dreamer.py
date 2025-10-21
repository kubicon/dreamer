import numpy as np
import os


from games.jax_game import JaxGame
from dreamer import DreamerConfig, Dreamer
from dreamer_ma import DreamerMAConfig, DreamerMA
from train_utils import load_model

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
  if args.saved_model_file and os.path.exists(args.saved_model_file):
    model = load_model(args.saved_model_file)
    assert isinstance(model, DreamerMA), f"The loaded model should be an instance of DreamerMA, not {model.__class__}"
  else:
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
    model = DreamerMA(
        config=config,
        game = game,
    )
  model_save_dir = args.model_save_dir
  game_name = game.game_name()
  empty = ""
  game_params = game.params_dict()
  params_str = f'{empty.join(f"_{value}" for key, value in game_params.items())}' 

  if not model_save_dir:
      model_save_dir = f"/trained_networks/dreamer/{game_name}{params_str}/seed{trajectory_seed}/network_seed{network_seed}/"
      model_save_dir = os.getcwd() + model_save_dir
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
  if args.saved_model_file and os.path.exists(args.saved_model_file):
    model = load_model(args.saved_model_file)
    assert isinstance(model, Dreamer), f"The loaded model should be an instance of Dreamer, not {model.__class__}"
  else:
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
    model = Dreamer(
        config=config,
        game = game,
    )
  
  model_save_dir = args.model_save_dir
  game_name = game.game_name()
  empty = ""
  game_params = game.params_dict()
  params_str = f'{empty.join(f"_{value}" for key, value in game_params.items())}'
  if not model_save_dir:
      model_save_dir = f"/trained_networks/dreamer/{game_name}{params_str}/seed{trajectory_seed}/network_seed{network_seed}/"
      model_save_dir = os.getcwd() + model_save_dir

  model.train_world_model(model_save_dir, args.num_steps, args.print_each, args.save_each)

