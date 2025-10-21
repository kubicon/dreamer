import numpy as np
import os

from train_utils import load_model
from dreamer_ma import DreamerMA
from rnad_dreamer import RNaDConfig, RNaDDreamer


def train_rnad(args):
  #profiler = Profiler()
  network_seed = args.network_seed
  trajectory_seed = args.trajectory_seed
  if network_seed == -1:
    network_seed = np.random.randint(0, 2**32 - 1)
  if trajectory_seed == -1:
    trajectory_seed = np.random.randint(0, 2**32 - 1)
  print(f"Using network seed: {network_seed}, trajectory seed: {trajectory_seed}")
  if args.saved_model_file and os.path.exists(args.saved_model_file):
    model = load_model(args.saved_model_file)
    assert isinstance(model, RNaDDreamer), f"The loaded model should be an instance of RNaDDreamer, not {model.__class__}"
  else:
    config = RNaDConfig(
        batch_size=args.batch_size,
        seed=args.trajectory_seed,
        use_learned_model = args.use_learned_model,

        beta_imagination = args.beta_imagination,
        beta_real = args.beta_real,

        eta=args.eta,
        vtrace_eta = args.vtrace_eta,
        sampling_epsilon=args.sampling_epsilon,

        #Dreamer extraction parameters
        state_sample_threshold=args.state_sample_threshold,
        terminal_threshold = args.terminal_threshold,
        legal_threshold = args.legal_threshold,

        # Entropy schedule parameters
        entropy_schedule_size = args.entropy_schedule_size,
        entropy_schedule_repeats = args.entropy_schedule_repeats,
        
        #V-Trace parameters
        rho_vtrace = args.rho_vtrace,
        c_vtrace = args.c_vtrace,
        gamma_vtrace = args.gamma_vtrace,
        lambda_vtrace = args.lambda_vtrace,

        # NeuRD parameters
        neurd_clip = args.neurd_clip,
        neurd_threshold = args.neurd_threshold,

        # Ordered as (hidden_layer_features, num_hidden_layers)
        rnad_network_details = (args.network_hidden_size, args.network_hidden_layers),

        learning_rate = args.learning_rate,
        network_seed = args.network_seed
    )
    saved_model_dir = args.dreamer_dir
    if not saved_model_dir.startswith("/"):
      saved_model_dir = os.getcwd() + "/" + saved_model_dir
    saved_model_dir = saved_model_dir + f"/step_{args.model_restore_step}.pkl"
    if not os.path.exists(saved_model_dir):
      raise FileNotFoundError(f"The given Dreamer path {saved_model_dir} does not exist!")
    world_model = load_model(saved_model_dir)
    assert isinstance(world_model, DreamerMA), f"The world model is expected to be an instance of DreamerMA not {world_model.__class__}"
    model = RNaDDreamer(
        config=config,
        dreamer_model= world_model
    )
  game = model.world_model.game
  model_save_dir = args.model_save_dir
  game_name = game.game_name()
  empty = ""
  game_params = game.params_dict()
  params_str = f'{empty.join(f"_{value}" for key, value in game_params.items())}' 
  
  if not model_save_dir:
      model_save_dir = f"/trained_networks/rnad/{game_name}{params_str}/seed{trajectory_seed}/network_seed{network_seed}/"
      model_save_dir = os.getcwd() + model_save_dir
  model.train_model(model_save_dir, args.num_steps, args.print_each, args.save_each)