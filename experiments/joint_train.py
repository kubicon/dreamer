import os
import numpy as np

from rnad_dreamer import RNaDDreamer, RNaDConfig
from dreamer_ma import DreamerMA, DreamerMAConfig
from games.jax_game import JaxGame
from train_utils import save_model



def _get_seed(seed:int):
  if seed == -1:
    return np.random.randint(0, 2**32 - 1)
  return seed

def joint_train_loop(args, game:JaxGame, ):
  dreamer_trajectory_seed = _get_seed(args.dreamer_trajectory_seed)
  rnad_trajectory_seed = _get_seed(args.rnad_trajectory_seed)
  dreamer_network_seed = _get_seed(args.dreamer_network_seed)
  rnad_network_seed = _get_seed(args.rnad_network_seed)

  dreamer_config = DreamerMAConfig(
      batch_size=args.dreamer_batch_size,
      seed=dreamer_trajectory_seed,


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

      learning_rate = args.dreamer_learning_rate,
      rng_seed = dreamer_network_seed
  )

  rnad_config = RNaDConfig(
      batch_size=args.rnad_batch_size,
      seed=rnad_trajectory_seed,
      use_learned_model = args.use_learned_model,
      send_signal_to_dreamer = True,

      eta=args.eta,
      vtrace_eta = args.vtrace_eta,
      sampling_epsilon=args.sampling_epsilon,
      state_sample_threshold=args.state_sample_threshold,

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

      learning_rate = args.rnad_learning_rate,
      network_seed = rnad_network_seed
  ) 
  model_save_dir = args.model_save_dir
  game_name = game.game_name()
  empty = ""
  game_params = game.params_dict()
  params_str = f'{empty.join(f"_{value}" for key, value in game_params.items())}'
  if not model_save_dir:
      
      model_save_dir = f"/trained_networks/compound/{game_name}{params_str}/seeds{dreamer_trajectory_seed}_{rnad_trajectory_seed}/network_seeds_{dreamer_network_seed}_{rnad_network_seed}/"
      model_save_dir = os.getcwd() + model_save_dir

  dreamer_world_model = DreamerMA(dreamer_config, game)
  rnad_model = RNaDDreamer(dreamer_world_model, rnad_config)
  dreamer_loss, rnad_loss = 0, 0
  for step in range(args.num_steps):
    if args.print_each > 0 and step % args.print_each == 0:
        print(f"Step {step}, Losses: dreamer (pre RNaD updates) {dreamer_loss}, rnad: {rnad_loss}")
    if args.save_each > 0 and step % args.save_each == 0:
      model_file = model_save_dir + f"step_{step}.pkl"
      #TODO: Save just the RNaD model, or save both of them?
      save_model(rnad_model, model_file)
    for ds in range(args.dreamer_steps_each_step):
      dreamer_loss = dreamer_world_model.world_model_train_step()
    for rs in range(args.rnad_steps_each_step):
      rnad_loss = rnad_model.step_with_model()
