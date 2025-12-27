import os
import numpy as np

from dreamer_ma import DreamerMA, DreamerMAConfig
from games.jax_game import JaxGame
from train_utils import *



def _get_seed(seed:int):
  if seed == -1:
    return np.random.randint(0, 2**32 - 1)
  return seed

def joint_train_loop(args, game:JaxGame):
  seed = _get_seed(args.seed)

  if args.train_mode =='el':
    print(f"To you my beloved El, who was by my side the whole time I was writing this.")
    return
  
  saved_model_file = args.saved_model_file
  if saved_model_file and not saved_model_file.startswith("/"):
    saved_model_file = os.getcwd() + "/" + saved_model_file
  if saved_model_file:
    print(f"Loading model from path {saved_model_file}")
    model = load_model(saved_model_file)
    # except FileNotFoundError:
    #   assert False, f"Given file {saved_model_file} does not exist!"
    assert isinstance(model, DreamerMA), f"The loaded model should be a DreamerMA instance, not {model.__class__}"
    seed = model.init_seed

  else:
    print("Creating clean model")
    wm_config = DreamerMAConfig(
      batch_size=args.batch_size,

      use_original_iset = args.use_original_iset,

      #Weights of the individual loss terms of the world model
      beta_prediction = args.beta_prediction,
      beta_dynamics = args.beta_dynamics,
      beta_representation = args.beta_representation,
      
      free_bits_clip_threshold = args.free_bits_threshold,
      uniform_mix = args.uniform_mix,
      
      encoded_classes = args.encoded_classes,
      encoded_categories = args.encoded_categories,
      bin_range = args.wm_bin_range,

      # Ordered as (hidden_layer_features, num_hidden_layers)
      sequential_network_details = (args.recurrent_state_size, args.sequential_mlp_features, args.sequential_mlp_layers),
      encoder_network_details = (args.encoder_tokens,args.encoder_hidden_features, args.encoder_hidden_layers),
      observer_network_details = (args.observer_hidden_features, args.observer_hidden_layers),
      decoder_network_details = (args.decoder_hidden_features, args.decoder_hidden_layers),
      dynamics_network_details = (args.dynamics_hidden_features, args.dynamics_hidden_layers),
      reward_predictor_network_details = (args.reward_predictor_hidden_features, args.reward_predictor_hidden_layers),
      done_predictor_network_details = (args.done_predictor_hidden_features, args.done_predictor_hidden_layers),
      legal_actions_network_details = (args.legal_predictor_hidden_features, args.legal_predictor_hidden_layers)
    )
    
    buffer_config = BufferConfig(buffer_size = args.buffer_size,
                                 on_policy = args.on_policy,
                                 replay_ratio = args.replay_ratio)
    opt_config = OptimizerConfig(lr = args.lr,
                                       agc = args.agc,
                                       eps = args.opt_eps,
                                       beta1 = args.beta_1,
                                       beta2 = args.beta_2,
                                       momentum = args.momentum,
                                       nesterov = args.nesterov,
                                       schedule = args.opt_schedule,
                                       warmup = args.warmup,
                                       anneal = args.anneal)
    if args.train_mode == "joint_rnad":
      ac_config = RNaDConfig(
          bin_range = args.ac_bin_range,

          beta_imagination = args.beta_imagination,
          beta_real = args.beta_real,

          eta=args.eta,
          vtrace_eta = args.vtrace_eta,
          sampling_epsilon=args.sampling_epsilon,
          
          upper_percentile = args.upper_percentile,
          lower_percentile = args.lower_percentile,
          range_ema_coeff = args.range_ema_coeff,
          num_last = args.num_last,

          #World model extraction parameters
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

          #Network parameters
          rnad_network_details = (args.rnad_hidden_features, args.rnad_hidden_layers),
          
          target_network_update = args.target_network_update
      )
    else:
      ac_config = ActorCriticConfig(

        bin_range = args.ac_bin_range,

        beta_imagination = args.beta_imagination,
        beta_real = args.beta_real,

        #Strenght of the
        # entropy exploration bonus
        eta=args.eta,

        upper_percentile = args.upper_percentile,
        lower_percentile = args.lower_percentile,
        range_ema_coeff = args.range_ema_coeff,
        num_last = args.num_last,

        sampling_epsilon=args.sampling_epsilon,
        #Dreamer extraction parameters
        state_sample_threshold=args.state_sample_threshold,
        terminal_threshold = args.terminal_threshold,
        legal_threshold = args.legal_threshold,
        
        #TD-estimate parameters
        gamma = args.gamma,
        td_lambda = args.td_lambda,

        # Ordered as (hidden_layer_features, num_hidden_layers)
        actor_network_details = (args.actor_hidden_features, args.actor_hidden_layers),
        critic_network_details = (args.critic_hidden_features, args.critic_hidden_layers),

        target_network_update = args.target_network_update
      )
    model = DreamerMA(wm_config, buffer_config, ac_config, opt_config, game, seed)
  model_save_dir = args.model_save_dir
  game_name = game.game_name()
  empty = ""
  game_params = game.params_dict()
  params_str = f'{empty.join(f"_{value}" for key, value in game_params.items())}'
  if not model_save_dir:
      
      model_save_dir = f"/trained_networks/{args.train_mode}/{game_name}{params_str}/seed_{seed}/"
      model_save_dir = os.getcwd() + model_save_dir

  model.train_model(model_save_dir, args.num_steps, args.print_each, args.save_each, args.save_first)
