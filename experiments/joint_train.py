import os

from shutil import rmtree

from dreamer_ma import DreamerMA, DreamerMAConfig, LATEST_STEP_FILENAME
from games.jax_game import JaxGame
from train_utils import *
from experiments.eval_utils import track


def train(args, game: JaxGame):
  """Perform the NashDreamer training on a particular game

  Args:
      args (_type_): Argument specification. Detailed description of arguments can be found in parsing_utils.py
      game (JaxGame): The game to train on
  """
  seeds = get_seeds(args.seeds)
  #Create the initial model. All the other
  # models will only change the model 
  # state to prevent retracing
  wm_config = DreamerMAConfig(
      batch_size=args.batch_size,
      report_gradnorms = args.report_gradnorms,

      use_original_infoset = args.use_original_infoset,

      #Weights of the individual loss terms of the world model
      beta_prediction = args.beta_prediction,
      beta_dynamics = args.beta_dynamics,
      beta_representation = args.beta_representation,
      beta_infoset = args.beta_infoset,
      
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
      legal_actions_network_details = (args.legal_predictor_hidden_features, args.legal_predictor_hidden_layers),
      infoset_network_details = (args.latent_infoset_size, args.infoset_network_hidden_features, args.infoset_network_hidden_layers),
      infoset_decoder_details = (args.infoset_decoder_hidden_features, args.infoset_decoder_hidden_layers),
      infoset_predictor_details = (args.infoset_predictor_hidden_features, args.infoset_predictor_hidden_layers)
    )
    
  buffer_config = BufferConfig(buffer_size = args.buffer_size,
                                sampling_epsilon = args.real_sampling_epsilon,
                                replay_ratio = args.replay_ratio,
                                smoothing_window = args.smoothing_window,
                                log_returns = args.log_returns,
                                return_log_frequency = args.return_log_frequency)
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
  if args.train_mode == "rnad":
    ac_config = RNaDConfig(
        bin_range = args.ac_bin_range,
        train_real_policy = args.train_real_policy,
        report_gradnorms = args.report_gradnorms,

        beta_imagination = args.beta_imagination,
        beta_real = args.beta_real,

        eta=args.eta,
        sampling_epsilon=args.img_sampling_epsilon,

        num_last = args.num_last,

        #World model extraction parameters
        state_sample_threshold=args.state_sample_threshold,
        terminal_threshold = args.terminal_threshold,
        legal_threshold = args.legal_threshold,

        # Entropy schedule parameters
        entropy_schedule_size = args.entropy_schedule_size,
        entropy_schedule_repeats = args.entropy_schedule_repeats,
        
        #V-Trace parameters
        rho_vtrace = args.rho_vtrace if args.rho_vtrace >= 0 else jnp.inf,
        c_vtrace = args.c_vtrace if args.c_vtrace >= 0 else jnp.inf,
        gamma_vtrace = args.gamma_vtrace,
        lambda_vtrace = args.lambda_vtrace,

        # NeuRD parameters
        neurd_clip = args.neurd_clip,
        neurd_threshold = args.neurd_threshold,

        # Ordered as (hidden_layer_features, num_hidden_layers)
        actor_network_details = (args.actor_hidden_features, args.actor_hidden_layers),
        critic_network_details = (args.critic_hidden_features, args.critic_hidden_layers),
        
        target_network_update = args.target_network_update
    )
  else:
    ac_config = ActorCriticConfig(

      bin_range = args.ac_bin_range,
      train_real_policy = args.train_real_policy,
      report_gradnorms = args.report_gradnorms,

      beta_imagination = args.beta_imagination,
      beta_real = args.beta_real,

      #Strenght of the
      # entropy exploration bonus
      eta=args.eta,

      upper_percentile = args.upper_percentile,
      lower_percentile = args.lower_percentile,
      range_ema_coeff = args.range_ema_coeff,
      num_last = args.num_last,

      sampling_epsilon=args.img_sampling_epsilon,

      #World model extraction parameters
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
  model = DreamerMA(wm_config, buffer_config, ac_config, opt_config, game, seeds[0])
  for seed in seeds:
    joint_train_loop(args, seed, model)

@track
def joint_train_loop(args, seed:int, template_model: DreamerMA):
  """Run the actual training loop

  Args:
      args (_type_): Argument specification. Detailed description of arguments can be found in parsing_utils.py
      seed (int): The PRNG seed for this training instance
      template_model (DreamerMA): A precreated template model, that has the same 
      parameters as all the models during the training, except seed. This is used
      to just update the network/optimizer state and seed of the template model
      instead of initializing new one each time, to avoid unnnecessary retracing.
  """
  print(f"Running the training for seed {seed}")
  game = template_model.game
  model_save_dir = args.model_save_dir
  if not model_save_dir:
      
      model_save_dir = f"/trained_networks/{args.train_mode}/{game.to_compact_str()}/seed_{seed}/"
      model_save_dir = os.getcwd() + model_save_dir
  saved_model_file = ""
  if args.clean_dir:
    if args.continue_train:
      print(f"Warning! clean_dir and continue_train flags were supplied together. clean_dir is taking precedence.")
    try:
      rmtree(model_save_dir)
    except Exception as e:
      print(f"Removing a directory {model_save_dir} failed with exception {e}.")
  if args.continue_train and not args.clean_dir:
    latest_step_file = model_save_dir + LATEST_STEP_FILENAME
    try:
      with open(latest_step_file, 'r') as f:
        latest_step_suffix = f.readline()
      saved_model_file = model_save_dir + latest_step_suffix
    except FileNotFoundError as e:
      print(f"File {latest_step_file} was not found. Creating a clean model.")

  if saved_model_file:
    print(f"Loading model from path {saved_model_file}")
    model = load_model(saved_model_file)
    assert isinstance(model, DreamerMA), f"The loaded model should be a DreamerMA instance, not {model.__class__}"
    assert seed == model.init_seed, f"The given seed {seed} and the initial seed of the stored model {model.init_seed} do not match."

  else:
    print("Creating clean model")
    model = DreamerMA(template_model.wm_config, template_model.buffer_config, template_model.ac_config, template_model.opt_config, game, seed)
  #Will still retrace the nnx networks.
  # We have to do this, as the seed affects
  # their initialization as well.
  # Dont know if there is a way to avoid the retracing
  # and still update the initial state.
  template_model.__setstate__(model.__getstate__())
  template_model.train_model(model_save_dir, args.num_steps, args.print_each, args.save_each, args.save_first)
