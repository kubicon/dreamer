import os
import numpy as np
import jax
import flax.nnx as nnx

from rnad_dreamer_joint import RNaDDreamerJoint, RNaDConfig
from dreamer_actor_critic import DreamerActorCritic, ActorCriticConfig
from dreamer_ma import DreamerMA, DreamerMAConfig
from replay_buffer import ReplayBuffer, BufferConfig
from games.jax_game import JaxGame
from train_utils import save_model, load_model



def _get_seed(seed:int):
  if seed == -1:
    return np.random.randint(0, 2**32 - 1)
  return seed

def joint_train_loop(args, game:JaxGame):
  dreamer_model_seed = _get_seed(args.dreamer_model_seed)
  ac_trajectory_seed = _get_seed(args.ac_trajectory_seed)
  dreamer_network_seed = _get_seed(args.dreamer_model_seed)
  ac_network_seed = _get_seed(args.ac_network_seed)
  
  trajectory_seed = _get_seed(args.replay_trajectory_seed)
  buffer_sample_seed = _get_seed(args.replay_sample_seed)

  saved_model_file = args.saved_model_file
  if saved_model_file and not saved_model_file.startswith("/"):
    saved_model_file = os.getcwd() + "/" + saved_model_file
  if saved_model_file:
    print(f"Loading model from path {saved_model_file}")
    #try:
    ac_model = load_model(saved_model_file)
    # except FileNotFoundError:
    #   assert False, f"Given file {saved_model_file} does not exist!"
    assert isinstance(ac_model, (RNaDDreamerJoint, DreamerActorCritic) ), f"The loaded model should be an instance of DreamerActorCritic, not {ac_model.__class__}"
    dreamer_model_seed = ac_model.world_model.config.seed
    dreamer_network_seed = ac_model.world_model.config.rng_seed
    ac_trajectory_seed = ac_model.config.seed
    ac_network_seed = ac_model.config.network_seed
    trajectory_seed = ac_model.world_model.buffer.config.trajectory_seed
    buffer_sample_seed = ac_model.world_model.buffer.config.buffer_sample_seed
    replay_buffer = ac_model.world_model.buffer
  else:
    print("Creating clean model")
    dreamer_config = DreamerMAConfig(
      batch_size=args.dreamer_batch_size,
      seed=dreamer_model_seed,
      rng_seed = dreamer_network_seed,


      #Weights of the individual loss terms of the world model
      beta_prediction = args.beta_prediction,
      beta_dynamics = args.beta_dynamics,
      beta_representation = args.beta_representation,
      
      free_bits_clip_threshold = args.free_bits_threshold,
      uniform_mix = args.uniform_mix,
      
      encoded_classes = args.encoded_classes,
      encoded_categories = args.encoded_categories,
      bin_range = args.dreamer_bin_range,

      # Ordered as (hidden_layer_features, num_hidden_layers)
      sequential_network_details = (args.hidden_state_size, args.sequential_mlp_size, args.sequential_mlp_layers),
      encoder_network_details = (args.encoder_hidden_size, args.encoder_hidden_layers),
      decoder_network_details = (args.decoder_hidden_size, args.decoder_hidden_layers),
      dynamics_network_details = (args.dynamics_hidden_size, args.dynamics_hidden_layers),
      predictor_network_details = (args.predictor_hidden_size, args.predictor_hidden_layers),
      legal_actions_network_details = (args.legal_hidden_size, args.legal_hidden_layers),

      learning_rate = args.dreamer_learning_rate
    )
    
    buffer_config = BufferConfig(trajectory_seed = trajectory_seed,
                                 buffer_sample_seed = buffer_sample_seed,
                                 buffer_size = args.buffer_size,
                                 on_policy = args.on_policy,
                                 replay_ratio = args.replay_ratio)
    replay_buffer = ReplayBuffer(game, config=buffer_config, world_model_config=dreamer_config)
    dreamer_world_model = DreamerMA(dreamer_config, replay_buffer)
    if args.train_mode == "joint_rnad":
      rnad_config = RNaDConfig(
          batch_size=args.ac_batch_size,
          network_seed = ac_network_seed,
          seed=ac_trajectory_seed,
          use_learned_model = True,
          bin_range = args.ac_bin_range,

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

          learning_rate = args.ac_learning_rate,
          
          target_network_update = args.target_network_update
      )
      ac_model = RNaDDreamerJoint(dreamer_world_model, rnad_config)
      replay_buffer.cache_sampling(dreamer_world_model.optimizers.sequence_optimizer.model,
                                   dreamer_world_model.optimizers.encoder_optimizer.model,
                                   ac_model.optimizers.rnad_optimizer.model)
    else:
      ac_config = ActorCriticConfig(
        seed=args.ac_trajectory_seed,
        network_seed = args.ac_network_seed,
        batch_size = args.ac_batch_size,

        bin_range = args.ac_bin_range,

        beta_imagination = args.beta_imagination,
        beta_real = args.beta_real,

        #Strenght of the
        # entropy exploration bonus
        eta=args.eta,

        upper_percentile = args.upper_percentile,
        lower_percentile = args.lower_percentile,
        range_ema_coeff = args.range_ema_coeff,

        #Dreamer extraction parameters
        state_sample_threshold=args.state_sample_threshold,
        terminal_threshold = args.terminal_threshold,
        legal_threshold = args.legal_threshold,
        
        #TD-estimate parameters
        gamma = args.gamma,
        td_lambda = args.td_lambda,

        # Ordered as (hidden_layer_features, num_hidden_layers)
        actor_network_details = (args.actor_hidden_size, args.actor_hidden_layers),
        critic_network_details = (args.critic_hidden_size, args.critic_hidden_layers),

        learning_rate = args.ac_learning_rate,
        target_network_update = args.target_network_update
      )
      ac_model = DreamerActorCritic(ac_config, dreamer_world_model)
      replay_buffer.cache_sampling(dreamer_world_model.optimizers.sequence_optimizer.model,
                                    dreamer_world_model.optimizers.encoder_optimizer.model,
                                    ac_model.optimizers.actor_optimizer.model)
  model_save_dir = args.model_save_dir
  game_name = game.game_name()
  empty = ""
  game_params = game.params_dict()
  params_str = f'{empty.join(f"_{value}" for key, value in game_params.items())}'
  if not model_save_dir:
      
      model_save_dir = f"/trained_networks/{args.train_mode}/{game_name}{params_str}/seeds{dreamer_model_seed}_{ac_trajectory_seed}/network_seeds_{dreamer_network_seed}_{ac_network_seed}/"
      model_save_dir = os.getcwd() + model_save_dir

  dreamer_loss, ac_img_loss, ac_real_loss = 0, 0, 0
  start_step = ac_model.learner_steps
  # _, p1_rnad_decoder_state = nnx.split(rnad_model.optimizers.p1_decoder_optimizer)
  # _, p1_dreamer_decoder_state = nnx.split(rnad_model.world_model.optimizers.p1_decoder_optimizer)
  # diff_tree = jax.tree.map(lambda x, y: np.sum((x - y) ** 2), p1_rnad_decoder_state, p1_dreamer_decoder_state)
  #print(f"RNaD init decoder state {p1_rnad_decoder_state}")
  #print(f"Dreamer init decoder state {p1_dreamer_decoder_state}")
  #jax.debug.breakpoint()

  #Start the training by sampling into the buffer,
  # to ensure that there are distinct data for at least one step
  replay_buffer.add_batch(dreamer_world_model.config.batch_size)
  for s in range(args.num_steps):
    step = start_step + s
    if args.print_each > 0 and s % args.print_each == 0:
        print(f"Step {step}, Losses: dreamer (pre RNaD updates) {dreamer_loss}, ac: img:{ac_img_loss} real: {ac_real_loss}")
    if args.save_each > 0 and s % args.save_each == 0 and (s > 0 or args.save_first):
      model_file = model_save_dir + f"step_{step}.pkl"
      save_model(ac_model, model_file)
    #TODO: How to handle the case of multiple
    # Dreamer timesteps created and multiple
    # starting points for RNaD required?
    for ds in range(args.dreamer_steps_each_step):
      dreamer_loss, dreamer_timestep, dreamer_prediction_step = ac_model.world_model.world_model_train_step()
    #jax.tree_util.tree_map(lambda x: print(x.dtype), dreamer_timestep)
    for rs in range(args.ac_steps_each_step):
      ac_img_loss, ac_real_loss = ac_model.step(dreamer_timestep, dreamer_prediction_step)