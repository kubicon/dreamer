import os
import numpy as np
import jax
import flax.nnx as nnx

from rnad_dreamer_joint import RNaDDreamerJoint, RNaDConfig
from dreamer_ma import DreamerMA, DreamerMAConfig
from replay_buffer import ReplayBuffer
from games.jax_game import JaxGame
from train_utils import save_model, load_model



def _get_seed(seed:int):
  if seed == -1:
    return np.random.randint(0, 2**32 - 1)
  return seed

def joint_train_loop(args, game:JaxGame):
  dreamer_model_seed = _get_seed(args.dreamer_model_seed)
  rnad_trajectory_seed = _get_seed(args.rnad_trajectory_seed)
  dreamer_network_seed = _get_seed(args.dreamer_model_seed)
  rnad_network_seed = _get_seed(args.rnad_network_seed)
  
  trajectory_seed = _get_seed(args.replay_trajectory_seed)
  buffer_sample_seed = _get_seed(args.replay_sample_seed)

  saved_model_file = args.saved_model_file
  if saved_model_file and not saved_model_file.startswith("/"):
    saved_model_file = os.getcwd() + "/" + saved_model_file
  if saved_model_file:
    print(f"Loading model from path {saved_model_file}")
    #try:
    rnad_model = load_model(saved_model_file)
    # except FileNotFoundError:
    #   assert False, f"Given file {saved_model_file} does not exist!"
    assert isinstance(rnad_model, RNaDDreamerJoint), f"The loaded model should be an instance of RnaDDreamerJoint, not {rnad_model.__class__}"
    dreamer_model_seed = rnad_model.world_model.config.seed
    dreamer_network_seed = rnad_model.world_model.config.rng_seed
    rnad_trajectory_seed = rnad_model.config.seed
    rnad_network_seed = rnad_model.config.network_seed
    trajectory_seed = rnad_model.world_model.buffer.jax_seed
    buffer_sample_seed = rnad_model.world_model.buffer.np_seed
    replay_buffer = rnad_model.world_model.buffer
  else:
    print("Creating clean model")
    dreamer_config = DreamerMAConfig(
        batch_size=args.dreamer_batch_size,
        seed=dreamer_model_seed,


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
        use_learned_model = True,

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

        learning_rate = args.rnad_learning_rate,
        network_seed = rnad_network_seed
    )
    replay_buffer = ReplayBuffer(game, trajectory_seed, buffer_sample_seed, args.replay_size)
    dreamer_world_model = DreamerMA(dreamer_config, replay_buffer)
    rnad_model = RNaDDreamerJoint(dreamer_world_model, rnad_config) 
  model_save_dir = args.model_save_dir
  game_name = game.game_name()
  empty = ""
  game_params = game.params_dict()
  params_str = f'{empty.join(f"_{value}" for key, value in game_params.items())}'
  if not model_save_dir:
      
      model_save_dir = f"/trained_networks/compound/{game_name}{params_str}/seeds{dreamer_model_seed}_{rnad_trajectory_seed}/network_seeds_{dreamer_network_seed}_{rnad_network_seed}/"
      model_save_dir = os.getcwd() + model_save_dir

  dreamer_loss, rnad_img_loss, rnad_real_loss = 0, 0, 0
  start_step = rnad_model.learner_steps
  # _, p1_rnad_decoder_state = nnx.split(rnad_model.optimizers.p1_decoder_optimizer)
  # _, p1_dreamer_decoder_state = nnx.split(rnad_model.world_model.optimizers.p1_decoder_optimizer)
  # diff_tree = jax.tree.map(lambda x, y: np.sum((x - y) ** 2), p1_rnad_decoder_state, p1_dreamer_decoder_state)
  #print(f"RNaD init decoder state {p1_rnad_decoder_state}")
  #print(f"Dreamer init decoder state {p1_dreamer_decoder_state}")
  #jax.debug.breakpoint()

  #Start the training by sampling into the buffer,
  # to ensure that there are distinct data for at least one step
  replay_buffer.add_batch(dreamer_world_model.config.batch_size)
  to_collect = 0
  for s in range(args.num_steps):
    step = start_step + s
    if args.print_each > 0 and s % args.print_each == 0:
        print(f"Step {step}, Losses: dreamer (pre RNaD updates) {dreamer_loss}, rnad: img:{rnad_img_loss} real: {rnad_real_loss}")
    if args.save_each > 0 and s % args.save_each == 0 and (s > 0 or args.save_first):
      model_file = model_save_dir + f"step_{step}.pkl"
      save_model(rnad_model, model_file)
    #TODO: How to handle the case of multiple
    # Dreamer timesteps created and multiple
    # starting points for RNaD required?
    for ds in range(args.dreamer_steps_each_step):
      #_, p1_dreamer_decoder_state = nnx.split(rnad_model.world_model.optimizers.p1_decoder_optimizer)
      #jax.debug.breakpoint()
      dreamer_loss, dreamer_timestep, dreamer_prediction_step = rnad_model.world_model.world_model_train_step()
    #jax.tree_util.tree_map(lambda x: print(x.dtype), dreamer_timestep)
    for rs in range(args.rnad_steps_each_step):
      rnad_img_loss, rnad_real_loss = rnad_model.step(dreamer_timestep, dreamer_prediction_step)
    to_collect += args.replay_fraction
    if to_collect >= 1:
      to_collect = int(to_collect)
      replay_buffer.add_batch(to_collect)
      to_collect = 0