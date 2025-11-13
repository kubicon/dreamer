import numpy as np
import os

from train_utils import load_model
from dreamer_ma import DreamerMA
from dreamer_actor_critic import ActorCriticConfig, DreamerActorCritic


def train_actor_critic(args):
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
    assert isinstance(model, DreamerActorCritic), f"The loaded model should be an instance of DreamerActorCritic, not {model.__class__}"
  else:
    config = ActorCriticConfig(
        seed=args.trajectory_seed,
        network_seed = args.network_seed,
        batch_size = args.batch_size,

        bin_range = args.bin_range,

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

        learning_rate = args.learning_rate,
        target_network_update = args.target_network_update
    )
    saved_model_dir = args.dreamer_dir
    if not saved_model_dir.startswith("/"):
      saved_model_dir = os.getcwd() + "/" + saved_model_dir
    saved_model_dir = saved_model_dir + f"/step_{args.model_restore_step}.pkl"
    if not os.path.exists(saved_model_dir):
      raise FileNotFoundError(f"The given Dreamer path {saved_model_dir} does not exist!")
    world_model = load_model(saved_model_dir)
    assert isinstance(world_model, DreamerMA), f"The world model is expected to be an instance of DreamerMA not {world_model.__class__}"
    model = DreamerActorCritic(
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