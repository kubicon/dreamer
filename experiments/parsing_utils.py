
from argparse import ArgumentParser

def add_actor_critic_arguments(parser: ArgumentParser) -> ArgumentParser:
  """Adds actor-critic required parameters to parser."""
  
  
  parser.add_argument(f"--eta", type=float, default=3e-4, help="Coefficient for entropy exploration bonus for Reinforce")
  parser.add_argument(f"--gamma", type=float, default=0.997, help="Discount factor for TD-learning")
  parser.add_argument(f"--td_lambda", type=float, default=0.95, help="Lambda parameter for TD-learning")


  parser.add_argument("--upper_percentile", type=float, default=95, help="Upper percentile for the return normalization range")
  parser.add_argument("--lower_percentile", type=float, default=5, help="Lower percentile for the return normalization range")
  parser.add_argument("--range_ema_coeff", type=float, default=0.99, help="Coefficient for the EMA update of return normalization range")
  parser.add_argument("--num_last", type=int, default=-1, help="How many steps from the end of the trajectory to take as starting points for imagination. If <= 0, take the entire trajectory.")



def add_wm_arguments(parser: ArgumentParser) ->ArgumentParser:
  """Adds all the world model required parameters to parser."""

  ##Model parameters 
  parser.add_argument("--encoded_categories", type=int, default=32, help="Number of options for each categorical distribution in the latent state.")
  parser.add_argument("--encoded_classes", type=int, default=32, help="Number of categorical distributions in the latent state")
  parser.add_argument("--recurrent_state_size", type=int, default =-1, help="Size of the RNN sequential state. If -1 it is set to the size of joint infoset over both player. ")
  parser.add_argument("--encoder_tokens", type=int, default=256, help="Size of the observation latent representation produced by encoder.")
  parser.add_argument(f"--wm_bin_range", type=int, default=20, help="Number of the exponentially spaced bins for certain predictions such as reward in one direction, bins will be spaced out as symexp([-bin_range, ..., bin_range])")

  parser.add_argument(f"--batch_size", type=int, default=32, help="Batch size for training")
  
  parser.add_argument(f"--use_original_infoset", action="store_true", help="A debug flag that forces training actor-critic on original game infosets even outside IIGs.")
  parser.add_argument(f"--latent_infoset_size", type=int, default=-1, help="Size of the latent infoset vector. If < 0, will make the latent infoset have the same size as real infoset")

  #Replay buffer parameters
  parser.add_argument("--buffer_size", type=int, default=32, help="Size of the replay buffer.")
  parser.add_argument("--replay_ratio", type=int, default=-1, help="The replay ratio, which defines the amount of online steps per minibatch. Respectively, the ratio is replay_ratio / (batch_size * trajectory_len). If -1, only online trajectories are sampled")
  parser.add_argument("--trajectory_sample_eps", action="store_true", help="A flag whether to use the actor policy for trajectory sampling. If not, uniform policy is used instead.")
  parser.add_argument("--return_log_frequency", type=int, default=10, help="How often to log trajectory return in the replay buffer in terms of collected minibatches")
  parser.add_argument("--smoothing_window", type=int, default=32, help="How many returns to use for the running average window")
  parser.add_argument("--log_returns", action="store_true", help="A flag whether to log the smoothed returns. They will be stored in the same directory as the model.")
  ## Loss function coefficients
  parser.add_argument("--beta_prediction", type=float, default=1, help="The beta coefficient for predictor loss")
  parser.add_argument("--beta_dynamics", type=float, default=1, help="The beta coefficient for dynamics (pushing prior prediction towards posterior) loss")
  parser.add_argument("--beta_representation", type=float, default=0.1, help="The beta coefficient for representation (pushing posterior representation toward prior) loss")
  parser.add_argument("--beta_infoset", type=float, default=1.0, help="The beta coefficient for latent infoset learning loss")
  
  parser.add_argument("--free_bits_threshold", type=float, default=1, help="Clipping threshold for the dynamics and representation losses in free bits.")
  parser.add_argument("--uniform_mix", type=float, default=0.01, help="Amount of uniform mixed with the network returned categoricals.")

  ##Network layer parameters
  parser.add_argument("--sequential_mlp_features", type=int, default=256, help="Number of hidden features in the sequential network MLP.")
  parser.add_argument("--encoder_hidden_features", type=int, default=256, help="Size of the hidden layer in the encoder network")
  parser.add_argument("--dynamics_hidden_features", type=int, default=256, help="Size of the hidden layer in the dynamics network")
  parser.add_argument("--decoder_hidden_features", type=int, default=256, help="Size of the hidden layer in the decoder network")
  parser.add_argument("--observer_hidden_features", type=int, default=256, help="Size of the hidden layer in the observer network, the one that produces stochastic state from observation encoding and recurrent state.")
  parser.add_argument("--reward_predictor_hidden_features", type=int, default=256, help="Size of the hidden layer in the reward predictor network")
  parser.add_argument("--done_predictor_hidden_features", type=int, default=256, help="Size of the hidden layer in the terminal predictor network")
  parser.add_argument("--legal_predictor_hidden_features", type=int, default=256, help="Size of the hidden layer in the legal actions predictor network")
  parser.add_argument("--infoset_network_hidden_features", type=int, default=256, help="Size of the hidden layer in the infoset network, the RNN creating latent infoset from action-observation history.")
  parser.add_argument("--infoset_decoder_hidden_features", type=int, default=256, help="Size of the hidden layer in the infoset decoder network, the one that real observation and previous action from latent infoset")
  parser.add_argument("--infoset_predictor_hidden_features", type=int, default=256, help="Size of the hidden layer in the infoset predictor network,the one that predicts the model state from both players latent infosets.")
  
  parser.add_argument("--sequential_mlp_layers", type=int, default=1, help="Number of hidden layers for the sequential network MLP")
  parser.add_argument("--encoder_hidden_layers", type=int, default=1, help="Number of hidden layers in the encoder network")
  parser.add_argument("--dynamics_hidden_layers", type=int, default=1, help="Number of hidden layers in the dynamics network")
  parser.add_argument("--decoder_hidden_layers", type=int, default=1, help="Number of hidden layers in the decoder network")
  parser.add_argument("--observer_hidden_layers", type=int, default=1, help="Number of hidden layers in the observer network")
  parser.add_argument("--reward_predictor_hidden_layers", type=int, default=1, help="Number of hidden layers in the reward predictor network")
  parser.add_argument("--done_predictor_hidden_layers", type=int, default=1, help="Number of hidden layers in the terminal predictor network")
  parser.add_argument("--legal_predictor_hidden_layers", type=int, default=1, help="Number of hidden layers in the legal actions predictor network")
  parser.add_argument("--infoset_network_hidden_layers", type=int, default=1, help="Number of hidden layers in the infoset network, the RNN creating latent infoset from action-observation history.")
  parser.add_argument("--infoset_decoder_hidden_layers", type=int, default=1, help="Number of hidden layers in the infoset decoder network, the one that real observation and previous action from latent infoset")
  parser.add_argument("--infoset_predictor_hidden_layers", type=int, default=1, help="Number of hidden layers in the infoset predictor network,the one that predicts the model state from both players latent infosets.")

  return parser


def add_rnad_arguments(parser: ArgumentParser) ->ArgumentParser:
  """Add all the RNaD required arguments to the given parser.
  If joint_train is specified, certain parameters, such as batch_size, 
  that share name with Dreamer, have rnad_ prepended to differentiate between them.
  Also, for joint_train some training loop arguments like dreamer_path are not specified."""
  
  ##RNaD parameters  
  parser.add_argument("--eta", type=float, default=0.2, help="Strenght of the regularization in RNaD. Used for the reward transformation and the KL regularization for V-trace.")
  ##Entropy schedule- network switching
  parser.add_argument("--entropy_schedule_size", default=(100, 1000), help="Defines how many iterations should be done for each item in the sequence.")
  parser.add_argument("--entropy_schedule_repeats", default=(10,1), help="Defines amount of network switching sequences for each item in the sequence. Make sure last element is 1. For details see the EntropySchedule class.")

  ##V-Trace paraemters
  parser.add_argument("--rho_vtrace", type=float, default=-1.0, help="Rho clipping parameter for V-Trace. If < 0 treated as infinity (no clipping)")
  parser.add_argument("--c_vtrace", type=float, default=-1.0, help="C clipping parameter for V-Trace. If < 0 treated as infinity (no clipping)")
  parser.add_argument("--gamma_vtrace", type=float, default=1.0, help="Discount factor for V-Trace")
  parser.add_argument("--lambda_vtrace", type=float, default=1.0, help="Lambda parameter for V-Trace")

  ##NeuRD parameters, currently not used, because currently the return normalization is used
  parser.add_argument("--neurd_clip", type=float, default=10000, help="Clip parameter for NeuRD")
  parser.add_argument("--neurd_threshold", type=float, default=2, help="Threshold parameter for NeuRD")

  return parser

def add_optimizer_arguments(parser: ArgumentParser) -> ArgumentParser:
  """Adds the parameters required by the optimizer to the parser.
  """

  # Core optimization hyperparameters
  parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for the optimizer.")
  parser.add_argument("--agc", type=float, default=0.3, help="Adaptive Gradient Clipping (AGC) threshold.")
  parser.add_argument("--opt_eps", type=float, default=1e-8, help="Epsilon term for numerical stability in the optimizer.")
  
  # Adam / Momentum specific
  parser.add_argument("--beta_1", type=float, default=0.9, help="The exponential decay rate for the 1st moment estimates.")
  parser.add_argument("--beta_2", type=float, default=0.999, help="The exponential decay rate for the 2nd moment estimates.")
  
  # Flags for momentum variants
  parser.add_argument("--no_momentum", action="store_false", dest="momentum", help="Disable momentum.")
  parser.set_defaults(momentum=True)
  parser.add_argument("--nesterov", action="store_true", help="Whether to use Nesterov momentum.")

  # Learning rate schedule parameters
  parser.add_argument("--opt_schedule", type=str, default="const", choices=["const", "linear", "cosine"], 
                      help="Type of learning rate schedule to use.")
  parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps for the schedule.")
  parser.add_argument("--anneal", type=int, default=0, help="Number of annealing steps for the schedule.")
  return parser


def prepare_experiment_parser():
  """Prepares a parser from the complete experiment, that allows
  distinguishing whether to train with RNaD or
  standard Actor-Critic."""
  parser = ArgumentParser()
  parser.add_argument("--seeds", type=str, default='(42, )', help="RNG seeds for the whole algorithm. Supplied as (seed_1, seed_2, ..., seed_n) If -1 a random seed is generated.")
  parser.add_argument("--num_steps", type=int, default=1001, help="Number of training steps")
  parser.add_argument("--save_each", type=int, default=100, help="Save model every N steps")
  parser.add_argument("--save_first", action="store_true", help="A flag whether to save the initial state of the model.")
  parser.add_argument("--print_each", type=int, default=100, help="Print loss every N steps")
  parser.add_argument("--model_save_dir", type=str, default="", help="Directory to save the trained model")
  parser.add_argument("--continue_train", action="store_true", help="A flag whether to continue training from the latest stored step. If specified a clean model is trained.")
  parser.add_argument("--clean_dir", action="store_true", help="A flag whether to first delete the model store directory, if it already exists. Incompatible with continue train and takes precedence over it, if supplied together")

  parser.add_argument("--train_real_policy", action="store_true", help="A flag whether to learn actor on real trajectories also")
  parser.add_argument("--report_gradnorms", action="store_true", help="Whether to report gradient norms as well as losses.")

  #Uniform mixtures in sampling policies.
  
  parser.add_argument("--img_sampling_epsilon", type=float, default=0.0, help="Defines mix of uniform policy to the network learned policy during imagination trajectory sampling.")
  parser.add_argument("--real_sampling_epsilon", type=float, default=0.0, help="Defines mix of uniform policy to the network learned policy during real trajectory sampling.")

  # Actor-critic parameters shared both for Reinforce and RNaD
  parser.add_argument("--num_last", type=int, default=-1, help="How many steps from the end of the trajectory to take as starting points for imagination. If <= 0, take the entire trajectory.")


  parser.add_argument("--state_sample_threshold", type=float, default=0.05, help="Threshold when sampling states. Outcomes below this threshold are ignored.")
  parser.add_argument("--terminal_threshold", type=float, default=0.5, help="Threshold when to consider the state terminal.")
  parser.add_argument("--legal_threshold", type=float, default=0.5, help="Threshold for considering actions legal.")
  parser.add_argument(f"--ac_bin_range", type=int, default=20, help="Number of the exponentially spaced bins for the value categorical distribution prediction")

  parser.add_argument("--actor_hidden_features", type=int, default=256, help="Size of the hidden layer the actor network.")
  parser.add_argument("--actor_hidden_layers", type=int, default=1, help="Number of hidden layers for the actor network.")
  parser.add_argument("--critic_hidden_features", type=int, default=256, help="Size of the hidden layer for the critic network.")
  parser.add_argument("--critic_hidden_layers", type=int, default=1, help="Number of hidden layers for the critic network.")

  parser.add_argument(f"--target_network_update", type=float, default=1e-3, help="1 - EMA coefficient for target network update")


  parser.add_argument(f"--beta_imagination", type=float, default=1.0, help="Coefficient for loss on Dreamer imagined trajectories")
  parser.add_argument(f"--beta_real", type=float, default=0.3, help="Coefficient for loss on real environment trajectories")

  parser = add_optimizer_arguments(parser)
  parser = add_wm_arguments(parser)

  subparsers = parser.add_subparsers(dest="train_mode", required=True, help="Which training mode to run. Either reinforce or rnad")

  joint_parser = subparsers.add_parser(name="reinforce", help="Train both world model and standard Dreamer Reinforce + TD-learning")
  joint_parser = add_actor_critic_arguments(joint_parser)

  joint_rnad_parser = subparsers.add_parser(name="rnad", help="Train both world model and RNaD as the actor-critic.")
  joint_rnad_parser = add_rnad_arguments(joint_rnad_parser)

  return parser