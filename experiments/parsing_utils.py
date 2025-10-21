
from argparse import ArgumentParser

def add_dreamer_arguments(parser: ArgumentParser, multi_agent: bool = True,
                          joint_train: bool = False) ->ArgumentParser:
  """Adds all the dreamer required parameters to parser.
  Use the multi-agent bool to control which version of Dreamer to 
  initialize for. If joint_train is set to True,
  it ensures that the parameter names that are shared among Dreamer and
  RNaD (such as batch size) get dreamer_ prepended to differentiate them."""
  ##Model parameters 
  parser.add_argument("--encoded_categories", type=int, default=32, help="Number of options for each categorical distribution in the latent state.")
  parser.add_argument("--encoded_classes", type=int, default=32, help="Number of categorical distributions in the latent state")
  parser.add_argument("--hidden_state_size", type=int, default =256, help="Size of the RNN hidden state")
  parser.add_argument("--bin_range", type=int, default=20, help="Number of the exponentially spaced bins for certain predictions such as reward in one direction, bins will be spaced out as symexp([-bin_range, ..., bin_range])")

  diff_string = "dreamer_" if joint_train else ""
  parser.add_argument(f"--{diff_string}batch_size", type=int, default=32, help="Batch size for training")
  parser.add_argument(f"--{diff_string}learning_rate", type=float, default=3e-4, help="Learning rate for the optimizer")
  parser.add_argument(f"--{diff_string}network_seed", type=int, default=-1, help="Random seed for network initialization")
  parser.add_argument(f"--{diff_string}trajectory_seed", type=int, default=-1, help="Random seed for trajectory generation")

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
  if joint_train:
    parser.add_argument("--dreamer_steps_each_step", type=int, default=1, help="How many Dreamer steps to perform in each step of the main algorithm loop.")
  return parser


def add_rnad_arguments(parser: ArgumentParser, joint_train: bool =False) ->ArgumentParser:
  """Add all the RNaD required arguments to the given parser.
  If joint_train is specified, certain parameters, such as batch_size, 
  that share name with Dreamer, have rnad_ prepended to differentiate between them.
  Also, for joint_train some training loop arguments like dreamer_path are not specified."""
  ##RNaD parameters  
  parser.add_argument("--target_network_update", type=float, default=1e-3, help="Update rate for target network")
  parser.add_argument("--eta", type=float, default=0.2, help="Strenght of the regularization in RNaD. Used for the reward transformation and the KL regularization for V-trace.")
  parser.add_argument("--vtrace_eta", type=float, default=0.2, help="Strenght of the additional KL regularization term in V-trace.")
  parser.add_argument("--sampling_epsilon", type=float, default=0.0, help="Defines mix of uniform policy to the network learned policy during trajectory sampling.")
  parser.add_argument("--use_learned_model", type=bool, default=True, help="Whether to use the Dreamer learned model for trajectory sampling. If not, trajectories are sampled from the game. Just for debugging.")

  #Dreamer model extraction parameters
  parser.add_argument("--state_sample_threshold", type=float, default=0.05, help="Threshold for the stochastic state sampling. If the probability of a class is below this threshold, it is not sampled.")
  parser.add_argument("--terminal_threshold", type=float, default=0.5, help="How much probability must the softmaxed logit have, to consider the state terminal.")
  parser.add_argument("--legal_threshold", type=float, default=0.5, help="How much probability must the softmaxed logit have, to consider the action legal.")

  #Loss coefficients
  parser.add_argument("--beta_imagination", type=float, default=1.0, help="Coefficient for the loss on Dreamer imagined trajectories.")
  parser.add_argument("--beta_real", type=float, default=0.3, help="Coefficient for the loss on trajectories sampled from the real environment.")

  diff_string = "rnad_" if joint_train else ""
  parser.add_argument(f"--{diff_string}batch_size", type=int, default=32, help="Batch size for training")
  parser.add_argument(f"--{diff_string}learning_rate", type=float, default=3e-4, help="Learning rate for the optimizer")
  parser.add_argument(f"--{diff_string}network_seed", type=int, default=-1, help="Random seed for network initialization")
  parser.add_argument(f"--{diff_string}trajectory_seed", type=int, default=-1, help="Random seed for trajectory generation")

  ##Entropy schedule- network switching
  parser.add_argument("--entropy_schedule_size", default=(100,), help="Defines how many iterations should be done for each item in the sequence.")
  parser.add_argument("--entropy_schedule_repeats", default=(1,), help="Defines amount of network switching sequences for each item in the sequence. Make sure last element is 1. For details see the EntropySchedule class.")

  ##V-Trace paraemters
  parser.add_argument("--rho_vtrace", type=float, default=1.0, help="Rho clipping parameter for V-Trace")
  parser.add_argument("--c_vtrace", type=float, default=1.0, help="C clipping parameter for V-Trace")
  parser.add_argument("--gamma_vtrace", type=float, default=1.0, help="Discount factor for V-Trace")
  parser.add_argument("--lambda_vtrace", type=float, default=1.0, help="Lambda parameter for V-Trace")


  ##NeuRD parameters
  parser.add_argument("--neurd_clip", type=float, default=10000, help="Clip parameter for NeuRD")
  parser.add_argument("--neurd_threshold", type=float, default=2, help="Threshold parameter for NeuRD")

  ##Network layer parameters
  parser.add_argument("--network_hidden_size", type=int, default=256, help="Size of the hidden layer in the RNaD network")
  parser.add_argument("--network_hidden_layers", type=int, default=1, help="Number of stacked hidden layers in the RNaD network")

  if not joint_train:
    #Complete path to restore the whole model
    ## World model path
    parser.add_argument("--dreamer_dir", type=str, default="trained_networks/dreamer/goofspiel_3/seed99/network_seed99", help="Path to where is the saved Dreamer trained world model") 
    parser.add_argument("--model_restore_step", type=int, default=1000, help="Which saved step of the Dreamer world model to restore.")
  else:
    parser.add_argument("--rnad_steps_each_step", type=int, default=1, help="How many RNaD steps to perform in each step of the main algorithm loop.")
  return parser

def prepare_experiment_parser(multi_agent: bool = True):
  """Prepares a parser from the complete experiment, that allows
  distinguishing whether to train Dreamer, RNaD, or both jointly.
  Importantly, if multi_agent is False, only Dreamer single agent Dreamer
  training is allowed."""
  parser = ArgumentParser()
  parser.add_argument("--num_steps", type=int, default=1001, help="Number of training steps")
  parser.add_argument("--save_each", type=int, default=100, help="Save model every N steps")
  parser.add_argument("--save_first", action="store_true", help="A flag whether to save the initial state of the model.")
  parser.add_argument("--print_each", type=int, default=100, help="Print loss every N steps")
  parser.add_argument("--model_save_dir", type=str, default="", help="Directory to save the trained model")
  parser.add_argument("--saved_model_file", type=str, default="", help="File with the complete model. Used for continuing to train it.")
  if not multi_agent:
    parser = add_dreamer_arguments(parser, multi_agent=False, joint_train=False)
    return parser
  #Handle the fact that this experiment can be called either
  # to train Dreamer, RNaD, or both jointly
  subparsers = parser.add_subparsers(dest="train_mode", required=True, help="Which training mode to run. dreamer for Dreamer train, rnad for RNaD train or joint for the joint training loop.")
  dreamer_parser = subparsers.add_parser(name="dreamer", help="Train only the Dreamer world model.")
  dreamer_parser = add_dreamer_arguments(dreamer_parser, multi_agent=True, joint_train=False)

  rnad_parser = subparsers.add_parser(name="rnad", help="Train only the RNaD algorithm on already trained Dreamer model.")
  rnad_parser = add_rnad_arguments(rnad_parser, joint_train=False)

  joint_parser = subparsers.add_parser(name="joint", help="Train both algorithms jointly. First performing K Dreamer model steps and then L RNaD steps (typically l = 1).")
  joint_parser = add_dreamer_arguments(joint_parser, multi_agent=True, joint_train=True)
  joint_parser = add_rnad_arguments(joint_parser, joint_train=True)

  return parser