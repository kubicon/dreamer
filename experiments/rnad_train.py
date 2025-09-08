from argparse import ArgumentParser
import numpy as np
import os

from train_utils import load_model
from dreamer_ma import DreamerMA
from rnad_dreamer import RNaDConfig, RNaDDreamer


parser = ArgumentParser()
##RNaD parameters  
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for the optimizer")
parser.add_argument("--target_network_update", type=float, default=1e-3, help="Update rate for target network")
parser.add_argument("--network_seed", type=int, default=-1, help="Random seed for network initialization")
parser.add_argument("--trajectory_seed", type=int, default=-1, help="Random seed for trajectory generation")
parser.add_argument("--eta", type=float, default=0.2, help="Strenght of the regularization in RNaD. Used for the reward transformation and the KL regularization for V-trace.")
parser.add_argument("--vtrace_eta", type=float, default=0.2, help="Strenght of the additional KL regularization term in V-trace.")
parser.add_argument("--sampling_epsilon", type=float, default=0.0, help="Defines mix of uniform policy to the network learned policy during trajectory sampling.")
parser.add_argument("--state_sample_threshold", type=float, default=0.05, help="Threshold for the stochastic state sampling. If the probability of a class is below this threshold, it is not sampled.")
parser.add_argument("--use_learned_model", type=bool, default=False, help="Whether to use the Dreamer learned model for trajectory sampling. If not, trajectories are sampled from the game. Just for debugging.")

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


## World model path
parser.add_argument("--dreamer_path", type=str, default="trained_networks/dreamer/goofspiel_3/seed99/network_seed99", help="Path to where is the saved Dreamer trained world model") 
parser.add_argument("--model_restore_step", type=int, default=1000, help="Which saved step of the Dreamer world model to restore.")

## Training parameters
parser.add_argument("--num_steps", type=int, default=1001, help="Number of training steps")
parser.add_argument("--save_each", type=int, default=100, help="Save model every N steps")
parser.add_argument("--print_each", type=int, default=100, help="Print loss every N steps")
parser.add_argument("--model_save_dir", type=str, default="", help="Directory to save the trained model")

def main():
  #profiler = Profiler()
  args = parser.parse_args()
  network_seed = args.network_seed
  trajectory_seed = args.trajectory_seed
  if network_seed == -1:
    network_seed = np.random.randint(0, 2**32 - 1)
  if trajectory_seed == -1:
    trajectory_seed = np.random.randint(0, 2**32 - 1)
  print(f"Using network seed: {network_seed}, trajectory seed: {trajectory_seed}")
  config = RNaDConfig(
      batch_size=args.batch_size,
      seed=args.trajectory_seed,
      use_learned_model = args.use_learned_model,

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

      learning_rate = args.learning_rate,
      network_seed = args.network_seed
  )
  saved_model_dir = args.dreamer_path
  if not saved_model_dir.startswith("/"):
    saved_model_dir = os.getcwd() + "/" + saved_model_dir
  saved_model_dir = saved_model_dir + f"/step_{args.model_restore_step}.pkl"
  if not os.path.exists(saved_model_dir):
    raise FileNotFoundError(f"The given Dreamer path {saved_model_dir} does not exist!")
  world_model = load_model(saved_model_dir)
  assert isinstance(world_model, DreamerMA), f"The world model is expected to be an instance of DreamerMA not {world_model.__class__}"
  game = world_model.game
  model_save_dir = args.model_save_dir
  game_name = game.game_name()
  empty = ""
  game_params = game.params_dict()
  params_str = f'{empty.join(f"_{value}" for key, value in game_params.items())}' 
  
  if not model_save_dir:
      model_save_dir = f"/trained_networks/rnad/{game_name}{params_str}/seed{trajectory_seed}/network_seed{network_seed}/"
      model_save_dir = os.getcwd() + model_save_dir
  model = RNaDDreamer(
      config=config,
      dreamer_model= world_model
  )
  model.train_model(model_save_dir, args.num_steps, args.print_each, args.save_each)

if __name__ == "__main__":
  main()