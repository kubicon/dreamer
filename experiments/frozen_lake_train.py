

from experiments.parsing_utils import prepare_experiment_parser
from experiments.parse_frozen_lake import parse_game
from experiments.joint_train import train


parser = prepare_experiment_parser()

parser.add_argument("--game_config_path", type=str, default="game_instances/frozen_lake3x3_(2,2)_50_0.1.txt", help="Path to the file containing description of the FrozenLake instance.")

def main():
  fl_game = parse_game(args.game_config_path)
  args = parser.parse_args()
  train(args, fl_game)

  
if __name__ == "__main__":
  main()