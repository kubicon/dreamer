
from games.jax_leduc import JaxLeduc
from experiments.joint_train import  train
from experiments.parsing_utils import prepare_experiment_parser



parser = prepare_experiment_parser()


def main():
  args = parser.parse_args()
  game = JaxLeduc()
  train(args, game)
    
if __name__ == "__main__":
  main()