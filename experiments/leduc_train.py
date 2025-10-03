
from games.jax_leduc import JaxLeduc
from experiments.train_dreamer import train_model_ma
from experiments.rnad_train import train_rnad
from experiments.joint_train import joint_train_loop
from experiments.parsing_utils import prepare_experiment_parser



parser = prepare_experiment_parser(multi_agent=True)


def main():
  args = parser.parse_args()
  game = JaxLeduc()
  if args.train_mode == "dreamer":
    train_model_ma(args, game)
  elif args.train_mode == "rnad":
    train_rnad(args)
  else:
    joint_train_loop(args, game)
    
if __name__ == "__main__":
  main()