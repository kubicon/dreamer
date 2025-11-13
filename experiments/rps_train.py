
from games.jax_rps import JaxRPS, JaxStochasticRPS
from experiments.train_dreamer import train_model_ma
from experiments.rnad_train import train_rnad
from experiments.joint_train import joint_train_loop
from experiments.train_actor_critic import train_actor_critic
from experiments.parsing_utils import prepare_experiment_parser



parser = prepare_experiment_parser(multi_agent=True)
parser.add_argument("--stochastic", action="store_true", help="A flag whether to use the stochastic or standard RPS.")


def main():
  args = parser.parse_args()
  game = JaxStochasticRPS() if args.stochastic else JaxRPS()
  if args.train_mode == "dreamer":
    train_model_ma(args, game)
  elif args.train_mode == "rnad":
    train_rnad(args)
  elif args.train_mode == "actor_critic":
    train_actor_critic(args)
  else:
    joint_train_loop(args, game)
    
if __name__ == "__main__":
  main()