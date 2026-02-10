from games.jax_goofspiel import JaxGoofspiel
from experiments.joint_train import train
from experiments.parsing_utils import prepare_experiment_parser



parser = prepare_experiment_parser()
parser.add_argument("--num_cards", type=int, default=3, help="Number of cards of the Goofspiel game")


def main():
  args = parser.parse_args()
  game = JaxGoofspiel(cards=args.num_cards)
  train(args, game)
    
if __name__ == "__main__":
  main()