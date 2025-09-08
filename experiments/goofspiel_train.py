from argparse import ArgumentParser
from games.jax_goofspiel import JaxGoofspiel
from experiments.train_dreamer import train_model_ma, create_dreamer_parser


parser = create_dreamer_parser(multi_agent=True)
# Game parameters
parser.add_argument("--num_cards", type=int, default=3, help="Number of cards of the game. Should be at least 3") 

def main():
  args = parser.parse_args()
  game = JaxGoofspiel(cards = args.num_cards)
  train_model_ma(args, game)

if __name__ == "__main__":
  main()