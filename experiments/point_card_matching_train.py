from argparse import ArgumentParser

from games.point_card_matching import PointCardMatching, PointCardMatchingStochastic
from experiments.parsing_utils import prepare_experiment_parser

from experiments.train_dreamer import train_model

parser = prepare_experiment_parser(multi_agent=False)
# Game parameters
parser.add_argument("--num_cards", type=int, default=3, help="Number of cards of the game. Should be at least 3")
parser.add_argument("--stochastic", type=bool, default=False, help="Whether to use the point card matching with chance node, or the one without.")
parser.add_argument("--chance_turn_before_terminal", type=int, default=1, help="How many turns before terminal does the chance turn happen. Only for stochastic point card matching.")



def main():
  args = parser.parse_args()
  game = PointCardMatchingStochastic(args.num_cards, chance_turn_before_terminal=args.chance_turn_before_terminal) if args.stochastic else PointCardMatching(args.num_cards)
  train_model(args, game)

if __name__ == "__main__":
  main()