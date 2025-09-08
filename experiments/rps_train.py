
from games.jax_rps import JaxRPS
from experiments.train_dreamer import train_model_ma, create_dreamer_parser


parser = create_dreamer_parser(multi_agent=True)


def main():
  args = parser.parse_args()
  game = JaxRPS()
  train_model_ma(args, game)

if __name__ == "__main__":
  main()