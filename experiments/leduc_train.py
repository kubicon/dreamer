
from games.jax_leduc import JaxLeduc
from experiments.train_dreamer import train_model_ma, create_dreamer_parser


parser = create_dreamer_parser(multi_agent=True)


def main():
  args = parser.parse_args()
  game = JaxLeduc()
  train_model_ma(args, game)

if __name__ == "__main__":
  main()