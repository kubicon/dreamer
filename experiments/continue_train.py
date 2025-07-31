from argparse import ArgumentParser
import os
import numpy as np

from dreamer import Dreamer
from train_utils import load_model, save_model

parser = ArgumentParser()
parser.add_argument("--model_dir", type=str, default="trained_networks/point_card_matching_3/seed99/network_seed42", help="Path to the directory of saved models")
parser.add_argument("--restore_step", type=int, default=1000, help="Saved step of the model to restore")

parser.add_argument("--num_steps", type=int, default=1001, help="Number of training steps")
parser.add_argument("--save_each", type=int, default=100, help="Save model every N steps")
parser.add_argument("--print_each", type=int, default=100, help="Print loss every N steps")


def main():
  args = parser.parse_args()
  model_path = args.model_dir
  if not model_path.startswith("/"):
    model_path = os.getcwd() + "/" + model_path
  model_path = model_path + f"/step_{args.restore_step}.pkl"
  if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} does not exist.")

  model = load_model(model_path)
  assert isinstance(model, Dreamer), "Loaded model is not an instance of Dreamer."
  assert model.game.game_name() in ["point_card_matching", "point_card_matching_stochastic"], f"Loaded model should be trained some point card matching game not {model.game.game_name()}"
  print(f"Restored model from {model_path}")

  model_save_dir = args.model_dir + "/"
  step = args.restore_step

  for i in range(args.num_steps):
    loss = model.world_model_train_step()
    if args.print_each > 0 and i % args.print_each == 0:
      print(f"Step {i + step}, Loss: {loss}")
    if args.save_each > 0 and i % args.save_each == 0:
      model_file = model_save_dir + f"step_{i + step}.pkl"
      save_model(model, model_file)


if __name__ == "__main__":
  main()