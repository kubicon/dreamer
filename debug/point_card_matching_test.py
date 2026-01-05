from games.point_card_matching import PointCardMatching, PointCardMatchingStochastic
from games.jax_game import JaxGame, GameState
import jax
import jax.numpy as jnp
import numpy as np
from argparse import ArgumentParser
from functools import partial

parser = ArgumentParser()
parser.add_argument("--num_cards", type=int, default=3, help="Number of cards of the game")
parser.add_argument("--chance_turn_before_terminal", type=int, default=1, help="How many turns before terminal does the chance turn happen. Only for stochastic point card matching.")
parser.add_argument("--gameplay_seed", type=int, default=-1, help="Seed used for gameplay. If -1 a random seed instead")
parser.add_argument("--stochastic", type=bool, default=False, help="Whether to use the stochastic or deterministic version of point card matching.")


@partial(jax.jit, static_argnums=(0, 2))
def unroll_chance_node(game: JaxGame, game_state: GameState, num_chance_outcomes:int):
  outcomes, probs = game.get_outcomes_and_probs(game_state)
  vectorized_apply = jax.vmap(game.apply_action, in_axes=(None, 0), out_axes=(0, 0, 0, 0))
  next_states, next_terminal, rewards, next_legal = vectorized_apply(game_state, outcomes)
  valid = jnp.nonzero(probs, size=num_chance_outcomes)[0]
  next_states = jax.tree_util.tree_map(lambda x: jnp.take_along_axis(x, jnp.expand_dims(valid, axis=range(1, x.ndim)), axis=0), next_states)
  next_terminal = jnp.take_along_axis(next_terminal, jnp.expand_dims(valid, axis=range(1, next_terminal.ndim)), axis=0)
  rewards = jnp.take_along_axis(rewards, jnp.expand_dims(valid, axis=range(1, rewards.ndim)), axis=0)
  next_legal = jnp.take_along_axis(next_legal, jnp.expand_dims(valid, axis=range(1, next_legal.ndim)), axis=0)
  return next_states, next_terminal, rewards, next_legal

def tree_walk_test(game: JaxGame):
  def _tree_walk(state, terminal, reward, legals, depth=0):
    legals = np.asarray(legals)
    print(f"State: {state}")
    print(f"Legals {legals}")
    print(f"Reward {reward}")
    print(f"Terminal {terminal}")
    if game.is_chance(state):
        print("Chance node encountered.")
        next_states, next_terminals, next_rewards, next_legals = unroll_chance_node(game,state, game.depth_chance_valid_outcomes(depth))
        next_terminals = np.asarray(next_terminals)
        next_rewards = np.asarray(next_rewards)
        next_legals = np.asarray(next_legals)
          
        for i in range(next_terminals.shape[0]):
          next_terminal = next_terminals[i]
          next_reward = next_rewards[i]
          next_legal = next_legals[i]
          next_state = jax.tree_util.tree_map(lambda x: x[i], next_states)
          _tree_walk(next_state, next_terminal, next_reward, next_legal, depth+1)
        return
    if terminal:
      return
    for ai, a in enumerate(legals):
      if a < 0.5:
        continue
      print(f"Applying action {ai}")
      next_state, next_terminal, next_reward, next_legals = game.apply_action(state, ai)
      _tree_walk(next_state, next_terminal, next_reward, next_legals, depth + 1)
  state, legals = game.initialize_structures()
  _tree_walk(state, False, 0 , legals)

  


def main():
  args = parser.parse_args()
  gameplay_seed = args.gameplay_seed
  if gameplay_seed == -1:
    gameplay_seed = np.random.randint(0, 2**32 - 1)
  game = PointCardMatchingStochastic(args.num_cards, chance_turn_before_terminal=args.chance_turn_before_terminal) if args.stochastic else PointCardMatching(args.num_cards)
  tree_walk_test(game)

if __name__ == "__main__":
  main()