from argparse import ArgumentParser
import os
import numpy as np
import jax
import flax.nnx as nnx
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt

from dreamer_ma import DreamerMA
from ma_rssm import MARSSM, symlog
from train_utils import load_model, RNaDConfig

from experiments.tree_view_utils import *
from experiments.eval_utils import *


parser = ArgumentParser()
parser.add_argument("--model_dir", type=str, default="trained_networks/joint/point_card_matching_mp_3/seed_42", help="Path to the directory of saved models")
parser.add_argument("--restore_step", type=int, default=-1, help="Saved step of the model to restore. If -1, checks all models within that folder.")

parser.add_argument("--verbose", action="store_true", help="A flag whether to also print information about states being checked")
parser.add_argument("--render_tree", action="store_true", help="A flag whether to create the model EFG-style tree and render it.")



@chex.dataclass
class WalkCarry:
  legals: chex.Array
  game_state: GameState
  obs: chex.Array
  recurrent_state: chex.Array
  stoch_state:chex.Array
  deter_state: chex.Array
  joint_latent_infoset: chex.Array
  reward: chex.Array
  terminal: chex.Array
  after_chance: chex.Array

def check_state_one_outcome(model: DreamerMA, carry:WalkCarry, eps:float, verbose = False):
  """Check whether the best fitting deterministic state for the state
  produces valid results. Used for post-chance node states, to check 
  whether it corresponds to the correct outcome."""
  mistake_probs = np.zeros(5)
  differences = np.zeros(5)
  ma_rssm = model.optimizer.model
  decoded_obs = ma_rssm.get_decoder_no_jit(carry.recurrent_state, carry.deter_state)
  p1_decoded_obs, p2_decoded_obs = decoded_obs[0], decoded_obs[1]
  pred_reward, pred_terminal, pred_legal = ma_rssm.get_predictor(carry.recurrent_state, carry.deter_state)
  p1_obs_max_difference = jnp.max(jnp.abs(carry.obs[0] - p1_decoded_obs))
  p2_obs_max_difference = jnp.max(jnp.abs(carry.obs[1] - p2_decoded_obs))
  reward_difference = jnp.abs(carry.reward - pred_reward)
  legal_diference = not carry.terminal and jnp.any(pred_legal != carry.legals)
  det_prob = jnp.prod(carry.stoch_state[carry.deter_state.astype(jnp.bool)])

  differences[0] = p1_obs_max_difference
  if p1_obs_max_difference >= eps:
    mistake_probs[0] = det_prob
    if verbose:
      print(f"Real obs and decoded obs for player 1 differ by more than {eps}.")
      print(f"Max difference {p1_obs_max_difference}")
      print(f"Real obs: {carry.obs[0]}")
      print(f"Decoded obs: {p1_decoded_obs}")
  differences[1] = p2_obs_max_difference
  if p2_obs_max_difference >= eps:
    mistake_probs[1] = det_prob
    if verbose:
      print(f"Real obs and decoded obs for player 2 differ by more than {eps}.")
      print(f"Max difference {p2_obs_max_difference}")
      print(f"Real obs: {carry.obs[1]}")
      print(f"Decoded obs: {p2_decoded_obs}")
  differences[2] = int(pred_terminal != carry.terminal)
  if pred_terminal != carry.terminal:
    mistake_probs[2] = det_prob
    if verbose:
      print(f"Predicted terminal {pred_terminal} does not match real terminal {carry.terminal}. ")
  differences[3] = reward_difference
  if reward_difference >= eps:
    mistake_probs[3] = det_prob
    if verbose:
      print(f"Predicted reward {pred_reward} differs from real reward {carry.reward} by more than {eps}.")
  #Do not check legal actions in terminal states
  differences[4] = int(legal_diference)
  if legal_diference:
    mistake_probs[4] = det_prob
    if verbose:
      print(f"Predicted legal actions {pred_legal} do not match real legal actions {carry.legals}.")
  #Ordered p1_obs, p2_obs, terminal, reward, legals
  #print(f"Mistake probs {mistake_probs}")
  return mistake_probs, differences


def model_walk_test(model:DreamerMA,
                     difference_eps = 0.2, probability_eps = 0.05, probability_threshold=0.05
                     , verbose=False, visualise_tree = False):
  """Walk through the entire game tree in each state, check
  all learned outcomes where the individual components of the deterministic
  state have probability outcome over probability eps. Then perform a tree based
  expansion of all these model states and check whether the model learned well enough in each.
  In case of a chance node, the next model states are clustered to the particular
  outcome based on the closeness of their decoder produced output to the real observation.

   Probability eps is used to control which outcomes under the learned policy to ignore
   (if the action component had pbt <= probability eps for either player, it will not be expanded), difference eps
   is used as a threshold of absolute difference, where mistake is reported (for real
   predictions. For boolean a mistake is always reported on a mismatch). Finally, probablity
   threshold is used to control which of the model states are expanded. Those where any
   component has pbt < than this threshold are ignored.

   Cannot handle more than 1 consecutive chance nodes (but note that 
   these can be represented as a single chance node.)
   Returns a numpy array of statistics of probablity of mistakes averaged over the states.
   For a single agent Dreamer ordered as obs_reconstruction, terminal, reward
   And for a multi agent Dreamer as obs1_reconstruction, obs2_reconstruction, terminal, reward, legal_actions.
  """
  def get_both_obs(state: GameState):
    _, p1_obs, p2_obs, _ = model.game.get_info(state)
    return jnp.stack([p1_obs, p2_obs], axis=0)
  use_real_infoset = model.use_real_infoset

  get_obs_fn = get_both_obs 
  #get_closest_deter_fn = get_closest_deter_ma if is_ma else get_closest_deter
  num_players = model.game.num_players()
  mistake_probs = 0
  visited = {}
  vectorized_get_obs = jax.vmap(get_both_obs, in_axes=(0), out_axes=(0))
  model_tree_root = Node("", data={"type": PAST_ACTION, "action": -1}) if visualise_tree else  None
  ma_rssm = model.optimizer.model
  def get_stoch_from_prediction(logits: chex.Array):
    stoch_unfiltered = np.asarray(jax.nn.softmax(logits, axis=-1))
    stoch_unnormalized = stoch_unfiltered * (stoch_unfiltered >= probability_threshold)
    stoch = stoch_unnormalized / np.sum(stoch_unnormalized, axis=-1, keepdims=True)
    return stoch


  def _tree_walk(carry: WalkCarry, depth=0, reach_probability:float = 1.0, action_outcome_history = "",
                 subtree_parent: Node = None, outcome:int = -1, outcome_prob: float = 0, create_model_node:bool = False):
    nonlocal mistake_probs

    visited[action_outcome_history] = True
    #print(f"Num visited states {num_visited_states}")
    # if carry.terminal:
    #   mistake_cum_probs = mistake_cum_probs + all_outcome_check_fn(model, carry, difference_eps, probability_threshold, verbose)
    #   return
    # else:

    state_mistake_probs, state_differences = check_state_one_outcome(model, carry, difference_eps, verbose)
    parent = subtree_parent
    if visualise_tree and create_model_node:
      deter_path = parent.name + f"d{outcome}"
      #print(f"Storing node with id {deter_path} and parent {parent.name}")
      deter_node = Node(deter_path,
                      parent = parent,
                      data = {"differences" : state_differences, 
                              "prob": outcome_prob,
                              "type": MODEL_NODE})
      parent = deter_node
    mistake_probs  = mistake_probs + (state_mistake_probs * reach_probability)
    if carry.terminal:
      return
    policy_obs = symlog(carry.obs) if use_real_infoset else ma_rssm.get_infoset(carry.recurrent_state, carry.deter_state, carry.joint_latent_infoset)
    pi = np.asarray(ma_rssm.get_policy_both(policy_obs, carry.legals, use_symlog=False))
    if verbose:
      print(f"Checking state {carry.game_state}")
      print(f"Reach probs {reach_probability}")
      print(f"Policy: {pi}")
    pi_mask = pi >= probability_eps
    actions = np.tile(np.arange(pi.shape[-1]), (num_players,1)).reshape(pi.shape)
    valid_actions = [actions[i][pi_mask[i]] for i in range(num_players)]
    actions = cartesian_product(*valid_actions)
    for a in actions:
      action_prob = np.prod(pi[np.arange(a.shape[0]), a])
      action_parent = parent
      if visualise_tree:
        action_path = parent.name + f"a{a}"
        action_node = Node(action_path,
                         parent = action_parent,
                         data = {"type": PAST_ACTION, "action": a, "prob": action_prob})
        action_parent = action_node
      next_state, next_terminal, next_reward, next_legals = model.game.apply_action(carry.game_state, a)
      ai_oh = jax.nn.one_hot(a, carry.legals.shape[-1])
      next_recurrent = ma_rssm.get_next_recurrent(carry.recurrent_state, carry.deter_state, ai_oh)
      dyn = ma_rssm.get_dynamics(next_recurrent)
      next_stoch_state = get_stoch_from_prediction(dyn)
            
      is_chance = model.game.is_chance(next_state)
      chance_outcomes = model.game.depth_chance_valid_outcomes(depth + 1)
      if is_chance:
        next_states, next_terminals, next_rewards, next_legals, next_probs = unroll_chance_node(model.game, next_state, chance_outcomes) 

        next_terminals = np.asarray(next_terminals)
        next_rewards = np.asarray(next_rewards)
        next_legals = np.asarray(next_legals)
      else:
        next_states =jax.tree.map(lambda x: x[None, ...], next_state)

        next_terminals = np.asarray(next_terminal)[None, ...]
        next_rewards = np.asarray(next_reward)[None, ...]
        next_legals = np.asarray(next_legals)[None, ...]
      next_obs = vectorized_get_obs(next_states)
      next_obs = np.asarray(next_obs)
      next_deters, next_probs= get_next_outcomes(model, next_stoch_state, next_recurrent, next_obs, probability_eps)
      #print(f"Next deters: {next_deters}")
      for i in range(next_terminals.shape[0]):
        outcome_parent = action_parent
        next_terminal = next_terminals[i] 
        next_joint_infoset = ma_rssm.get_next_infoset_all(carry.joint_latent_infoset, next_obs[i], ai_oh)
        next_reward = next_rewards[i]
        next_legal = next_legals[i]
        next_state = jax.tree.map(lambda x: x[i], next_states)
        single_outcome_deters = next_deters[i]
        single_outcome_probs = next_probs[i]
        outcome_prob = np.sum(single_outcome_probs)
        if next_terminals.shape[0] > 1  and visualise_tree:
          outcome_path = parent.name + f"o{i}"
          outcome_node = Node(outcome_path,
                          parent=outcome_parent,
                          data = {"prob": outcome_prob, "type": PAST_CHANCE})
          outcome_parent = outcome_node
        for j, deter in enumerate(single_outcome_deters):
          new_carry = WalkCarry(legals= next_legal,
                                obs = next_obs[i],
                                game_state = next_state,
                                recurrent_state= next_recurrent,
                                stoch_state=next_stoch_state,
                                deter_state=deter,
                                joint_latent_infoset=next_joint_infoset,
                                reward=next_reward,
                                terminal=next_terminal,
                                after_chance=is_chance)
          prob = jnp.prod(next_stoch_state[deter.astype(jnp.bool)])  
          _tree_walk(new_carry, depth = depth+ 1 + int(is_chance), subtree_parent = outcome_parent,
                     action_outcome_history= action_outcome_history + f"a{a}o{i}",
                     reach_probability= reach_probability * prob,
                     outcome = j, outcome_prob=single_outcome_probs[j] / outcome_prob,
                     create_model_node=True)
  
  init_state, init_legals = model.game.initialize_structures()
  init_recurrent = ma_rssm.get_init_recurrent()
  init_chance =  model.game.is_chance(init_state)
  per_player_init_infoset = MARSSM.vmap_over_net(ma_rssm.infoset_network, in_axes=[(None, 0, None)], out_axes=[(0)])
  dummy_infoset = jnp.zeros((ma_rssm.infoset_size))
  dummy_action = jnp.zeros((model.action_dimension))
  if init_chance:
    chance_outcomes = model.game.depth_chance_valid_outcomes(0)
    next_states, next_terminals, next_rewards, next_legals, next_probs = unroll_chance_node(model.game, init_state, chance_outcomes)
        

    next_terminals = np.asarray(next_terminals)
    next_rewards = np.asarray(next_rewards)
    next_legals = np.asarray(next_legals)
    for i in range(next_terminals.shape[0]):
      outcome_parent = model_tree_root
      next_terminal = next_terminals[i]
      next_reward = next_rewards[i]
      next_legal = next_legals[i]
      next_state = jax.tree.map(lambda x: x[i], next_states)
      #This is a special case handled differently than the
      # chance nodes from dynamics, which share a stochastic state
      # and we just pick the deterministic states most likely
      # beloning to the outcome.
      # The first prediction is posterior, so each outcome has its own stochastic state
      # because they are differentiated by the observations.
      init_obs = get_obs_fn(next_state)
      #print(f"Init obs for outcome {i}, is {init_obs}")
      # if verbose:
      #   print(f"Checking state {next_state}")
      init_stoch_state = get_stoch_from_prediction(ma_rssm.get_encoder(init_recurrent, init_obs))
      init_deters, init_probs = get_next_outcomes(model, init_stoch_state, init_recurrent, init_obs, probability_eps)
      init_deters = init_deters[0]
      init_probs = init_probs[0]
      outcome_path = f"o{i}"
      outcome_prob = np.sum(init_probs)
      if visualise_tree:
        init_chance = Node(outcome_path,
                         parent= outcome_parent,
                         data = {"prob": outcome_prob, "type": PAST_CHANCE})
        outcome_parent = init_chance
      #num_init_deters = len(init_deters)
      for j, deter in enumerate(init_deters):
        init_carry = WalkCarry(legals= next_legal,
                              obs = init_obs,
                              game_state = next_state,
                              recurrent_state= init_recurrent,
                              stoch_state=init_stoch_state,
                              deter_state=deter,
                              joint_latent_infoset= per_player_init_infoset(dummy_infoset, init_obs, dummy_action),
                              reward=next_reward,
                              terminal=next_terminal,
                              after_chance=init_chance)
        _tree_walk(init_carry, depth=1,
                   action_outcome_history=f"o{i}",
                   subtree_parent = outcome_parent,
                     outcome = j, outcome_prob=init_probs[j] / outcome_prob,
                     create_model_node=True)
    num_visited_states = len(visited)
    avg_mistake_probs = mistake_probs / num_visited_states
    avg_mistake_probs = np.minimum(avg_mistake_probs, 1.0)
    if visualise_tree:
      render_tree(model_tree_root, model)
    return avg_mistake_probs
  init_obs = get_obs_fn(init_state)
  init_stoch_state = get_stoch_from_prediction(ma_rssm.get_encoder(init_recurrent, init_obs))
  #The simplest 
  if verbose:
    print(f"Checking state {init_state}")
  init_deters, init_probs = get_next_outcomes(model, init_stoch_state, init_recurrent, init_obs, probability_eps)
  init_deters = init_deters[0]
  init_probs = init_probs[0]
  #num_init_deters = len(init_deters)
  for i, deter in enumerate(init_deters):
    init_carry = WalkCarry(legals= init_legals,
                              obs=init_obs,
                              game_state = init_state,
                              recurrent_state= init_recurrent,
                              stoch_state=init_stoch_state,
                              deter_state=deter,
                              joint_latent_infoset=per_player_init_infoset(dummy_infoset, init_obs, dummy_action),
                              reward=jnp.array(0),
                              terminal=jnp.array(False),
                              after_chance=jnp.array(False))
    prob = jnp.prod(init_stoch_state[deter.astype(jnp.bool)])
    _tree_walk(init_carry, subtree_parent = model_tree_root,  
                    reach_probability=prob,
                     outcome = i, outcome_prob=init_probs[i],
                     create_model_node=True)
  num_visited_states = len(visited)
  avg_mistake_probs = mistake_probs / num_visited_states
  avg_mistake_probs = np.minimum(avg_mistake_probs, 1.0)
  if visualise_tree:
    render_tree(model_tree_root, model)
  return avg_mistake_probs

def main():
  args = parser.parse_args()
  model_dir = args.model_dir
  all_mistake_probs = []
  steps = []
  if not model_dir.startswith("/"):
    model_dir = os.getcwd() + "/" + model_dir
  if not os.path.exists(model_dir):
      raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
  #profiler = Profiler()
  print("Starting evaluation")
  start_time = time.time()
  #profiler.start()
  first = True
  model = None
  plot_subdir_str = "actor_critic"
  for filename in os.listdir(model_dir):
    if not os.path.isfile(os.path.join(model_dir, filename)):
      continue
    name, filetype = filename.split(".")
    if not filetype == "pkl":
      continue
    step = int(name.split("_")[-1])

    if not (args.restore_step == -1 or step == args.restore_step):
      continue

    model_path = model_dir + "/"  + filename
    
    #This assumes all the models were trained with the same config
    # else it will break
    if first:
      model = load_model(model_path)
      assert isinstance(model, DreamerMA), f"The saved model should be an instance of DreamerMA, instead got {model.__class__}"
      if isinstance(model.ac_config, RNaDConfig):
        plot_subdir_str = "rnad"
      first=False
    else:
      temp_model = load_model(model_path)
      assert isinstance(temp_model, DreamerMA), f"The saved model should be an instance of DreamerMA, instead got {temp_model.__class__}"
      #TODO: Updating this way still forces retracing of get_info and
      # initialize_structures of the game, since it is called in init. In general
      # we just need the state of the optimizers object from the model
      # and the rest of the operations are redundant.
      nnx.update(model.optimizer, nnx.state(temp_model.optimizer))
      #model.optimizers = model.update_nnx(model.optimizers, nnx.split(temp_model.optimizers)[1])

    print(f"Restored model from {model_path}")
    #breakpoint()
    mistake_probs = model_walk_test(model,
                    verbose=args.verbose,
                    visualise_tree=args.render_tree)
    all_mistake_probs.append(mistake_probs)
    steps.append(step)
    # print(f"Mistake probs {mistake_probs}")
    # print(f"Player 1 obs average mistake probability {mistake_probs[0]}")
    # print(f"Player 2 obs average mistake probability {mistake_probs[1]}")
    # print(f"Terminal average mistake probability {mistake_probs[2]}")
    # print(f"Reward average mistake probability {mistake_probs[3]}")
    # print(f"Legal actions average mistake probability {mistake_probs[4]}")
  print("Ended evaluation")
  print(f"Evaluation took {time.time() - start_time:.2f} seconds.")
  #profiler.stop()
  #print(profiler.output_text(color=True, unicode=True))
  if len(all_mistake_probs) == 0:
    raise FileNotFoundError(f"Model directory {model_dir} and restore step {args.restore_step}. Did not find any file. Make sure"
                            " the directory contains a file in a form of step_restore_step.pkl, "
                            "where restore_step is either the specified number, or arbitrary integer if -1.")
  all_mistake_probs = np.asarray(all_mistake_probs)
  steps = np.asarray(steps)
  sort_indices = np.argsort(steps)
  sorted_mistake_probs = all_mistake_probs[sort_indices]
  sorted_steps = steps[sort_indices]

  fig, ax = plt.subplots()
  ax.plot(sorted_steps, sorted_mistake_probs[:, 0], label="Player 1 obs")
  ax.plot(sorted_steps, sorted_mistake_probs[:, 1], label="Player 2 obs")
  ax.plot(sorted_steps, sorted_mistake_probs[:, 2], label="Terminal")
  ax.plot(sorted_steps, sorted_mistake_probs[:, 3], label="Reward")
  ax.plot(sorted_steps, sorted_mistake_probs[:, 4], label="Legal actions")
  ax.legend()
  ax.set_xlabel("Training step")
  ax.set_ylabel("Average mistake probability")
  ax.set_title("Evaluation of DreamerMA model")
  empty = ""
  game_params = model.game.params_dict()
  params_str = f'{empty.join(f"_{value}" for key, value in game_params.items())}'
  plt_dir = f"plots/mistake_probs/{plot_subdir_str}"
  if not os.path.exists(plt_dir):
    os.makedirs(plt_dir)
  plt.savefig(f"{plt_dir}/{model.game.game_name()}{params_str}.pdf")
  print(f"Saved plot at {plt_dir}/{model.game.game_name()}{params_str}.pdf")

if __name__ == "__main__":
  main()