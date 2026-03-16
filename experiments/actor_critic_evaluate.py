from argparse import ArgumentParser
import os
import time
import matplotlib.pyplot as plt

from experiments.policy_eval_utils import *
from experiments.eval_utils import track
from train_utils import load_model, get_seeds


parser = ArgumentParser()

parser.add_argument("--base_path", type=str, default="trained_networks", help="Path to the directory of saved models")
parser.add_argument("--game_name", type=str, default="goofspiel_3", help="Name and parameter string of the game to evaluate for")
parser.add_argument("--seeds", type=str, default='(42, )', help="Seeds of the stored models to check. Supplied as a string (seed_1, seed_2, ..., seed_n)")
parser.add_argument("--restore_step", type=int, default=10000, help="Saved step of the model to restore. If checking entire directory, -1 is also supported for all steps")

parser.add_argument("--scale_factor", type=float, default=1.0, help="Scale factor to multiply all rewards by. Useful if the game implementation scaled rewards in a different way than traditional implementations."
                    "Then, this should be the inverse of the game scaling factor. For example, JaxLeduc divides all rewards by 13, so to get values appriopriately scaled as in literature, this should be set to 13.")

experiment_parsers = parser.add_subparsers(dest="experiment_type", required=True, help="Which experiment type to run. Currently available are: loaded"
                                          "evaluate best responses against, or expected values of particular loaded model, or all models in the directory if restore_step is -1" \
                                          "nash: evaluate expected values of the model, best response values against it and also of a saved reference nash equilibrium strategy.")

loaded_parser = experiment_parsers.add_parser(name="loaded", help="Evaluate best responses against particular loaded model, or all models in the directory if restore_step is -1")
loaded_parser.add_argument("--metric", type=str, default="nash_conv", choices=("nash_conv", "expected_util", "env_return"), help="Type of metric to plot. Either NashConv, expected_utility, or smoothed environment returns during training.")

nash_parser = experiment_parsers.add_parser(name="nash", help="Evaluate expected values of the model, best response values against it and also of a saved reference nash equilibrium strategy.")
nash_parser.add_argument("--nash_strategy_path", type=str, default="experiments/goofspiel_nash.pkl", help="Path to the saved nash strategy in pickle format. Must be formatted as a tuple of behavioral strategies per tree depth and infoset map per tree_depth.")

   
def parse_env_returns(model_dir):
    """ Parse the environment return smoothed averages logged
    in the directory and prepare them into an array for plotting.
    Returns: (steps, return_values, game_str, smoothing_window)
    """
    return_file = model_dir + "/env_returns.txt"
    assert os.path.exists(return_file), f"File {return_file} does not exist!"
    with open(return_file, 'r') as f:
       lines = f.readlines()
    #The first line contains the game_string
    game_str = lines[0].strip()
    #Second line contains the smoothing window
    smoothing_window = int(lines[1].split(':')[1].strip())
    steps = []
    returns = []
    for l in lines[2:]:
       step_part, return_part = l.split(',')
       step, ret = [float(p.split(':')[1].strip()) for p in (step_part, return_part)]
       steps.append(step)
       returns.append(ret)
    return np.asarray(steps, dtype=np.int32), np.asarray(returns), game_str, smoothing_window
    

@track
def get_metrics_from_dir(model_dir, args):
    """
    Scans a directory for checkpoints and calculates either
    expected return or NashConv metrics.
    Returns: (steps, metrics)
    """
    metrics = []
    steps = []
    
    if not model_dir.startswith("/"):
        model_dir = os.path.join(os.getcwd(), model_dir)
        
    if not os.path.exists(model_dir):
        print(f"Skipping {model_dir} (Not found)")
        return None, None, None

    print(f"Starting evaluation for: {model_dir}")
    start_time = time.time()
    
    first = True
    model = None
    game = None
    
    # Get all .pkl files and sort them by step to avoid jumping around
    files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    
    for filename in files:
        step = int(filename.split("_")[-1].split(".")[0])

        model_path = os.path.join(model_dir, filename)

        if args.restore_step >=0 and (not step == args.restore_step):
            continue
        
        #try:
        if first:
            model = load_model(model_path)
            assert isinstance(model, DreamerMA), f"Expected DreamerMA, got {model.__class__}"
            
            # Initialize Game
            if not model.optimizer.model.use_real_infoset:
                game = DreamerModelGame(model)
            else:
                game = model.game
            first = False
        else:
            temp_model = load_model(model_path)
            nnx.update(model.optimizer, nnx.state(temp_model.optimizer))
            model.actor_critic.learner_steps = temp_model.actor_critic.learner_steps
            model.learner_steps = temp_model.learner_steps
            
            if not model.optimizer.model.use_real_infoset:
                game = DreamerModelGame(model)

        # Calculate Metric
        if args.metric == "nash_conv":
            metric = nash_conv(model, game)
            #jax.debug.breakpoint()
        else:
            model_map_and_behaviorals = extract_model_policy(model, game)
            metric, _ = policy_expected_value(game, model_map_and_behaviorals)
        
        metric = args.scale_factor * metric
        metrics.append(metric)
        steps.append(step)
            
        # except Exception as e:
        #     breakpoint()
        #     print(f"Failed to process {filename}: {e}")
        #     continue

    print(f"Evaluation for {model_dir} took {time.time() - start_time:.2f} seconds.")
    
    # Sort results
    steps = np.asarray(steps)
    metrics = np.asarray(metrics)
    if len(steps) > 0:
        sort_indices = np.argsort(steps)
        return steps[sort_indices], metrics[sort_indices], game
    else:
        return [], [], game


def plot_comparison(args):
    """
    Main function to plot Reinforce vs RNaD for a specific game/seed.
    """
    base_path = args.base_path
    game_path = args.game_name 
    seeds = get_seeds(args.seeds)
    seed_paths = [f"seed_{s}" for s in seeds]
    
    # Define the two algorithms to compare
    algos = {
        "Reinforce": [os.path.join(base_path, "reinforce", game_path, p) for p in seed_paths],
        "RNaD": [os.path.join(base_path, "rnad", game_path, p) for p in seed_paths]
    }
    
    results = {k: {} for k in algos}
    game_str = ""
    game = None
    smoothing_window = -1
    max_steps = 0
    
    def metrics_wrapper(directory, args):
        """Just a wrapper function to 
        handle the interface discrepancy between 
        the environment returns, and game theoretic metric
        computation."""
        if args.metric == 'env_return':
            steps, metrics, game_str, smoothing_window = parse_env_returns(directory)
            game = None
        else:
            steps, metrics, game = get_metrics_from_dir(d, args)
            game_str = str(game)
            smoothing_window = -1
        return steps, metrics, game, game_str, smoothing_window


    
    # 1. Collect Data
    for algo_name, dir_paths in algos.items():
        for s, d in zip(seeds,dir_paths):
            steps, metrics, game, new_game_str, new_smoothing_window = metrics_wrapper(d, args)
            if steps is not None and len(steps) > 0:
                max_steps = max(max_steps, len(steps))
                results[algo_name][s] = (steps, metrics)
                #Check if all experiments used the same game
                if not game_str:
                    game_str = new_game_str
                else:
                    assert new_game_str == game_str, f"Expected all models to use the same game {game_str}. Found {new_game_str} for algorithm {algo_name} seed {s} instead!"
                #Check if all experiments
                # used the same smoothing window
                # (relevant for the environment returns only) 
                if smoothing_window < 0:
                    smoothing_window = new_smoothing_window
                else:
                    assert smoothing_window == new_smoothing_window, f"Expected all models to use the same smoothing window {smoothing_window}. Found {new_smoothing_window} for algorithm {algo_name} seed {s} instead!"

    if not results:
        print("No data found for either algorithm.")
        return

    # 2. Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    x_name = "Gradient steps"

    if args.metric == 'nash_conv':
        metric_str = "NashConv"
        plot_str = "nash_conv"
    elif args.metric == 'expected_util':
        metric_str = "Expected Utility"
        plot_str = "expected_utility"
    else:
        metric_str = f"Env returns smoothed with a {smoothing_window} window"
        plot_str = f"env_return_window_{smoothing_window}"
        x_name = "Env steps"
    
    # Plot Algorithm Curves
    colors = {'Reinforce': 'tab:red', 'RNaD': 'tab:blue'}
    
    for algo_name, seed_data in results.items():
        if not seed_data:
            continue
            
        # seed_data is {seed: (steps, metrics)}
        # We need to aggregate them.
        
        # Assumption: All seeds have the same steps. 
        # If not, we take the intersection or reference the first one.
        first_seed = list(seed_data.keys())[0]
        ref_steps = seed_data[first_seed][0] 
        
        # Create list of metric arrays
        stacked_metrics = []
        for s, (steps, metrics) in seed_data.items():
            if np.array_equal(steps, ref_steps):
                stacked_metrics.append(metrics)
            else:
                print(f"Warning: Step mismatch for {algo_name} seed {s}. Skipping aggregation for this seed.")
        
        if not stacked_metrics:
            continue

        # Convert to matrix: (N_seeds, N_steps)
        matrix = np.vstack(stacked_metrics)
        
        # Calculate Statistics
        mean = np.mean(matrix, axis=0)
        
        # Plot Mean Line
        color = colors.get(algo_name, 'black')
        for i in range(matrix.shape[0]):
            ax.plot(ref_steps, matrix[i], 
                    color=color, 
                    alpha=0.3,       # Make it faint
                    linestyle='--',   # Dotted/Dashed line
                    linewidth=1)      # Thinner line

        # 2. Plot Mean Line
        # We plot this LAST so it appears on top of the individual seeds.
        # We add the label here so it appears in the legend once.
        ax.plot(ref_steps, mean, 
                label=algo_name, 
                color=color, 
                linestyle='-', 
                linewidth=2.5)    # Thicker, solid line

    # Plot Uniform Baseline (Dashed Line)
    if args.metric == "nash_conv" and game:
        #Using the overloaded functionality of extract model policy
        # to get uniform policy for the game
        uniform_policy = extract_model_policy(None, game, uniform=True)
        uniform_nash_conv = args.scale_factor * nash_conv(None, game, uniform_policy)
        ax.axhline(y=uniform_nash_conv, xmin=0, xmax=max_steps, color='orange', linestyle='--', label="Uniform Policy", alpha=0.7)

    # Styling
    ax.legend()
    ax.set_xlabel(x_name)
    ax.set_ylabel(metric_str)
    ax.set_title(f"NashDreamer {metric_str} on {args.game_name}")
    ax.grid(True, alpha=0.3)
    
    # if args.metric == "nash_conv":
    #     ax.set_yscale("log")

    # Save
    save_dir = f"plots/comparison/{game_path}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = f"{plot_str}_comparison.pdf"
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    print(f"Plot saved to {os.path.join(save_dir, filename)}")

def test_nash(args, saved_nash_path: str):
  model_path = args.model_dir
  if not model_path.startswith("/"):
    model_path = os.getcwd() + "/" + model_path
  model_path = model_path + f"/step_{args.restore_step}.pkl"
  if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} does not exist.")
  
  nash_path= saved_nash_path
  if not nash_path.startswith("/"):
    nash_path = os.getcwd() + "/" + nash_path
  if not os.path.exists(nash_path):
    raise FileNotFoundError(f"Nash policy file {nash_path} does not exist.")
  
  print(f"Evaluating policy of model loaded from {model_path} against nash policy loaded from {saved_nash_path}")

  model = load_model(model_path)
  # game = JaxLeduc()
  # buffer = ReplayBuffer(game, 0, 0, 100)
  # dreamer_model = DreamerMA(DreamerMAConfig(), buffer)
  # model = RNaDDreamerJoint(dreamer_model, RNaDConfig())
  assert isinstance(model, DreamerMA), f"The loaded model should be an instance of DreamerMA. Instead got {model.__class__}"
  
  game = DreamerModelGame(model) if not model.optimizer.model.is_iig else  model.game
  p1_nash_val, p2_nash_val, nash_infoset_map, nash_behaviorals = load_model(nash_path)
  print(f"Loaded nash policies of game with game value {p1_nash_val} (from player 1 perspective)")
  model_map, model_behaviorals = extract_model_policy(model, game)
  found_p1_nash, found_p2_nash = policy_expected_value(game, (nash_infoset_map, nash_behaviorals), eps=1e-5)
  found_p1_nash, found_p2_nash = args.scale_factor * found_p1_nash, args.scale_factor * found_p2_nash
  print(f"Found nash values: {found_p1_nash} {found_p2_nash}")
  assert np.isclose(found_p1_nash, p1_nash_val, atol=1e-5), f"Found nash value {found_p1_nash} and saved nash value {p1_nash_val} for player 1 differ!"
  assert np.isclose(found_p1_nash, p1_nash_val, atol=1e-5), f"Found nash value {found_p2_nash} and saved nash value {p2_nash_val} for player 2 differ!"
  p2_br_val, p1_br_val, p1_br, p2_br = model_best_response(model, game, (nash_infoset_map, nash_behaviorals))
  print(f"Found nash exploitabilities:")
  print(f"P2 best response value against p1: {p2_br_val}")
  print(f"P1 best response value against p2 {p1_br_val}")
  model_p1_val, model_p2_val = policy_expected_value(game, (model_map, model_behaviorals))
  model_p1_val, model_p2_val = args.scale_factor * model_p1_val, args.scale_factor * model_p2_val
  print(f"Model values {model_p1_val}, {model_p2_val}")
  p2_br_val, p1_br_val, p1_br, p2_br = model_best_response(model, game)
  p1_br_val, p2_br_val = args.scale_factor * p1_br_val, args.scale_factor * p2_br_val
  print(f"P2 best response value against p1: {p2_br_val}")
  print(f"P1 best response value against p2 {p1_br_val}")
  #compare_policies(model.world_model.game, (model_map, model_behaviorals), (nash_infoset_map, nash_behaviorals))
        
  

def main():
  args = parser.parse_args()
  if args.experiment_type == "nash":
    test_nash(args, saved_nash_path=args.nash_strategy_path)
  else:
    plot_comparison(args)
  

if __name__ == "__main__":
  main()