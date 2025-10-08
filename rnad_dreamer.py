

from typing import Sequence, Any, Tuple
import jax
import jax.numpy as jnp
import jax.lax as lax

import flax.nnx as nnx
import chex
import optax

import numpy as np
import os


from functools import partial

from dreamer_ma import DreamerMA, DreamerMAOptimizers
from networks import initialize_rnad_optimizers, initialize_joint_optimizers, RNaDOptimizers, JointOptimizers, RNaDNetwork, IsetDecoder, Predictor, LegalActionsNetwork, DynamicsPredictor, SequenceModel, JointIsetEncoder
from train_utils import RNaDConfig, RNaDTimeStep, load_model, save_model
from distributions import sample_categorical
from games.jax_game import GameState

  


"""BEGGINING OF CODE FROM OpenSpiel RNaD"""
"""The Entropy schedule class taken from the
(now removed) RNaD implementation in the OpenSpiel library"""
class EntropySchedule:
  """An increasing list of steps where the regularisation network is updated.

  Example
    EntropySchedule([3, 5, 10], [2, 4, 1])
    =>   [0, 3, 6, 11, 16, 21, 26, 36]
          | 3 x2 |      5 x4     | 10 x1
  """

  def __init__(self, *, sizes: Sequence[int], repeats: Sequence[int]):
    """Constructs a schedule of entropy iterations.

    Args:
      sizes: the list of iteration sizes.
      repeats: the list, parallel to sizes, with the number of times for each
        size from `sizes` to repeat.
    """
    try:
      if len(repeats) != len(sizes):
        raise ValueError("`repeats` must be parallel to `sizes`.")
      if not sizes:
        raise ValueError("`sizes` and `repeats` must not be empty.")
      if any([(repeat <= 0) for repeat in repeats]):
        raise ValueError("All repeat values must be strictly positive")
      if repeats[-1] != 1:
        raise ValueError("The last value in `repeats` must be equal to 1, "
                         "ince the last iteration size is repeated forever.")
    except ValueError as e:
      raise ValueError(
          f"Entropy iteration schedule: repeats ({repeats}) and sizes"
          f" ({sizes})."
      ) from e

    schedule = [0]
    for size, repeat in zip(sizes, repeats):
      schedule.extend([schedule[-1] + (i + 1) * size for i in range(repeat)])

    self.schedule = np.array(schedule, dtype=np.int32)

  def __call__(self, learner_step: int) -> Tuple[float, bool]:
    """Entropy scheduling parameters for a given `learner_step`.

    Args:
      learner_step: The current learning step.

    Returns:
      alpha: The mixing weight (from [0, 1]) of the previous policy with
        the one before for computing the intrinsic reward.
      update_target_net: A boolean indicator for updating the target network
        with the current network.
    """

    # The complexity below is because at some point we might go past
    # the explicit schedule, and then we'd need to just use the last step
    # in the schedule and apply the logic of
    # ((learner_step - last_step) % last_iteration) == 0)

    # The schedule might look like this:
    # X----X-------X--X--X--X--------X
    # learner_step | might be here ^    |
    # or there     ^                    |
    # or even past the schedule         ^

    # We need to deal with two cases below.
    # Instead of going for the complicated conditional, let's just
    # compute both and then do the A * s + B * (1 - s) with s being a bool
    # selector between A and B.

    # 1. assume learner_step is past the schedule,
    #    ie schedule[-1] <= learner_step.
    last_size = self.schedule[-1] - self.schedule[-2]
    last_start = self.schedule[-1] + (
        learner_step - self.schedule[-1]) // last_size * last_size
    # 2. assume learner_step is within the schedule.
    start = jnp.amax(self.schedule * (self.schedule <= learner_step))
    finish = jnp.amin(
        self.schedule * (learner_step < self.schedule),
        initial=self.schedule[-1],
        where=(learner_step < self.schedule))
    size = finish - start

    # Now select between the two.
    beyond = (self.schedule[-1] <= learner_step)  # Are we past the schedule?
    iteration_start = (last_start * beyond + start * (1 - beyond))
    iteration_size = (last_size * beyond + size * (1 - beyond))

    update_target_net = jnp.logical_and(
        learner_step > 0,
        jnp.sum(learner_step == iteration_start + iteration_size - 1),
    )
    alpha = jnp.minimum(
        (2.0 * (learner_step - iteration_start)) / iteration_size, 1.0)
    return alpha, update_target_net
  
"""END OF CODE FROM OpenSpiel RNaD"""



  

def _policy_ratio(pi: chex.Array, mu: chex.Array, actions_oh: chex.Array, valid: chex.Array) -> chex.Array: 
  pi_actions_prob = jnp.sum(pi * actions_oh, axis=-1, keepdims=True) * valid + (1 - valid)
  mu_actions_prob = jnp.sum(mu * actions_oh, axis=-1, keepdims=True) * valid + (1 - valid)
  
  return pi_actions_prob / mu_actions_prob
  
def tree_where(pred: chex.Array, x: chex.ArrayTree, y: chex.ArrayTree) -> chex.ArrayTree:
  
  def _where(x, y):
    return jnp.where(pred, x, y)
  
  return jax.tree.map(_where, x, y)
  
def apply_force_with_threshold(decision_outputs: chex.Array, force: chex.Array,
                               threshold: float,
                               threshold_center: chex.Array) -> chex.Array:
  """Apply the force with below a given threshold."""
  chex.assert_equal_shape((decision_outputs, force, threshold_center))
  can_decrease = decision_outputs - threshold_center > -threshold
  can_increase = decision_outputs - threshold_center < threshold
  force_negative = jnp.minimum(force, 0.0)
  force_positive = jnp.maximum(force, 0.0)
  clipped_force = can_decrease * force_negative + can_increase * force_positive
  return decision_outputs * lax.stop_gradient(clipped_force)



def neurd_loss(
  logits: chex.Array,
  policy: chex.Array,
  q_values: chex.Array, 
  legal: chex.Array,
  importance_sampling: chex.Array,
  clip: float=10_000,
  threshold: float=2.0
):
  advantage = q_values - jnp.sum(policy * q_values, axis=-1, keepdims=True)
  advantage = advantage * importance_sampling
  advantage = lax.stop_gradient(jnp.clip(advantage, -clip, clip))
  mean_logit = jnp.sum(logits * legal, axis=-1, keepdims=True) / jnp.sum(legal, axis=-1, keepdims=True)
  
  logits_shifted = logits - mean_logit
  threshold_ceter = jnp.zeros_like(logits_shifted)
  
  neurd_loss_value = jnp.sum(legal * apply_force_with_threshold(logits_shifted, advantage, threshold, threshold_ceter), axis=-1, keepdims=True)
  
  return neurd_loss_value

def v_trace(
  v: chex.Array,
  valid: chex.Array,
  sampling_policy: chex.Array,
  network_policy: chex.Array,
  regularization_term: chex.Array,
  action_oh: chex.Array,
  reward: chex.Array, # Still not regularized
  lambda_: float = 1.0, # Lambda parameter for V-trace
  c: float = 1.0, # Importance sampling clipping
  rho: float = 1.0, # Importance sampling clipping
  eta: float = 0.2, # Regularization factor for reward transformation
  entropy_eta: float = 0.2, # Regularization factor for the additional KL-regularization  
  gamma: float = 1.0 # Discount factor
):
  
  importance_sampling = _policy_ratio(network_policy, sampling_policy, action_oh, valid)
  
  # The reason we use this is to ensure this is weighted by the amount of the times we sample it
  inverted_sampling = _policy_ratio(jnp.ones_like(sampling_policy), sampling_policy, action_oh, valid)
  
  #[Trajectory, Batch, Player]
  #This actually computes KL-divergence from the reference policy, despite being called entropy.
  #The reason for being called "entropy", is because it serves simliar purpose.
  #More on that below
  regularization_entropy = entropy_eta * jnp.sum(network_policy * regularization_term, axis=-1)
  weighted_regularization_term = -eta * regularization_term
  
  #[Trajectory, Batch]
  #For value estimates. Adding opponents KL divergence
  # amounts to: the value of this state is higher, because the opponent
  # did something surprising, hence, the state should be explored more, to either
  # find out a possible opponent mistake, or confirm that the state is bad.
  # similarly, subtracting our own KL-divergence penalizes being too "surprising"
  # with regards to the reference policy, to discourage erratic policies
  both_player_entropy = (regularization_entropy[..., 1] - regularization_entropy[..., 0])

  #[Trajectory, Batch]
  entropy_reward = reward + both_player_entropy
  #[Trajectory, Batch, Player, 1]
  entropy_reward = jnp.expand_dims(jnp.stack((entropy_reward, -entropy_reward), axis=-1), -1)
  
  #[Trajectory, Batch, Player]
  # Once again, similar logic. Adding opponents KL divergence
  # to give a bonus to the rewards if the opponent did something surprising there
  # to either confirm that it is not good for us, or take advantage of it.
  q_reward = jnp.stack((reward, -reward), axis=-1) + regularization_entropy[..., (1, 0)]
  
  q_reward = jnp.expand_dims(q_reward, -1)
  
  
  
  @chex.dataclass(frozen=True)
  class VTraceCarry: 
    next_value: chex.Array # Network value in the next timestep 
    delta_v: chex.Array # Propagated delta V in V-trace from the next timestep
  
  
  init_carry = VTraceCarry(
    next_value=jnp.zeros_like(v[-1]),
    delta_v=jnp.zeros_like(v[-1])
  )

  def _v_trace(carry: VTraceCarry, x) -> tuple[VTraceCarry, Any]:
    (importance_sampling, v, q_reward, entropy_reward, weighted_regularization_term, valid, inverted_sampling, action_oh) = x 
    # reward_uncorrected = reward + gamma * carry.reward_uncorrected + entropy
    # discounted_reward = reward + gamma * carry.reward
    
    delta_v = jnp.minimum(rho, importance_sampling) * (entropy_reward + gamma * carry.next_value - v)
    carry_delta_v = delta_v + lambda_ * jnp.minimum(c, importance_sampling) * gamma * carry.delta_v
    
    v_target = v + carry_delta_v
    
    
    # We use importance sampling of the opponent.
    opponent_sampling = jnp.flip(importance_sampling, -2)
    
    q_value = v + weighted_regularization_term  + action_oh * opponent_sampling * inverted_sampling  * (q_reward + gamma * (carry.next_value + carry.delta_v) - v )
    
    next_carry = VTraceCarry(
      next_value=v,
      delta_v=carry_delta_v
    )
    reset_carry = init_carry
  
    reset_v_target = jnp.zeros_like(v_target)
    reset_q_value = jnp.zeros_like(q_value) 
    
    reset_carry = init_carry
    return tree_where(valid, (next_carry, (v_target, q_value)), (reset_carry, (reset_v_target, reset_q_value)))
    # return jnp.where(valid, next_carry, reset_carry), (v_target, q_value)
    
    
    
  _, (v_target, q_value) = lax.scan(
    f=_v_trace,
    init=init_carry,
    xs=(importance_sampling, v, q_reward, entropy_reward, weighted_regularization_term, valid, inverted_sampling, action_oh),
    reverse=True
  ) 
  return v_target, q_value
  



class RNaDDreamer():
  """Implementation of the Regularized Nash Dynamics algorithm (RNaD)
  introduced in https://arxiv.org/pdf/2206.15378. Unlike the original implementation,
  this is designed to work only in two player simultaneous move games and 
  also uses the learned Dreamer model to generate trajectories instead of the 
  original game environment."""
  def __init__(self, dreamer_model: DreamerMA, config: RNaDConfig) -> None:
    
    self.config = config
    self.world_model = dreamer_model
    self.init()

  def init(self):

    self.actions = self.world_model.action_dimension
    #Unlike Dreamer, we operate with rewards defined 
    # as (state, action, next_state) and only care how to
    # act in non-terminal states, hence we end one turn before terminal
    self.trajectory_max = self.world_model.trajectory_max - 1
    self.non_chance_trajectory_max = self.world_model.non_chance_trajectory_max - 1
    self.num_players = self.world_model.game.num_players()

    self.example_hidden = jnp.zeros(self.world_model.config.hidden_state_size)
    self.example_categorical = jnp.zeros((self.world_model.config.encoded_classes, self.world_model.config.encoded_categories))
    
    self.rng_key = jax.random.key(self.config.seed)
    self.nnx_rngs = nnx.Rngs(jax.random.key(self.config.network_seed))

    self.iset_size = self.world_model.game.information_state_tensor_shape()
    
    
    self.example_timestep = self.default_timestep()
    
    self._entropy_schedule = EntropySchedule(
        sizes=self.config.entropy_schedule_size,
        repeats=self.config.entropy_schedule_repeats)
    
    self.prev_network = RNaDNetwork(self.iset_size, self.actions, self.config.rnad_network_details[0], self.config.rnad_network_details[1], rngs=self.nnx_rngs)
    self._prev_network = RNaDNetwork(self.iset_size, self.actions, self.config.rnad_network_details[0], self.config.rnad_network_details[1], rngs=self.nnx_rngs)
    
    
    if self.config.send_signal_to_dreamer:
      self.optimizers = initialize_joint_optimizers(self.world_model.optimizers, self.config, self.iset_size, self.actions, self.nnx_rngs)
      self.cached_step = nnx.cached_partial(self._jit_step_with_model, self.optimizers, self.prev_network, self._prev_network)
    else:
      self.optimizers = initialize_rnad_optimizers(self.config, self.iset_size, self.actions, self.nnx_rngs)
      self.cached_step = nnx.cached_partial(self._jit_step, self.optimizers, self.prev_network, self._prev_network, self.world_model.optimizers)
    self.learner_steps = 0
    self.policy_switch_steps = 0
  
  
  def default_timestep(self):
    obs = np.zeros(self.iset_size, dtype=np.float32)
    
    legal = np.ones((1, self.actions), dtype=np.int8)
    action = np.ones((1, self.actions), dtype=np.float32)
    policy = np.ones((1,self.actions), dtype=np.float32)
    valid = np.array(0, dtype=np.float32)
    reward = np.array(0, dtype=np.float32)
    
    ts = RNaDTimeStep(
      valid = valid,
      obs = obs,
      legal = legal,
      action = action, 
      policy = policy,
      reward = reward
    )
    return ts
    

  @partial(nnx.jit, static_argnums=(0,))
  def _jit_get_network(self, network: RNaDNetwork, obs, legal) -> chex.Array:
    return network(obs, legal)
  
  @partial(nnx.jit, static_argnums=(0,))
  @nnx.vmap(in_axes=(None, None, 1, 1), out_axes=(1))
  def _jit_get_batch_network(self, network: RNaDNetwork, obs, legal) -> chex.Array:
    return network(obs, legal)
  
  @partial(nnx.jit, static_argnums=(0,))
  def _jit_get_policy(self, network: RNaDNetwork, obs, legal) -> chex.Array:
    return self._jit_get_network(network, obs, legal)[0]
  
  # TODO: Be careful, this sometimes produces an action that is illegal
  @partial(nnx.jit, static_argnums=(0,))
  def _jit_sample_action(self, key, pi: chex.Array):
    
    def choice_wrapper(key, pi):
      return jax.random.choice(key, self.actions, p=pi)
    
    action = jax.vmap(choice_wrapper, in_axes=(0, 0), out_axes=0)(key, pi)
    action_oh = jax.nn.one_hot(action, self.actions)
    return action, action_oh
  
  @partial(nnx.jit, static_argnums=(0,))
  def _jit_get_policy_and_action(self, network: RNaDNetwork, key, obs, legal) -> chex.Array:
    pi = self._jit_get_policy(network, obs, legal)
    action, action_oh = self._jit_sample_action(key, pi)
    return pi, action, action_oh
  
  @partial(nnx.jit, static_argnums=(0,))
  @nnx.vmap(in_axes=(None, None, 1, 1, 1), out_axes=1)
  def _jit_get_batch_policy(self, network: RNaDNetwork, key, obs, legal) -> chex.Array:
    return self._jit_get_policy_and_action(network, key, obs, legal)
  
  # Expects obs and legal to be in shape [Batch, Player, ...]
  # @partial(nnx.jit, static_argnums=(0))
  # def batch_policy_and_action(self, network: RNaDNetwork, obs, legal):
    
  #   keys = self.get_next_rng_keys_dimensional(obs.shape[:2])
  #   keys = np.array(keys)
  #   pi, action, action_oh = self._jit_get_batch_policy(network, keys, obs, legal)
  #   pi = np.array(pi, dtype=np.float64)
  #   pi = pi / np.sum(pi, axis=-1, keepdims=True) # TODO: Remove this
  #   action = np.array(action, dtype=np.int32)
  #   action_oh = np.array(action_oh, dtype=np.float64)
  #   return pi, action, action_oh
  
  # def get_policy(self, network: RNaDNetwork, obs, legal, player: int):
  #   pi = self._jit_get_policy(network, obs, legal)
  #   return pi[player]
  
  @partial(nnx.jit, static_argnums=0)
  def get_policy_both(self, network: RNaDNetwork, joint_obs, joint_legal):
    #vmap over the player dimension
    players_get_policy = nnx.vmap(self._jit_get_policy, in_axes=(None, 0, 0), out_axes=(0))
    pi = players_get_policy(network, joint_obs, joint_legal)
    return pi
  
  #TODO: Is it necessary to pass all the models explicitly like this?
  @partial(nnx.jit, static_argnums=0)
  def sample_trajectories(self, key, rnad_network: RNaDNetwork, sequence_model: SequenceModel, dynamics: DynamicsPredictor,
                        predictor: Predictor, legal_network: LegalActionsNetwork, encoder: JointIsetEncoder, p1_iset_decoder:IsetDecoder,
                        p2_iset_decoder: IsetDecoder) ->RNaDTimeStep:
    keys = jax.random.split(key, self.config.batch_size)
    sample_trajectory_func = self.sample_trajectory if self.config.use_learned_model else self.sample_trajectory_from_game
    batch_sample_trajectory = nnx.vmap(sample_trajectory_func, in_axes=(0, None, None, None, None, None, None, None, None), out_axes=1) 
    return batch_sample_trajectory(keys, rnad_network, sequence_model, dynamics, predictor, legal_network, encoder, p1_iset_decoder, p2_iset_decoder)


  #TODO: Is it necessary to pass all the models explicitly like this?
  @partial(nnx.jit, static_argnums=0)
  def sample_trajectory(self, key, rnad_network: RNaDNetwork, sequence_model: SequenceModel, dynamics: DynamicsPredictor,
                        predictor: Predictor, legal_network: LegalActionsNetwork, encoder: JointIsetEncoder, p1_iset_decoder:IsetDecoder,
                        p2_iset_decoder: IsetDecoder) ->RNaDTimeStep:
    init_chance_sample_key, init_sample_key, trajectory_key, = jax.random.split(key, 3)
    trajectory_key = jax.random.split(trajectory_key, self.trajectory_max)
  
    
    #get initial state from the environment
    # to get the posterior estimate for the initial state
    game_state, legal_actions = self.world_model.game.initialize_structures()
    is_chance = self.world_model.game.is_chance(game_state)
    # if the game begins with a chance node, sample an initial outcome,
    # since the Dreamer model was not trained to represent chance nodes
    # explicitly
    def sample_init_chance():
        outcomes, probs = self.world_model.game.get_outcomes_and_probs(game_state)
        # Do not forget for deterministic games to put nonzero probs
        # to sample something for shape consistency
        probs = jnp.where(is_chance, probs, jnp.ones_like(probs)/ probs.shape[0])
        chosen_outcome = jax.random.choice(init_chance_sample_key, outcomes, p=probs)
        outcome, terminal, reward, chosen_legals = self.world_model.game.apply_action(game_state, chosen_outcome)
        return outcome, chosen_legals
    def keep_init_state():
      return game_state, legal_actions
    game_state, legal_actions = jax.lax.cond(is_chance, sample_init_chance, keep_init_state)
    _, init_p1_iset, init_p2_iset, _ = self.world_model.game.get_info(game_state)
    init_obs = jnp.stack([init_p1_iset, init_p2_iset], axis=0)
    init_hidden = jnp.zeros(self.world_model.config.hidden_state_size)
    init_stoch = encoder(init_hidden, init_obs)
    init_deter = sample_categorical(init_stoch, init_sample_key, uniform_mix=0.0, sample_threshold=self.config.state_sample_threshold)
    
    @chex.dataclass(frozen=True)
    class SampleTrajectoryCarry:
      hidden_state:chex.Array
      deter_state: chex.Array
      legal_actions: chex.Array
      terminal: bool
      
    init_carry = SampleTrajectoryCarry(
      hidden_state = init_hidden,
      deter_state = init_deter,
      legal_actions = legal_actions.astype(jnp.int8),
      terminal = jnp.array(False)
    )
    
    
    @nnx.jit
    def choice_wrapper(key, p):
      action = jax.random.choice(key, self.actions, p=p)
      action_oh = jax.nn.one_hot(action, self.actions)
      return action, action_oh

    
    vectorized_sample_action = nnx.vmap(choice_wrapper, in_axes=(0, 0), out_axes=0)

    @nnx.scan(in_axes = (nnx.Carry, 0, None, None, None, None, None, None, None), out_axes=(nnx.Carry, 0))
    def _sample_trajectory(carry: SampleTrajectoryCarry, key , rnad_network: RNaDNetwork, sequence_model: SequenceModel, dynamics: DynamicsPredictor, 
                        predictor: Predictor, legal_network: LegalActionsNetwork, p1_iset_decoder:IsetDecoder,
                        p2_iset_decoder: IsetDecoder) -> tuple[SampleTrajectoryCarry, chex.Array]:
      
      #TODO: For now, iset decoder is used to create trajectories 
      # trained on the "original" isets. This might be changed later
      p1_iset = p1_iset_decoder(carry.hidden_state, carry.deter_state)
      p2_iset = p2_iset_decoder(carry.hidden_state, carry.deter_state)
      obs = jnp.stack([p1_iset, p2_iset], axis=0)

      #get policy 
      pi = self.get_policy_both(rnad_network, obs, carry.legal_actions)
      #uniform mix to the policy
      uniform_pi = carry.legal_actions / jnp.sum(carry.legal_actions, axis=-1, keepdims=True)
      pi = self.config.sampling_epsilon * uniform_pi + (1 - self.config.sampling_epsilon) * pi
      # For each player samples a single action
      
      action_sample_key, state_sample_key = jax.random.split(key)
      action_sample_keys = jax.random.split(action_sample_key, self.num_players)
      action, action_oh = vectorized_sample_action(action_sample_keys, pi)
      
      
      flattened_action = jnp.reshape(action_oh, (*carry.deter_state.shape[:-2], -1))
      gru_input = jnp.concatenate([carry.deter_state.reshape(*carry.deter_state.shape[:-2], -1), flattened_action], axis=-1) 
      next_hidden = sequence_model(carry.hidden_state, gru_input)
      next_stoch = dynamics(next_hidden)
      next_deter = sample_categorical(next_stoch, state_sample_key, uniform_mix=0.0, sample_threshold=self.config.state_sample_threshold)
      next_reward, next_terminal, next_legal = self.world_model.get_predictor(predictor, legal_network, next_hidden, next_deter)
      next_terminal = jnp.logical_or(carry.terminal, next_terminal)
      valid = jnp.logical_not(carry.terminal)
      timestep = RNaDTimeStep(
        obs = obs,
        legal = carry.legal_actions,
        action = action_oh,
        policy = pi,
        reward = next_reward,
        valid = valid
      )
      new_carry = SampleTrajectoryCarry(
        hidden_state = next_hidden,
        deter_state = next_deter,
        legal_actions=jnp.where(next_terminal, self.example_timestep.legal, next_legal),
        terminal = next_terminal,
      )
         
      timestep = jax.tree.map(lambda t, f: jnp.where(carry.terminal, t, f), self.example_timestep, timestep)
      return new_carry, timestep
    _, timestep = _sample_trajectory(init_carry, trajectory_key, rnad_network, sequence_model, dynamics, predictor, legal_network, p1_iset_decoder, p2_iset_decoder)
    #[Trajectory, ...]
    return timestep
  
  @partial(nnx.jit, static_argnums=0)
  def sample_trajectory_from_game(self, key, rnad_network: RNaDNetwork, sequence_model: SequenceModel, dynamics: DynamicsPredictor,
                        predictor: Predictor, legal_network: LegalActionsNetwork, encoder: JointIsetEncoder, p1_iset_decoder:IsetDecoder,
                        p2_iset_decoder: IsetDecoder) ->RNaDTimeStep:
    
    trajectory_key = jax.random.split(key, self.trajectory_max)
  
    
    game_state, legal_actions = self.world_model.game.initialize_structures()
    
    @chex.dataclass(frozen=True)
    class SampleTrajectoryCarry:
      game_state: GameState
      legal_actions: chex.Array
      terminal: bool
      
    init_carry = SampleTrajectoryCarry(
      game_state = game_state,
      legal_actions = legal_actions.astype(jnp.int8),
      terminal = jnp.array(False),
    )
    
    
    @nnx.jit
    def choice_wrapper(key, p):
      action = jax.random.choice(key, self.actions, p=p)
      action_oh = jax.nn.one_hot(action, self.actions)
      return action, action_oh

    
    vectorized_sample_action = nnx.vmap(choice_wrapper, in_axes=(0, 0), out_axes=0)

    @nnx.scan(in_axes = (nnx.Carry, 0, None), out_axes=(nnx.Carry, 0, 0, 0))
    def _sample_trajectory(carry: SampleTrajectoryCarry, key , rnad_network: RNaDNetwork) -> tuple[SampleTrajectoryCarry, chex.Array]:
      
      _, p1_iset, p2_iset, _ = self.world_model.game.get_info(carry.game_state)
      obs = jnp.stack([p1_iset, p2_iset], axis=0)

      #get policy 
      #pi = self.get_policy_both(rnad_network, obs, carry.legal_actions)
      is_chance = self.world_model.game.is_chance(carry.game_state)
      pi = jnp.where(is_chance, carry.legal_actions / jnp.sum(carry.legal_actions, axis=-1, keepdims=True), self._jit_get_policy(rnad_network, obs, carry.legal_actions))
      #uniform mix to the policy
      uniform_pi = carry.legal_actions / jnp.sum(carry.legal_actions, axis=-1, keepdims=True)
      pi = self.config.sampling_epsilon * uniform_pi + (1 - self.config.sampling_epsilon) * pi
      # For each player samples a single action
      
      action_sample_key, chance_key = jax.random.split(key)
      action_sample_keys = jax.random.split(action_sample_key, self.num_players)
      action, action_oh = vectorized_sample_action(action_sample_keys, pi)
      
      
      def apply_action():
        return self.world_model.game.apply_action(carry.game_state, action)
      def sample_chance():
        outcomes, probs = self.world_model.game.get_outcomes_and_probs(carry.game_state)
        # Do not forget for deterministic games to put nonzero probs
        # to sample something for shape consistency
        probs = jnp.where(is_chance, probs, jnp.ones_like(probs)/ probs.shape[0])
        chosen_outcome = jax.random.choice(chance_key, outcomes, p=probs)
        outcome, terminal, reward, chosen_legals = self.world_model.game.apply_action(carry.game_state, chosen_outcome)
        return outcome, terminal, reward, chosen_legals
      next_game_state, next_terminal, next_rewards, next_legal = jax.lax.cond(is_chance, sample_chance, apply_action)
      next_terminal = jnp.logical_or(carry.terminal, next_terminal)
      next_chance = self.world_model.game.is_chance(next_game_state)
      valid = jnp.ones_like(next_rewards) - carry.terminal
      timestep = RNaDTimeStep(
        obs = obs,
        legal = carry.legal_actions,
        action = action_oh,
        policy = pi,
        reward = next_rewards,
        valid = valid
      )
      new_carry = SampleTrajectoryCarry(
        game_state = next_game_state,
        legal_actions=jnp.where(next_terminal, self.example_timestep.legal, next_legal.astype(jnp.int8)),
        terminal = next_terminal
      )
         
      timestep = jax.tree_util.tree_map(lambda t, f: jnp.where(carry.terminal, t, f), self.example_timestep, timestep)
      return new_carry, timestep, is_chance, next_chance
    _, timestep, is_chance, next_chance = _sample_trajectory(init_carry, trajectory_key, rnad_network)
    #Filter out the chance nodes.
    # Just a bit tricky, since most of the timestep is related
    # directly to the state, but reward is actually also related to the next state.
    # So, because playing an action that produces chance node returns an invalid reward,
    # and the actual reward will be returned in the chance node, we need to filter
    # out the reward one step BEFORE the chance node
    non_chance = jnp.nonzero(~is_chance, size=self.non_chance_trajectory_max)[0]
    non_next_chance = jnp.nonzero(~next_chance, size=self.non_chance_trajectory_max)[0]
    filtered_timestep = RNaDTimeStep(obs = jnp.take_along_axis(timestep.obs, non_chance[..., None, None], axis=0),
                                    legal = jnp.take_along_axis(timestep.legal, non_chance[..., None, None], axis=0),
                                    action = jnp.take_along_axis(timestep.action, non_chance[..., None, None], axis=0),
                                    policy = jnp.take_along_axis(timestep.policy, non_chance[..., None, None], axis=0),
                                    reward = jnp.take_along_axis(timestep.reward, non_next_chance, axis=0),
                                    valid = jnp.take_along_axis(timestep.valid, non_chance, axis=0))
    #[Trajectory, ...]
    return filtered_timestep
    
  
  def get_next_rng_key(self):
    self.rng_key, key = jax.random.split(self.rng_key)
    return key
  
  def get_next_nnx_rngs(self):
    self.nnx_rng_key, key = jax.random.split(self.nnx_rng_key)
    return nnx.Rngs(key)
  
  # First it generates keys for the batch
  def get_next_rng_keys_dimensional(self, n):
    key = self.get_next_rng_key()
    keys = jax.random.split(key, n)
    return keys
  


  @partial(nnx.jit, static_argnums=(0,))
  def update_parameters_and_model(
    self,
    optimizers: JointOptimizers,
    prev_network: RNaDNetwork,
    _prev_network: RNaDNetwork,
    trajectory_key,
    alpha,
    update_net 
  ):
    """Same functionality as update parameters, but also
    updates the world model. TODO: Lots of repeated code. Either 
    try to merge these, or create new RNaD version for the joint training."""

    def rnad_loss(
      rnad_network: RNaDNetwork,
      sequence_model: SequenceModel, 
      dynamics: DynamicsPredictor,
      predictor: Predictor, 
      legal_network: LegalActionsNetwork, 
      encoder: JointIsetEncoder, 
      p1_iset_decoder:IsetDecoder,
      p2_iset_decoder: IsetDecoder,
      target_network: RNaDNetwork,
      prev_network: RNaDNetwork,
      _prev_network: RNaDNetwork,
      trajectory_key,
      alpha: float,
    ):
      timestep = self.sample_trajectories(trajectory_key, rnad_network, 
                                          sequence_model,
                                          dynamics,
                                          predictor,
                                          legal_network,
                                          encoder,
                                          p1_iset_decoder,
                                          p2_iset_decoder)
      # Per player vmap
      per_player_net_apply = nnx.vmap(self._jit_get_network, in_axes=(None, 0, 0), out_axes=(0))
      #Per trajectory and batch dimensions
      vectorized_net_apply = nnx.vmap(nnx.vmap(per_player_net_apply, in_axes=(None, 0, 0), out_axes=(0)), in_axes=(None, 0, 0), out_axes=(0))
      pi, v, log_pi, logit = vectorized_net_apply(rnad_network, timestep.obs, timestep.legal)
      
      _, v_target, _, _ = vectorized_net_apply(target_network, timestep.obs, timestep.legal)
      _, _, log_pi_prev, _ = vectorized_net_apply(prev_network, timestep.obs, timestep.legal)
      _, _, log_pi_prev_, _ = vectorized_net_apply(_prev_network, timestep.obs, timestep.legal)
      

      # This creates the regularization term for rewards
      regularized_term = log_pi - (alpha * log_pi_prev + (1 - alpha) * log_pi_prev_) 
      
      expanded_valid = jnp.expand_dims(timestep.valid, (-1, -2))
      
      v_train_target, q_value = v_trace(v_target, expanded_valid, timestep.policy, pi, regularized_term, timestep.action, timestep.reward,
                                        self.config.lambda_vtrace, self.config.c_vtrace, self.config.rho_vtrace,
                                        self.config.eta, self.config.vtrace_eta, self.config.gamma_vtrace)
      
      # We multiply by 2, since each player acts
      normalization = jnp.sum(timestep.valid) * 2 
      v_loss = jnp.sum((expanded_valid * (v - lax.stop_gradient(v_train_target)) ** 2)) / (normalization + (normalization == 0))
      
      # Each Q is multiplied by product of importance_sampling of opponent and inverted sampling policy by the acting player.
      # This computes counterfactual importance sampling
      sampling_policy = jnp.sum(timestep.policy * timestep.action, axis=-1, keepdims=True)
      network_policy = jnp.sum(pi * timestep.action, axis=-1, keepdims=True)
      
      # We do not take into account the player reaches, since infoset is always reached with the same prob
      sampling_policy = jnp.prod(sampling_policy, axis=-2, keepdims=True)
      
      importance_sampling = network_policy / sampling_policy
      
      importance_sampling = jnp.concatenate((jnp.ones((1, *importance_sampling.shape[1:])), importance_sampling[:-1]), axis=0)
      importance_sampling = jnp.cumprod(importance_sampling, axis=0)
      importance_sampling = jnp.flip(importance_sampling, axis=-2)
      
      
      loss_neurd = neurd_loss(logit, pi, q_value, timestep.legal, importance_sampling)
      
      # The multiplication by -1 is critical here, otherwise we would
      # be minimizing the neurd term, but we want to maximize it.
      neurd_loss_value = -jnp.sum(loss_neurd * expanded_valid) / (normalization + (normalization == 0))
      #jax.debug.breakpoint()
      return v_loss + neurd_loss_value
      
    loss, grads = nnx.value_and_grad(rnad_loss, argnums=(0 ,1, 2, 3, 4, 5, 6, 7))(
      optimizers.rnad_optimizer.model,
      optimizers.sequence_optimizer.model,
      optimizers.dynamics_optimizer.model,
      optimizers.predictor_optimizer.model,
      optimizers.legal_actions_optimizer.model,
      optimizers.encoder_optimizer.model,
      optimizers.p1_decoder_optimizer.model,
      optimizers.p2_decoder_optimizer.model, 
      optimizers.rnad_target_optimizer.model,
      prev_network,
      _prev_network
      , trajectory_key, alpha)

    optimizers.rnad_optimizer.update(grads[0])
    optimizers.sequence_optimizer.update(grads[1])
    optimizers.dynamics_optimizer.update(grads[2])
    optimizers.predictor_optimizer.update(grads[3])
    optimizers.legal_actions_optimizer.update(grads[4])
    optimizers.encoder_optimizer.update(grads[5])
    optimizers.p1_decoder_optimizer.update(grads[6])
    optimizers.p2_decoder_optimizer.update(grads[7])

    rnad_graphdef, state = nnx.split(optimizers.rnad_optimizer.model)
    _, state_target = nnx.split(optimizers.rnad_target_optimizer.model)
    _, state_prev = nnx.split(prev_network)
    _, _state_prev = nnx.split(_prev_network)

    #This grad coupled with vanilla SGD optimizer 
    # is equivalent to the EMA formula (1 - alpha) * state_target + alpha * state
    target_grad = jax.tree.map(lambda a, b: a - b, state_target, state)
    optimizers.rnad_target_optimizer.update(target_grad)
      

    state_prev, _state_prev = jax.lax.cond(
        update_net,
        lambda: (state_target, state_prev),
        lambda: (state_prev, _state_prev))
    prev_network = nnx.merge(rnad_graphdef, state_prev)
    _prev_network = nnx.merge(rnad_graphdef, _state_prev)
    return prev_network, _prev_network, loss
  
  @partial(nnx.jit, static_argnums=(0,))
  def update_parameters(
    self,
    optimizers: RNaDOptimizers,
    prev_network: RNaDNetwork,
    _prev_network: RNaDNetwork,
    timestep: RNaDTimeStep,
    alpha: float,
    update_net, 
  ):
     
    def rnad_loss(
      rnad_network: RNaDNetwork,
      target_network: RNaDNetwork,
      prev_network: RNaDNetwork,
      _prev_network: RNaDNetwork,
      timestep: RNaDTimeStep,
      alpha: float,
    ):
      
      # Per player vmap
      per_player_net_apply = nnx.vmap(self._jit_get_network, in_axes=(None, 0, 0), out_axes=(0))
      #Per trajectory and batch dimensions
      vectorized_net_apply = nnx.vmap(nnx.vmap(per_player_net_apply, in_axes=(None, 0, 0), out_axes=(0)), in_axes=(None, 0, 0), out_axes=(0))
      pi, v, log_pi, logit = vectorized_net_apply(rnad_network, timestep.obs, timestep.legal)
      
      _, v_target, _, _ = vectorized_net_apply(target_network, timestep.obs, timestep.legal)
      _, _, log_pi_prev, _ = vectorized_net_apply(prev_network, timestep.obs, timestep.legal)
      _, _, log_pi_prev_, _ = vectorized_net_apply(_prev_network, timestep.obs, timestep.legal)
      

      # This creates the regularization term for rewards
      regularized_term = log_pi - (alpha * log_pi_prev + (1 - alpha) * log_pi_prev_) 
      
      expanded_valid = jnp.expand_dims(timestep.valid, (-1, -2))
      
      v_train_target, q_value = v_trace(v_target, expanded_valid, timestep.policy, pi, regularized_term, timestep.action, timestep.reward,
                                        self.config.lambda_vtrace, self.config.c_vtrace, self.config.rho_vtrace,
                                        self.config.eta, self.config.vtrace_eta, self.config.gamma_vtrace)
      
      # We multiply by 2, since each player acts
      normalization = jnp.sum(timestep.valid) * 2 
      v_loss = jnp.sum((expanded_valid * (v - lax.stop_gradient(v_train_target)) ** 2)) / (normalization + (normalization == 0))
      
      # Each Q is multiplied by product of importance_sampling of opponent and inverted sampling policy by the acting player.
      # This computes counterfactual importance sampling
      # This counterfactual correction allows for an-off policy
      # RNaD.
      sampling_policy = jnp.sum(timestep.policy * timestep.action, axis=-1, keepdims=True)
      network_policy = jnp.sum(pi * timestep.action, axis=-1, keepdims=True)
      
      # We do not take into account the player reaches, since infoset is always reached with the same prob
      sampling_policy = jnp.prod(sampling_policy, axis=-2, keepdims=True)
      
      importance_sampling = network_policy / sampling_policy
      
      importance_sampling = jnp.concatenate((jnp.ones((1, *importance_sampling.shape[1:])), importance_sampling[:-1]), axis=0)
      importance_sampling = jnp.cumprod(importance_sampling, axis=0)
      importance_sampling = jnp.flip(importance_sampling, axis=-2)
      
      
      loss_neurd = neurd_loss(logit, pi, q_value, timestep.legal, importance_sampling)
      
      # The multiplication by -1 is critical here, otherwise we would
      # be minimizing the neurd term, but we want to maximize it.
      neurd_loss_value = -jnp.sum(loss_neurd * expanded_valid) / (normalization + (normalization == 0))
      return v_loss + neurd_loss_value
      
    loss, grad = nnx.value_and_grad(rnad_loss, argnums=0)(optimizers.rnad_optimizer.model, optimizers.rnad_target_optimizer.model, prev_network, _prev_network, timestep, alpha)

    
    optimizers.rnad_optimizer.update(grad)

    rnad_graphdef, state = nnx.split(optimizers.rnad_optimizer.model)
    _, state_target = nnx.split(optimizers.rnad_target_optimizer.model)
    _, state_prev = nnx.split(prev_network)
    _, _state_prev = nnx.split(_prev_network)

    #This grad coupled with vanilla SGD optimizer 
    # is equivalent to the EMA formula (1 - alpha) * state_target + alpha * state
    target_grad = jax.tree.map(lambda a, b: a - b, state_target, state)
    optimizers.rnad_target_optimizer.update(target_grad)
      

    state_prev, _state_prev = jax.lax.cond(
        update_net,
        lambda: (state_target, state_prev),
        lambda: (state_prev, _state_prev))
    prev_network = nnx.merge(rnad_graphdef, state_prev)
    _prev_network = nnx.merge(rnad_graphdef, _state_prev)
    return prev_network, _prev_network, loss
  
  @partial(nnx.jit, static_argnums=(0))
  def _jit_step(self, optimizers: RNaDOptimizers, prev_network: RNaDNetwork, _prev_network: RNaDNetwork, world_model_optimizers: DreamerMAOptimizers
                , trajectory_key, learner_steps: int):
    timestep = self.sample_trajectories(trajectory_key, optimizers.rnad_optimizer.model, 
                                          world_model_optimizers.sequence_optimizer.model,
                                          world_model_optimizers.dynamics_optimizer.model,
                                          world_model_optimizers.predictor_optimizer.model,
                                          world_model_optimizers.legal_actions_optimizer.model,
                                          world_model_optimizers.encoder_optimizer.model,
                                          world_model_optimizers.p1_decoder_optimizer.model,
                                          world_model_optimizers.p2_decoder_optimizer.model)
    
    alpha, update_regularization = self._entropy_schedule(learner_steps)
    
    prev_network, _prev_network, loss = self.update_parameters(
      optimizers, prev_network, _prev_network, timestep, alpha, update_regularization)
    return prev_network, _prev_network, loss, update_regularization
  
  @partial(nnx.jit, static_argnums=(0))
  def _jit_step_with_model(self, optimizers: JointOptimizers, prev_network: RNaDNetwork, _prev_network: RNaDNetwork
                , trajectory_key, learner_steps: int):
    alpha, update_regularization = self._entropy_schedule(learner_steps)
    prev_network, _prev_network, loss = self.update_parameters_and_model(
      optimizers, prev_network, _prev_network, trajectory_key, alpha, update_regularization
    )
    return prev_network, _prev_network, loss, update_regularization


  def step(self):
    trajectory_key = self.get_next_rng_key()
    #self.prev_network, self._prev_network, loss, update_regularization =  self._jit_step(self.optimizers,self.prev_network, self._prev_network, self.world_model.optimizers,trajectory_key, self.learner_steps)
    self.prev_network, self._prev_network, loss, update_regularization = self.cached_step(trajectory_key, self.learner_steps)
    self.learner_steps += 1
    self.policy_switch_steps += int(update_regularization)
    return loss
  
  def step_with_model(self):
    trajectory_key = self.get_next_rng_key()
    #self.prev_network, self._prev_network, loss, update_regularization =  self._jit_step_with_model(self.optimizers, self.prev_network, self._prev_network, trajectory_key, self.learner_steps)
    self.prev_network, self._prev_network, loss, update_regularization = self.cached_step(trajectory_key, self.learner_steps)
    self.learner_steps += 1
    self.policy_switch_steps += int(update_regularization)
    return loss

  
  def train_model(self, model_save_dir:str, num_steps:int, print_each: int = -1, save_each: int = -1):
     
    for i in range(num_steps):
      loss = self.step()
      if print_each > 0 and i % print_each == 0:
        print(f"Step {i}, Loss: {loss}")
      if save_each > 0 and i % save_each == 0:
        model_file = model_save_dir + f"step_{i}.pkl"
        save_model(self, model_file)

  def __getstate__(self):
    return {"config": self.config,
            "world_model": self.world_model,
            "optimizers": nnx.state(self.optimizers),
            "prev_network": nnx.state(self.prev_network),
            "_prev_network": nnx.state(self._prev_network),
            "steps": self.learner_steps,
            "trajectory_key": self.rng_key}
  
  def __setstate__(self, state):
    self.config = state["config"]
    self.world_model = state["world_model"]

    self.init()

    def update_nnx(model: nnx.Module, saved_state: nnx.State):
      graphdef, _ = nnx.split(model)
      updated_model = nnx.merge(graphdef, saved_state)
      return updated_model

    self.rng_key = state["trajectory_key"]
    self.learner_steps = state["steps"]
    self.optimizers = update_nnx(self.optimizers, state["optimizers"])
    self.prev_network = update_nnx(self.prev_network, state["prev_network"])
    self._prev_network = update_nnx(self._prev_network, state["_prev_network"])

  
  
  
def main():
  cards = 3
  network_seed = 99
  trajectory_seed = 99
  restore_step = 1000
  model_path = f"trained_networks/goofspiel_{cards}/seed{trajectory_seed}/network_seed{network_seed}/step_{restore_step}.pkl"
  model_path = os.getcwd() + "/" + model_path
  model = load_model(model_path)
  config = RNaDConfig(batch_size = 4)
  solver = RNaDDreamer(dreamer_model=model, config=config)
  for _ in range(10):
    solver.step()
  
  
if __name__ == "__main__":
  main()