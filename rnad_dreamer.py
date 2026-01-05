
import jax
import jax.numpy as jnp
import jax.lax as lax
import optax

import flax.nnx as nnx
import chex

import numpy as np


from functools import partial
from typing import Any

from games.jax_game import JaxGame
from ma_rssm import *
from train_utils import *
from distributions import get_bin_log_prob

#jax.config.update("jax_debug_nans", True)



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
  
  importance_sampling = policy_ratio(network_policy, sampling_policy, action_oh, valid)
  
  # The reason we use this is to ensure this is weighted by the amount of the times we sample it
  inverted_sampling = policy_ratio(jnp.ones_like(sampling_policy), sampling_policy, action_oh, valid)
  
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

    
    
    
  _, (v_target, q_value) = lax.scan(
    f=_v_trace,
    init=init_carry,
    xs=(importance_sampling, v, q_reward, entropy_reward, weighted_regularization_term, valid, inverted_sampling, action_oh),
    reverse=True
  )
  return v_target, q_value
  




def neurd_loss_with_range(
  logits: chex.Array,
  policy: chex.Array,
  q_values: chex.Array, 
  legal: chex.Array,
  importance_sampling: chex.Array,
  return_range: chex.Array
):
  """A version of NeuRD loss that does not do any thresholding
  or clipping like standard NeuRD, but instead normalizes by the
  range obtained as from exponential moving average of 
  highest return differences, as described in https://arxiv.org/pdf/2301.04104 page 6."""
  advantage = q_values - jnp.sum(policy * q_values, axis=-1, keepdims=True)
  advantage = advantage * importance_sampling
  advantage = jax.lax.stop_gradient(advantage / jnp.maximum(1, return_range))

  
  neurd_loss_value = jnp.sum(legal * logits * advantage, axis=-1, keepdims=True)
  
  return neurd_loss_value


class RNaDDreamer():
  """A Regularized Nash Dynamics actor-critic algorithm, as 
  described in https://arxiv.org/abs/2206.15378. This version is changed
  for simultaneous move 2p0s games specifically and also to be able
  to handle the imagination trajectories produced by Dreamer world model."""
  def __init__(self, game: JaxGame, config: RNaDConfig, full_optimizer: nnx.Optimizer, target_optimizer: nnx.Optimizer) -> None:
    self.config = config
    self.optimizer = full_optimizer
    self.target_optimizer = target_optimizer
    self.init(game)

  def init(self, game: JaxGame):

    self.actions = game.num_distinct_actions()
    self.num_players = game.num_players()

    ma_rssm = self.optimizer.model
    self.is_iig = ma_rssm.is_iig
    self.input_size = ma_rssm.infoset_size

    num_last = self.config.num_last
    #If negative, take all for unroll, the Dreamer trajectories
    # will have one more timestep, hence + 1
    if num_last <= 0:
      #Unlike World model, we operate with rewards defined 
      # as (state, action, next_state) and only care how to
      # act in non-terminal states, hence we end one turn before terminal
      num_last = game.max_trajectory_lenght_no_chance()
    self.num_last = num_last

    self.return_range = jnp.array(0)
    #self.cached_step = nnx.cached_partial(self.update_parameters_and_model, self.optimizer, self.target_optimizer, self.prev_network, self._prev_network)
    self.learner_steps = 0
    self.policy_switch_steps = 0
    
    self._entropy_schedule = EntropySchedule(
        sizes=self.config.entropy_schedule_size,
        repeats=self.config.entropy_schedule_repeats)
    
    rnad_graphdef, rnad_state = nnx.split(ma_rssm.actor_critic)
    self.prev_network = nnx.merge(rnad_graphdef, rnad_state)

    self._prev_network = nnx.merge(rnad_graphdef, rnad_state)

    self.metrics_keys = ['img_val', 'img_policy', 'real_val']
    if self.config.train_real_policy:
      self.metrics_keys.append('real_policy')
    self.metrics = {k: 0 for k in self.metrics_keys}
    self.grad_norms = {'img': {}, 'real': {}}
    self.network_keys = (ma_rssm.network_names[-1], )
    
  
  
  @partial(nnx.jit, static_argnums=0)
  def _jit_get_network(self, network: RNaDNetwork, obs, legal):
    pi, v_dist_logits, log_pi, logits = network(obs, legal)
    return pi, v_dist_logits, log_pi, logits


  @partial(nnx.jit, static_argnums=(0,))
  def update_parameters_and_model(
    self,
    optimizer: nnx.Optimizer,
    target_optimizer: nnx.Optimizer,
    prev_network: RNaDNetwork,
    _prev_network: RNaDNetwork,
    trajectory_key,
    wm_timestep: TimeStep,
    wm_prediction_step: PredictionStepWithLegal,
    return_range: chex.Array,
    learner_steps: int
  ):
    """Compute RNaD loss and use it to perform
    a gradient step of both RNaD and Dreamer."""
    alpha, update_regularization = self._entropy_schedule(learner_steps)

    def rnad_loss(
      timestep: ActorCriticTimeStep,
      rnad_network: RNaDNetwork,
      target_network: RNaDNetwork,
      prev_network: RNaDNetwork,
      _prev_network: RNaDNetwork,
      start_reaches_is: chex.Array,
      return_range: chex.Array,
      alpha: float,
      compute_actor_loss=True
    ):
      
      bins = jnp.arange((2 * self.config.bin_range) + 1) - self.config.bin_range
      # Per player vmap
      per_player_net_apply = nnx.vmap(self._jit_get_network, in_axes=(None, 0, 0), out_axes=(0))
      #Per trajectory and batch dimensions
      vectorized_net_apply = nnx.vmap(nnx.vmap(per_player_net_apply, in_axes=(None, 0, 0), out_axes=(0)), in_axes=(None, 0, 0), out_axes=(0))
      pi, v_dist_logits, log_pi, logit = vectorized_net_apply(rnad_network, timestep.obs, timestep.legal)

      _, v_target_dist_logits, _, _ = vectorized_net_apply(target_network, timestep.obs, timestep.legal)
      _, _, log_pi_prev, _ = vectorized_net_apply(prev_network, timestep.obs, timestep.legal)
      _, _, log_pi_prev_, _ = vectorized_net_apply(_prev_network, timestep.obs, timestep.legal)
       
      v_target = get_value_from_bins(v_target_dist_logits, self.config.bin_range)
      # This creates the regularization term for rewards
      regularized_term = log_pi - (alpha * log_pi_prev + (1 - alpha) * log_pi_prev_) 
      
      expanded_valid = jnp.expand_dims(timestep.valid, (-1, -2))
      
      v_train_target, q_value = v_trace(v_target, expanded_valid, timestep.policy, pi, regularized_term, timestep.action, timestep.reward,
                                        self.config.lambda_vtrace, self.config.c_vtrace, self.config.rho_vtrace,
                                        self.config.eta, self.config.vtrace_eta, self.config.gamma_vtrace)
      
      q_mask = expanded_valid * timestep.legal
      percentiles = get_percentiles_with_mask(q_value, q_mask, jnp.array([95, 5]))
      current_range = (percentiles[0] - percentiles[1])
      new_range = 0.01 * current_range + 0.99 * return_range
      #v_train_target, q_value = jnp.zeros_like(v), jnp.zeros_like(pi)
      # We multiply by 2, since each player acts
      #v_loss = jnp.sum((expanded_valid * (v - lax.stop_gradient(v_train_target)) ** 2)) / (normalization + (normalization == 0))
      v_loss = -get_bin_log_prob(v_dist_logits, bins, jax.lax.stop_gradient(v_train_target))
      v_loss_value = get_loss_mean_with_mask(v_loss, expanded_valid)

      if compute_actor_loss:
        sampling_policy = jnp.sum(timestep.policy * timestep.action, axis=-1, keepdims=True) * expanded_valid + (1 - expanded_valid)
        network_policy = jnp.sum(pi * timestep.action, axis=-1, keepdims=True)* expanded_valid + (1 - expanded_valid)
        
        # We do not take into account the player reaches, since infoset is always reached with the same prob
        sampling_policy = jnp.prod(sampling_policy, axis=-2, keepdims=True)
        
        importance_sampling = network_policy / sampling_policy
        
        importance_sampling = jnp.concatenate((start_reaches_is, importance_sampling[:-1]), axis=0)
        importance_sampling = jnp.cumprod(importance_sampling, axis=0)
        #Flip to turn into counterfactual importance sampling
        importance_sampling = jnp.flip(importance_sampling, axis=-2)
        
        
        #loss_neurd = neurd_loss_with_range(logit, pi, q_value, timestep.legal, importance_sampling, new_range)
        
        loss_neurd = neurd_loss(logit, pi, q_value, timestep.legal, importance_sampling,
                                self.config.neurd_clip, self.config.neurd_threshold)

        # The multiplication by -1 is critical here, otherwise we would
        # be minimizing the neurd term, but we want to maximize it.
        neurd_loss_value = -get_loss_mean_with_mask(loss_neurd, expanded_valid)
      else:
        neurd_loss_value = 0
      return v_loss_value + neurd_loss_value, new_range, v_loss_value, neurd_loss_value

    def imagination_loss(model: MARSSM,
      target_network: RNaDNetwork,
      prev_network: RNaDNetwork,
      _prev_network: RNaDNetwork,
      trajectory_key,
      starting_points: PredictionStepWithLegal,
      start_reaches_is: chex.Array,
      return_range: chex.Array,
      alpha: float,
      beta_imagination: float):
        timestep = jax.lax.stop_gradient(model.imagine_trajectories(trajectory_key, starting_points))
        loss_val, new_range, v_loss, p_loss = rnad_loss(timestep, model.actor_critic, target_network, prev_network, _prev_network, start_reaches_is, return_range, alpha) 
        img_keys = self.metrics_keys[:2]
        losses = (v_loss, p_loss)
        metrics = {k: beta_imagination * v for k, v in zip(img_keys, losses)}
        return beta_imagination * loss_val, (new_range, metrics)
    
    def real_loss(model: MARSSM,
      target_network: RNaDNetwork,
      prev_network: RNaDNetwork,
      _prev_network: RNaDNetwork,
      timestep: ActorCriticTimeStep,
      return_range: chex.Array,
      alpha: float,
      beta_real: float):
        start_reaches_is = jnp.ones((1, *timestep.policy.shape[1:-1], 1))
        loss_val, new_range, v_loss, p_loss = rnad_loss(timestep, model.actor_critic, target_network, prev_network, _prev_network,
                                         start_reaches_is, return_range, alpha, compute_actor_loss=self.config.train_real_policy)
        real_keys = self.metrics_keys[2:]
        losses = (v_loss, p_loss) if self.config.train_real_policy else (v_loss, )
        metrics = {k: beta_real * v for k, v in zip(real_keys, losses)}
        return beta_real * loss_val, (new_range, metrics)
      
    def take_starts_for_unroll(network_pi:chex.Array, timestep: TimeStep, prediction_step: PredictionStepWithLegal):
      """Choose a starting point that is not invalid or terminal in the timestep
      uniformly. Chooses over the trajectory dimension and should be 
      vmaped over the batch dimension"""
      expanded_valid = timestep.valid[..., None, None] * ~timestep.terminal[..., None, None]
      #We also need to compute the importance sampling
      # for the player reaches, since we do not start at
      # the beggining of the trajectory.
      #Shape [T, Pl, 1]
      timestep_pi = jnp.sum(timestep.policy* timestep.action, axis=-1, keepdims=True) * expanded_valid + (1 - expanded_valid)
      #Shape [T, 1, 1]
      timestep_joint_pi = jnp.prod(timestep_pi, axis=-2, keepdims=True)
      #[T, Pl, 1]
      network_pi = jnp.sum(network_pi * timestep.action, axis=-1, keepdims=True) * expanded_valid + (1 - expanded_valid)
      trajectory_is = network_pi / timestep_joint_pi
      
      #[T, Pl, 1]
      start_reaches_is = jnp.cumprod(trajectory_is, axis=0)
      #[N_last, ...]
      starts = jax.tree.map(lambda x: x[-self.num_last:], prediction_step)
      #[N_last, Pl, 1]
      start_reaches_is = jax.tree.map(lambda x: x[-self.num_last: ], start_reaches_is)
      #sampled_start = jax.tree_util.tree_map(lambda x: x[0], prediction_step)
      return starts, start_reaches_is
    vectorized_starting_point = jax.vmap(take_starts_for_unroll, in_axes=(1, 1,1), out_axes=(0, 0))
    per_player_net_apply = nnx.vmap(self._jit_get_network, in_axes=(None, 0, 0), out_axes=(0))
      #Per trajectory and batch dimensions
    vectorized_net_apply = nnx.vmap(nnx.vmap(per_player_net_apply, in_axes=(None, 0, 0), out_axes=(0)), in_axes=(None, 0, 0), out_axes=(0))
    
    #TODO: This will be called again in the real loss. Cannot get rid of the
    # redundant call somehow?
    timestep_pi, _, _, _ = vectorized_net_apply(optimizer.model.actor_critic, wm_timestep.obs, wm_timestep.legal)
    starting_points, start_reaches_is = vectorized_starting_point(jax.lax.stop_gradient(timestep_pi), wm_timestep, wm_prediction_step)
    
    #Flatten the [n_last, batch] into n_last * batch
    starting_points = jax.tree.map(lambda x: jnp.reshape(x, (-1, *x.shape[2:])), starting_points)
    #For the reaches just add a leading 1 dimension for shape consistency
    start_reaches_is = jnp.reshape(start_reaches_is, (-1, *start_reaches_is.shape[2:]))[None, ...]

    img_return, igrad = nnx.value_and_grad(imagination_loss, argnums=(0), has_aux=True)(
      optimizer.model,
      target_optimizer.model,
      prev_network,
      _prev_network,
      trajectory_key, starting_points, start_reaches_is, return_range, alpha, self.config.beta_imagination)
    

    img_loss, (new_range, img_metrics) = img_return
    optimizer.update(igrad)

    rnad_timestep = wm_timestep_to_timestep(wm_timestep, wm_prediction_step, self.is_iig)                                   
    r_return, rgrad = nnx.value_and_grad(real_loss, argnums=(0), has_aux=True)(
      optimizer.model,
      target_optimizer.model,
      prev_network,
      _prev_network,
      rnad_timestep,
      new_range,
      alpha,
      self.config.beta_real
    )

    grad_norms = self.grad_norms.copy()
    if self.config.report_gradnorms:
      grad_keys = self.grad_norms.keys()
      grads = (igrad, rgrad)
      for k, g in zip(grad_keys, grads):
        for n in self.network_keys:
          grad_norms[k][n] = optax.tree.norm(g[n], ord=2)
    r_loss, (new_range, r_metrics) = r_return
    optimizer.update(rgrad)

    rnad_graphdef, state = nnx.split(optimizer.model.actor_critic)
    _, state_target = nnx.split(target_optimizer.model)
    _, state_prev = nnx.split(prev_network)
    _, _state_prev = nnx.split(_prev_network)

    #This grad coupled with vanilla SGD optimizer 
    # is equivalent to the EMA formula (1 - alpha) * state_target + alpha * state
    target_grad = jax.tree.map(lambda a, b: a - b, state_target, state)
    target_optimizer.update(target_grad)
      
    img_metrics.update(r_metrics)
    state_prev, _state_prev = jax.lax.cond(
        update_regularization,
        lambda: (state_target, state_prev),
        lambda: (state_prev, _state_prev))
    prev_network = nnx.merge(rnad_graphdef, state_prev)
    _prev_network = nnx.merge(rnad_graphdef, _state_prev)
    return prev_network, _prev_network, img_metrics, grad_norms, new_range, update_regularization
  

  
  def step(self, wm_timestep: TimeStep, wm_prediction_step:PredictionStepWithLegal, trajectory_key: chex.Array):
    self.prev_network, self._prev_network, self.metrics, self.grad_norms, self.return_range, update_regularization =  self.update_parameters_and_model(self.optimizer, self.target_optimizer, self.prev_network, 
                                                                                                                       self._prev_network, trajectory_key, wm_timestep, wm_prediction_step,
                                                                                                                         self.return_range, self.learner_steps)
    self.learner_steps += 1
    self.policy_switch_steps += int(update_regularization)
  
  def getstate(self):
      return {'return_range': self.return_range,
              'learner_steps': self.learner_steps,
              'policy_steps': self.policy_switch_steps,
              'target_optimizer': nnx.state(self.target_optimizer),
              'prev_network': nnx.state(self.prev_network),
              '_prev_network': nnx.state(self._prev_network)
              }
  
  def setstate(self, state):
    self.return_range = state['return_range']
    self.learner_steps = state['learner_steps']
    nnx.update(self.target_optimizer, state['target_optimizer'])
    self.policy_switch_steps = state['policy_steps']
    nnx.update(self.prev_network, state['prev_network'])
    nnx.update(self._prev_network, state['_prev_network'])

