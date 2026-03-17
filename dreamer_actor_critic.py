import jax
import numpy as np
import jax.numpy as jnp
import chex
import optax
import flax.nnx as nnx

from functools import partial



from distributions import get_bin_log_prob
from ma_rssm import *
from train_utils import *
from typing import Any



def reinforce_loss_with_range(
  pi: chex.Array,
  log_pi: chex.Array,
  returns: chex.Array,
  values: chex.Array, 
  action_oh: chex.Array,
  return_range: chex.Array,
  entropy_eta: float = 0.2 # Regularization factor for the additional entropy regularization
):
  """Compute the Reinforce estimator score, with
  entropy exploration bonus. Crucial! Reinforce really
  needs to add it explicitly here, adding it in TD-estimation
  will not promote exploration, but rather
  even further artifically upweight values. Multiply this by -1 to get loss."""
  advantage = returns - values
  advantage = jax.lax.stop_gradient(advantage / jnp.maximum(1, return_range))

  entropy_bonus = -entropy_eta * jnp.sum(log_pi * pi, axis=-1, keepdims=True)

  reinforce_loss_value = jnp.sum(action_oh * log_pi * advantage, axis=-1, keepdims=True)

  return reinforce_loss_value + entropy_bonus

def td_estimate(
  v: chex.Array,
  valid: chex.Array,
  reward: chex.Array,
  lambda_: float = 1.0,
  gamma: float = 1.0 # Discount factor
):
  """Computes the TD-lambda estimate of the return. Only for the on-policy case.
  (This implementation is esentially V-trace in RNaD without the importance sampling).
  This is designed to work over the entire trajectory without bootstrapping"""
  
  reward = jnp.expand_dims(jnp.stack((reward, -reward), axis=-1), -1)
  
  @chex.dataclass(frozen=True)
  class TDCarry: 
    next_value: chex.Array # Network value in the next timestep 
    delta_v: chex.Array # Propagated delta V in TD-lambda from the next timestep
  
  
  init_carry = TDCarry(
    next_value=jnp.zeros_like(v[-1]),
    delta_v=jnp.zeros_like(v[-1])
  )

  def _td_estimate(carry: TDCarry, x) -> tuple[TDCarry, Any]:
    (v, entropy_reward,valid) = x 
    
    delta_v = (entropy_reward + gamma * carry.next_value - v)
    carry_delta_v = delta_v + lambda_ * gamma * carry.delta_v
    
    v_target = v + carry_delta_v
    
    next_carry = TDCarry(
      next_value=v,
      delta_v=carry_delta_v
    )
    reset_carry = init_carry
  
    reset_v_target = jnp.zeros_like(v_target)
    
    reset_carry = init_carry
    return tree_where(valid, (next_carry, v_target), (reset_carry, reset_v_target))
  
  _, v_target = jax.lax.scan(
    f=_td_estimate,
    init=init_carry,
    xs=(v, reward, valid),
    reverse=True
  )
  return v_target

class DreamerActorCritic():
  
  def __init__(self, game: JaxGame, config: ActorCriticConfig, full_optimizer: nnx.Optimizer, target_optimizer: nnx.Optimizer):
    """A class that has the standard Dreamer
    actor-critic, as described in https://arxiv.org/pdf/2301.04104"""
    self.config = config
    self.optimizer = full_optimizer
    self.target_optimizer = target_optimizer
    self.init(game)

  def init(self, game: JaxGame):

    self.actions = game.num_distinct_actions()
    self.num_players = game.num_players()
    ma_rssm = self.optimizer.model
    self.use_real_infoset = ma_rssm.use_real_infoset
    self.input_size = ma_rssm.infoset_size

    num_last = self.config.num_last
    #If negative, take all for unroll, the Dreamer trajectories
    # will have one more timestep, hence + 1
    if num_last <= 0:
      num_last = game.max_trajectory_lenght_no_chance()
    self.num_last = num_last

    self.return_range = jnp.array(0)
    #self.cached_step = nnx.cached_partial(self.update_paramaters_and_model, self.optimizer, self.target_optimizer)
    self.learner_steps = 0
    self.metrics_keys = ['img_val', 'img_policy', 'real_val']
    if self.config.train_real_policy:
      self.metrics_keys.append('real_policy')
    self.metrics = {k: 0 for k in self.metrics_keys}
    self.grad_norms = {'img': {}, 'real': {}}
    self.network_keys = (*ma_rssm.network_names[-2:], )
  
  


  @partial(nnx.jit, static_argnums=(0,))
  def update_paramaters_and_model(
    self,
    optimizer: nnx.Optimizer,
    target_optimizer: nnx.Optimizer,
    trajectory_key: chex.Array,
    wm_timestep: TimeStep,
    wm_prediction_step: PredictionStepWithLegal,
    return_range: chex.Array,
  ):
    """Compute RNaD loss and use it to perform
    a gradient step of both RNaD and Dreamer."""

    def actor_critic_loss(
      timestep: ActorCriticTimeStep,
      actor_network: ActorNetwork,
      critic_network: CriticNetwork,
      target_network: CriticNetwork,
      return_range: chex.Array,
      compute_actor_loss = True
    ):
      obs = symlog(timestep.obs) if self.use_real_infoset else timestep.obs
      bins = jnp.arange((2 * self.config.bin_range) + 1) - self.config.bin_range
      # Per player vmap
      per_player_net_apply = nnx.vmap(MARSSM.call_net, in_axes=(None, 0, 0), out_axes=(0))
      #Per trajectory and batch dimensions
      vectorized_net_apply = nnx.vmap(nnx.vmap(per_player_net_apply, in_axes=(None, 0, 0), out_axes=(0)), in_axes=(None, 0, 0), out_axes=(0))
      #The critic is centralized
      vectorized_critic_apply = nnx.vmap(nnx.vmap(MARSSM.call_net, in_axes=(None, 0), out_axes=(0)), in_axes=(None, 0), out_axes=(0))
      pi, log_pi, logit = vectorized_net_apply(actor_network, obs, timestep.legal)

      joint_obs = jnp.reshape(obs, (*obs.shape[:-2], -1))

      v_dist_logits = vectorized_critic_apply(critic_network, joint_obs)

      v_target_dist_logits = vectorized_critic_apply(target_network, joint_obs)
       
      v_target = get_value_from_bins(v_target_dist_logits, self.config.bin_range)
      
      expanded_valid = jnp.expand_dims(timestep.valid, (-1, -2))
      #Watch out! Do not call legal_log_policy here, as
      # it assumes a logit and not a softmaxed policy, so we get
      # different results
      mask = (timestep.policy <= 1e-8)
      log_timestep_pi = jnp.log(timestep.policy + mask)
      log_timestep_pi = (1 - mask) * log_timestep_pi

      v_train_target= td_estimate(v_target, expanded_valid, timestep.reward,
                                        self.config.td_lambda, self.config.gamma)
      
      percentiles = get_percentiles_with_mask(v_train_target, expanded_valid, jnp.array([self.config.upper_percentile, self.config.lower_percentile]))
      current_range = (percentiles[0] - percentiles[1])
      new_range = self.config.range_ema_coeff * current_range + (1 - self.config.range_ema_coeff) * return_range
      #v_train_target, q_value = jnp.zeros_like(v), jnp.zeros_like(pi)
      v_loss = -get_bin_log_prob(v_dist_logits, bins, jax.lax.stop_gradient(v_train_target))
      v_loss_value = get_loss_mean_with_mask(v_loss, expanded_valid)
      if compute_actor_loss:    
        
        loss_reinforce = reinforce_loss_with_range(pi, log_pi, v_train_target, v_target, timestep.action, new_range, self.config.eta)
        # The multiplication by -1 is critical here, otherwise we would
        # be minimizing the neurd term, but we want to maximize it.
        reinforce_loss_value = -get_loss_mean_with_mask(loss_reinforce, expanded_valid)
      else:
        reinforce_loss_value = 0

      return v_loss_value + reinforce_loss_value, new_range, v_loss_value, reinforce_loss_value

    def imagination_loss(model: MARSSM,
      target_network: ActorNetwork,
      trajectory_key: chex.Array,
      starting_points: PredictionStepWithLegal,
      return_range: chex.Array,
      beta_imagination: float):
        timestep = jax.lax.stop_gradient(model.imagine_trajectories(trajectory_key, starting_points))
        loss_val, new_range, v_loss, p_loss = actor_critic_loss(timestep, model.actor, model.critic, target_network, return_range) 
        img_keys = self.metrics_keys[:2]
        losses = (v_loss, p_loss)
        metrics = {k: beta_imagination * v for k, v in zip(img_keys, losses)}
        return beta_imagination * loss_val, (new_range, metrics)
    
    def real_loss(model: MARSSM,
      target_network: ActorNetwork,
      timestep: ActorCriticTimeStep,
      return_range: chex.Array,
      beta_real: float):
        loss_val, new_range, v_loss, p_loss = actor_critic_loss(timestep, model.actor, model.critic, target_network, return_range, compute_actor_loss=self.config.train_real_policy)
        real_keys = self.metrics_keys[2:]
        losses = (v_loss, p_loss) if self.config.train_real_policy else (v_loss, )
        metrics = {k: beta_real * v for k, v in zip(real_keys, losses)}
        return beta_real * loss_val, (new_range, metrics)
      
    
    ac_timestep = wm_timestep_to_timestep(wm_timestep, wm_prediction_step, self.use_real_infoset)  
    starting_points = jax.tree.map(lambda x: x[-self.num_last: ].reshape((-1, *x.shape[2:])), wm_prediction_step)
    #starting_points = jax.tree.map(lambda x: x[0].reshape((-1, *x.shape[2:])), wm_prediction_step)
    #jax.tree.map(lambda x: print(x.shape), starting_points)
    img_return, igrad = nnx.value_and_grad(imagination_loss, argnums=(0), has_aux=True)(
      optimizer.model,
      target_optimizer.model,
      trajectory_key, 
      starting_points,
      return_range,
      self.config.beta_imagination)
    
    img_loss, (new_range, img_metrics) = img_return
    optimizer.update(igrad)       
    r_return, rgrad = nnx.value_and_grad(real_loss, argnums=(0), has_aux=True)(
      optimizer.model,
      target_optimizer.model,
      ac_timestep,
      new_range,
      self.config.beta_real
    )
    r_loss, (new_range, r_metrics) = r_return
    optimizer.update(rgrad)

    grad_norms = self.grad_norms.copy()
    if self.config.report_gradnorms:
      grad_keys = self.grad_norms.keys()
      grads = (igrad, rgrad)
      for k, g in zip(grad_keys, grads):
        for n in self.network_keys:
          grad_norms[k][n] = optax.tree.norm(g[n], ord=2)

    critic_graphdef, state = nnx.split(optimizer.model.critic)
    _, state_target = nnx.split(target_optimizer.model)

    img_metrics.update(r_metrics)

    #This grad coupled with vanilla SGD optimizer 
    # is equivalent to the EMA formula (1 - alpha) * state_target + alpha * state
    target_grad = jax.tree.map(lambda a, b: a - b, state_target, state)
    target_optimizer.update(target_grad)

    return img_loss + r_loss, new_range, img_metrics, grad_norms

  
  def step(self, wm_timestep: TimeStep, wm_prediction_step:PredictionStepWithLegal, trajectory_key: chex.Array):
    loss, self.return_range, self.metrics, self.grad_norms =  self.update_paramaters_and_model(self.optimizer, self.target_optimizer, trajectory_key, wm_timestep, wm_prediction_step, self.return_range)
    self.learner_steps += 1
  
  def getstate(self):
    return {'return_range': self.return_range,
            'learner_steps': self.learner_steps,
            'target_optimizer': nnx.state(self.target_optimizer)
            }
  
  def setstate(self, state):
    self.return_range = state['return_range']
    self.learner_steps = state['learner_steps']
    nnx.update(self.target_optimizer, state['target_optimizer'])