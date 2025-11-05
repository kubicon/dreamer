
import jax
import jax.numpy as jnp
import jax.lax as lax

import flax.nnx as nnx
import chex

import numpy as np
import os


from functools import partial

from dreamer_ma import DreamerMA
from networks import *
from train_utils import RNaDConfig, RNaDTimeStep, TimeStep, PredictionStepWithLegal, load_model, symexp, get_loss_mean_with_mask
from rnad_dreamer import EntropySchedule, neurd_loss, v_trace
from distributions import sample_categorical, get_bin_log_prob



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


class RNaDDreamerJoint():
  """A version of RNaDDreamer that performs joint training steps
  Eg. it sends the signal back to the Dreamer model."""
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
    
    self.prev_network = RNaDNetwork(self.iset_size, self.actions, self.config.bin_range, self.config.rnad_network_details[0], self.config.rnad_network_details[1], rngs=self.nnx_rngs)
    self._prev_network = RNaDNetwork(self.iset_size, self.actions, self.config.bin_range, self.config.rnad_network_details[0], self.config.rnad_network_details[1], rngs=self.nnx_rngs)
    
    self.return_range = jnp.array(0)

    self.optimizers = initialize_joint_optimizers(self.world_model.optimizers, self.config, self.iset_size, self.actions, self.nnx_rngs)
    self.cached_step = nnx.cached_partial(self._jit_step_with_model, self.optimizers, self.prev_network, self._prev_network)
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
  def _jit_get_network(self, network: RNaDNetwork, obs, legal):
    pi, v_dist_logits, log_pi, logits = network(obs, legal)
    return pi, v_dist_logits, log_pi, logits
  
  @partial(nnx.jit, static_argnums=(0,))
  def get_v_from_dist(self, v_dist_logits):
    """Reads out the v prediction from the predicted
    logits of the categorical distribution, by multiplying it with the bins."""
    #Implementing the summation order suggestion
    # from https://arxiv.org/pdf/2301.04104 page 18
    bins = jnp.arange((2 * self.config.bin_range) + 1) - self.config.bin_range
    bins = bins.reshape((1,) * (v_dist_logits.ndim - 1) + bins.shape)
    v_probs = nnx.softmax(v_dist_logits)
    pos_bins = bins * (bins >= 0)
    # flip the probs and bins for the negative
    # to ensure summation from small to large in magnitude 
    neg_bins = bins * (bins < 0)
    v_pos_part = jnp.sum(v_probs * pos_bins, axis=-1, keepdims=True)
    v_neg_part = jnp.sum(jnp.flip(v_probs * neg_bins), axis=-1, keepdims=True)
    v = v_pos_part + v_neg_part
    v = symexp(v)
    return v
  
  @partial(nnx.jit, static_argnums=(0,))
  def _jit_get_actor_critic(self, network: RNaDNetwork, obs, legal):
    """Same as get network, but also reads out the v value
    from the categorical distribution"""
    pi, v_dist_logits, log_pi, logits = self._jit_get_network(network, obs, legal)
    v = self.get_v_from_dist(v_dist_logits)
    return pi, v, log_pi, logits
  
  @partial(nnx.jit, static_argnums=(0,))
  def _jit_get_policy(self, network: RNaDNetwork, obs, legal) -> chex.Array:
    return network(obs, legal)[0]
  
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
  
  @partial(nnx.jit, static_argnums=0)
  def get_policy_both(self, network: RNaDNetwork, joint_obs, joint_legal):
    #vmap over the player dimension
    players_get_policy = nnx.vmap(self._jit_get_policy, in_axes=(None, 0, 0), out_axes=(0))
    pi = players_get_policy(network, joint_obs, joint_legal)
    return pi
  
  #TODO: Is it necessary to pass all the models explicitly like this?
  @partial(nnx.jit, static_argnums=0)
  def sample_trajectories(self, key, starting_points: PredictionStepWithLegal, rnad_network: RNaDNetwork, sequence_model: SequenceModel, dynamics: DynamicsPredictor,
                        predictor: Predictor, legal_network: LegalActionsNetwork, encoder: JointIsetEncoder, p1_iset_decoder:IsetDecoder,
                        p2_iset_decoder: IsetDecoder) ->RNaDTimeStep:
    keys = jax.random.split(key, self.config.batch_size)
    batch_sample_trajectory = nnx.vmap(self.sample_trajectory, in_axes=(0, 0, None, None, None, None, None, None, None, None), out_axes=1) 
    return batch_sample_trajectory(keys, starting_points, rnad_network, sequence_model, dynamics, predictor, legal_network, encoder, p1_iset_decoder, p2_iset_decoder)


  #TODO: Is it necessary to pass all the models explicitly like this?
  @partial(nnx.jit, static_argnums=0)
  def sample_trajectory(self, key, starting_point: PredictionStepWithLegal, rnad_network: RNaDNetwork, sequence_model: SequenceModel, dynamics: DynamicsPredictor,
                        predictor: Predictor, legal_network: LegalActionsNetwork, encoder: JointIsetEncoder, p1_iset_decoder:IsetDecoder,
                        p2_iset_decoder: IsetDecoder) ->RNaDTimeStep:
    #init_sample_key, trajectory_key, = jax.random.split(key)
    trajectory_key = jax.random.split(key, self.trajectory_max)
  
    
    
    @chex.dataclass(frozen=True)
    class SampleTrajectoryCarry:
      hidden_state:chex.Array
      deter_state: chex.Array
      legal_actions: chex.Array
      terminal: bool
      
    init_carry = SampleTrajectoryCarry(
      hidden_state = starting_point.hidden_state,
      deter_state = starting_point.deter_state, #TODO: Take the one that Dreamer sampled, or sample anew?
      legal_actions = (nnx.sigmoid(starting_point.legal_logit) >= self.config.legal_threshold).astype(jnp.int8), 
      terminal = (nnx.sigmoid(starting_point.done_logit) >= self.config.terminal_threshold)[0]
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
      normalization = jnp.sum(carry.legal_actions, axis=-1, keepdims=True)
      uniform_pi = carry.legal_actions / (normalization + (normalization == 0))
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
      next_reward, next_terminal, next_legal = self.world_model.get_predictor(predictor, legal_network, next_hidden, next_deter, self.config.terminal_threshold, self.config.legal_threshold)
      next_terminal = jnp.logical_or(carry.terminal, next_terminal)
      # Dreamer can produce all actions to be invalid
      # even when one of the players does not act, he always has one legal
      # NOOP action. So, if one of the players has all actions invalid, then
      # the state is not valid
      valid = jnp.logical_and(jnp.logical_not(carry.terminal), jnp.all(normalization > 0))
      timestep = RNaDTimeStep(
        obs = obs,
        legal = carry.legal_actions.astype(jnp.int8),
        action = action_oh.astype(jnp.int8),
        policy = pi,
        reward = next_reward,
        valid = valid
      )
      new_carry = SampleTrajectoryCarry(
        hidden_state = next_hidden,
        deter_state = next_deter,
        legal_actions=jnp.where(next_terminal, self.example_timestep.legal, next_legal),
        terminal = jnp.logical_or(next_terminal, jnp.logical_not(valid)),
      )
         
      timestep = jax.tree.map(lambda t, f: jnp.where(valid, t, f).astype(t.dtype), timestep, self.example_timestep)
      return new_carry, timestep
    _, timestep = _sample_trajectory(init_carry, trajectory_key, rnad_network, sequence_model, dynamics, predictor, legal_network, p1_iset_decoder, p2_iset_decoder)
    #[Trajectory, ...]
    return timestep
  
  
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
    dreamer_timestep: TimeStep,
    dreamer_prediction_step: PredictionStepWithLegal,
    return_range: chex.Array,
    alpha,
    update_net 
  ):
    """Compute RNaD loss and use it to perform
    a gradient step of both RNaD and Dreamer."""

    def rnad_loss(
      timestep: RNaDTimeStep,
      rnad_network: RNaDNetwork,
      target_network: RNaDNetwork,
      prev_network: RNaDNetwork,
      _prev_network: RNaDNetwork,
      start_reaches_is: chex.Array,
      return_range: chex.Array,
      alpha: float,
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
       
      v_target = self.get_v_from_dist(v_target_dist_logits)
      # This creates the regularization term for rewards
      regularized_term = log_pi - (alpha * log_pi_prev + (1 - alpha) * log_pi_prev_) 
      
      expanded_valid = jnp.expand_dims(timestep.valid, (-1, -2))
      
      v_train_target, q_value = v_trace(v_target, expanded_valid, timestep.policy, pi, regularized_term, timestep.action, timestep.reward,
                                        self.config.lambda_vtrace, self.config.c_vtrace, self.config.rho_vtrace,
                                        self.config.eta, self.config.vtrace_eta, self.config.gamma_vtrace)
      
      current_range = (jnp.percentile(q_value, 0.95) - jnp.percentile(q_value, 0.05))
      new_range = 0.01 * current_range + 0.99 * return_range
      #v_train_target, q_value = jnp.zeros_like(v), jnp.zeros_like(pi)
      # We multiply by 2, since each player acts
      normalization = jnp.sum(timestep.valid) * 2
      #v_loss = jnp.sum((expanded_valid * (v - lax.stop_gradient(v_train_target)) ** 2)) / (normalization + (normalization == 0))
      v_loss = -get_bin_log_prob(v_dist_logits, bins, jax.lax.stop_gradient(v_train_target))
      v_loss_value = jnp.sum(v_loss * expanded_valid) / (normalization + (normalization == 0))

      # Each Q is multiplied by product of importance_sampling of opponent and inverted sampling policy by the acting player.
      # This computes counterfactual importance sampling
      sampling_policy = jnp.sum(timestep.policy * timestep.action, axis=-1, keepdims=True) * expanded_valid + (1 - expanded_valid)
      network_policy = jnp.sum(pi * timestep.action, axis=-1, keepdims=True)* expanded_valid + (1 - expanded_valid)
      
      # We do not take into account the player reaches, since infoset is always reached with the same prob
      sampling_policy = jnp.prod(sampling_policy, axis=-2, keepdims=True)
      
      importance_sampling = network_policy / sampling_policy
      
      importance_sampling = jnp.concatenate((start_reaches_is, importance_sampling[:-1]), axis=0)
      importance_sampling = jnp.cumprod(importance_sampling, axis=0)
      importance_sampling = jnp.flip(importance_sampling, axis=-2)
      
      
      loss_neurd = neurd_loss_with_range(logit, pi, q_value, timestep.legal, importance_sampling, new_range)
      
      # The multiplication by -1 is critical here, otherwise we would
      # be minimizing the neurd term, but we want to maximize it.
      neurd_loss_value = -jnp.sum(loss_neurd * expanded_valid) / (normalization + (normalization == 0))
      return v_loss_value + neurd_loss_value, new_range

    def imagination_loss(rnad_network: RNaDNetwork,
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
      starting_points: PredictionStepWithLegal,
      start_reaches_is: chex.Array,
      return_range: chex.Array,
      alpha: float,
      beta_imagination: float):
        timestep = self.sample_trajectories(trajectory_key, starting_points, rnad_network, 
                                                sequence_model,
                                                dynamics,
                                                predictor,
                                                legal_network,
                                                encoder,
                                                p1_iset_decoder,
                                                p2_iset_decoder)
        loss_val, new_range = rnad_loss(timestep, rnad_network, target_network, prev_network, _prev_network, start_reaches_is, return_range, alpha) 
        return beta_imagination * loss_val, new_range
    
    def real_loss(rnad_network: RNaDNetwork,
      target_network: RNaDNetwork,
      prev_network: RNaDNetwork,
      _prev_network: RNaDNetwork,
      timestep: RNaDTimeStep,
      return_range: chex.Array,
      alpha: float,
      beta_real: float):
        start_reaches_is = jnp.ones((1, *timestep.policy.shape[1:-1], 1))
        loss_val, new_range = rnad_loss(timestep, rnad_network, target_network, prev_network, _prev_network, start_reaches_is, return_range, alpha)
        return beta_real * loss_val, new_range
      
    starting_key, trajectory_key = jax.random.split(trajectory_key)
    starting_key = jax.random.split(starting_key, self.world_model.config.batch_size)
    def choose_starting_point(network_pi:chex.Array, timestep: TimeStep, prediction_step: PredictionStepWithLegal, key):
      """Choose a starting point that is not invalid or terminal in the timestep
      uniformly. Chooses over the trajectory dimension and should be 
      vmaped over the batch dimension"""
      validity_mask = timestep.valid * ~(timestep.terminal)
      normalization = jnp.sum(validity_mask)
      p = validity_mask / (normalization + (normalization == 0))
      chosen_idx = jax.random.choice(key, p.shape[0], p = p)
      previous_steps = jnp.arange(timestep.policy.shape[0]) < chosen_idx
      #We also need to compute the importance sampling
      # for the player reaches, since we do not start at
      # the beggining of the trajectory.
      #Shape [T, Pl, 1]
      timestep_pi = jnp.sum(timestep.policy* timestep.action, axis=-1, keepdims=True)
      #Shape [T, 1, 1]
      timestep_joint_pi = jnp.prod(timestep_pi, axis=-2, keepdims=True)
      #[T, Pl, A]
      #[T, Pl, 1]
      network_pi = jnp.sum(network_pi * timestep.action, axis=-1, keepdims=True)
      masked_is = jnp.where(previous_steps[..., None, None], network_pi / timestep_joint_pi, 1)
      #[1, Pl, 1]
      start_reaches_is = jnp.prod(masked_is, axis=0, keepdims=True)
      sampled_start = jax.tree_util.tree_map(lambda x: jnp.take_along_axis(x, chosen_idx.reshape((1,) * x.ndim), axis=0).squeeze(0), prediction_step)
      #sampled_start = jax.tree_util.tree_map(lambda x: x[0], prediction_step)
      return sampled_start, start_reaches_is
    vectorized_starting_point = jax.vmap(choose_starting_point, in_axes=(1, 1,1, 0), out_axes=(0, 1))
    per_player_net_apply = nnx.vmap(self._jit_get_network, in_axes=(None, 0, 0), out_axes=(0))
      #Per trajectory and batch dimensions
    vectorized_net_apply = nnx.vmap(nnx.vmap(per_player_net_apply, in_axes=(None, 0, 0), out_axes=(0)), in_axes=(None, 0, 0), out_axes=(0))
    
    #TODO: This will be called again in the real loss. Cannot get rid of the
    # redundant call somehow?
    timestep_pi, _, _, _ = vectorized_net_apply(optimizers.rnad_optimizer.model, dreamer_timestep.obs, dreamer_timestep.legal)
    starting_points, start_reaches_is = vectorized_starting_point(jax.lax.stop_gradient(timestep_pi),dreamer_timestep, dreamer_prediction_step, starting_key)
  
    img_return, grad = nnx.value_and_grad(imagination_loss, argnums=(0), has_aux=True)(
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
      _prev_network,
      trajectory_key, starting_points, start_reaches_is, return_range, alpha, self.config.beta_imagination)
    
    img_loss, new_range = img_return
    # optimizers.rnad_optimizer.update(grads[0])
    # optimizers.sequence_optimizer.update(grads[1])
    # optimizers.dynamics_optimizer.update(grads[2])
    # optimizers.predictor_optimizer.update(grads[3])
    # optimizers.legal_actions_optimizer.update(grads[4])
    # optimizers.encoder_optimizer.update(grads[5])
    # optimizers.p1_decoder_optimizer.update(grads[6])
    # optimizers.p2_decoder_optimizer.update(grads[7])
    optimizers.rnad_optimizer.update(grad)

    #Do not forget that the Dreamer timestep rewards and terminal are shifted by 1 in time
    # compared to our desired RNaD timesteps.
    # We also do not want the last step, since that is always a terminal state
    rnad_timestep = RNaDTimeStep(obs = dreamer_timestep.obs[:-1],
                                 legal = dreamer_timestep.legal[:-1],
                                 action = dreamer_timestep.action[:-1],
                                 policy = dreamer_timestep.policy[:-1],
                                 reward = dreamer_timestep.reward[1:],
                                 valid = jnp.logical_and(~dreamer_timestep.terminal[:-1], dreamer_timestep.valid[:-1]))                                      
    r_return, grad = nnx.value_and_grad(real_loss, argnums=(0), has_aux=True)(
      optimizers.rnad_optimizer.model,
      optimizers.rnad_target_optimizer.model,
      prev_network,
      _prev_network,
      rnad_timestep,
      new_range,
      alpha,
      self.config.beta_real
    )
    r_loss, new_range = r_return
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
    return prev_network, _prev_network, img_loss, r_loss, new_range
  
  @partial(nnx.jit, static_argnums=(0))
  def _jit_step_with_model(self, optimizers: JointOptimizers, prev_network: RNaDNetwork, _prev_network: RNaDNetwork
                ,trajectory_key, dreamer_timestep: TimeStep, dreamer_prediction_step:PredictionStepWithLegal, return_range:chex.Array, learner_steps: int):
    #
    
    alpha, update_regularization = self._entropy_schedule(learner_steps)
    prev_network, _prev_network, img_loss, r_loss, new_range = self.update_parameters_and_model(
      optimizers, prev_network, _prev_network, trajectory_key, dreamer_timestep, dreamer_prediction_step, return_range, alpha, update_regularization
    )
    return prev_network, _prev_network, img_loss, r_loss, new_range, update_regularization

  
  def step(self, dreamer_timestep: TimeStep, dreamer_prediction_step:PredictionStepWithLegal):
    trajectory_key = self.get_next_rng_key()
    #self.prev_network, self._prev_network, img_loss, r_loss, update_regularization =  self._jit_step_with_model(self.optimizers, self.prev_network, self._prev_network, trajectory_key, dreamer_timestep, dreamer_prediction_step, self.return_range, self.learner_steps)
    self.prev_network, self._prev_network, img_loss, r_loss, self.return_range, update_regularization = self.cached_step(trajectory_key, dreamer_timestep, dreamer_prediction_step, self.return_range, self.learner_steps)
    self.learner_steps += 1
    self.policy_switch_steps += int(update_regularization)
    return img_loss, r_loss


  def __getstate__(self):
    return {"config": self.config,
            "world_model": self.world_model,
            "optimizers": nnx.state(self.optimizers),
            "prev_network": nnx.state(self.prev_network),
            "_prev_network": nnx.state(self._prev_network),
            "steps": self.learner_steps,
            "trajectory_key": self.rng_key,}
  
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
    #Necessary to correctly continue to train on the updated state of the optimizers
    self.cached_step = nnx.cached_partial(self._jit_step_with_model, self.optimizers, self.prev_network, self._prev_network)
    #Explicitly reinitialize the world model optimizers
    # to make sure that the deserialization does not make them point to
    # two different objects.
    wm_optimizers = DreamerMAOptimizers(sequence_optimizer= self.optimizers.sequence_optimizer,
          encoder_optimizer = self.optimizers.encoder_optimizer,
          p1_decoder_optimizer = self.optimizers.p1_decoder_optimizer,
          p2_decoder_optimizer = self.optimizers.p2_decoder_optimizer,
          dynamics_optimizer = self.optimizers.dynamics_optimizer,
          predictor_optimizer = self.optimizers.predictor_optimizer,
          legal_actions_optimizer = self.optimizers.legal_actions_optimizer)

    self.world_model.optimizers = wm_optimizers
    #Do not forget to also reset the cached step of the world model
    self.world_model.cached_step = nnx.cached_partial(self.world_model.world_model_train, self.world_model.optimizers) 
  
  
  
def main():
  cards = 3
  network_seed = 99
  trajectory_seed = 99
  restore_step = 1000
  model_path = f"trained_networks/goofspiel_{cards}/seed{trajectory_seed}/network_seed{network_seed}/step_{restore_step}.pkl"
  model_path = os.getcwd() + "/" + model_path
  model = load_model(model_path)
  config = RNaDConfig(batch_size = 4)
  solver = RNaDDreamerJoint(dreamer_model=model, config=config)
  for _ in range(10):
    solver.step()
  
  
if __name__ == "__main__":
  main()