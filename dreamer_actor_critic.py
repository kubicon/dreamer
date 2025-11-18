import jax
import numpy as np
import jax.numpy as jnp
import chex
import flax.nnx as nnx

from functools import partial


from games.jax_game import InformationType

from dreamer_ma import DreamerMA
from distributions import sample_categorical, get_bin_log_prob
from networks import *
from train_utils import *
from typing import Any



def reinforce_loss_with_range(
  log_pi: chex.Array,
  policy: chex.Array,
  q_values: chex.Array, 
  action_oh: chex.Array,
  return_range: chex.Array
):
  """Compute the Reinforce estimator score. Multiply this by -1 to get loss.
  Expects the estimates to already have the entropy bonus accounted for"""
  advantage = q_values - jnp.sum(policy * q_values, axis=-1, keepdims=True)
  advantage = jax.lax.stop_gradient(advantage / jnp.maximum(1, return_range))

  
  reinforce_loss_value = jnp.sum(action_oh * log_pi * advantage, axis=-1, keepdims=True)
  

  return reinforce_loss_value

def td_estimate(
  v: chex.Array,
  valid: chex.Array,
  action_oh: chex.Array,
  sampling_policy: chex.Array,
  log_sampling_policy: chex.Array,
  reward: chex.Array, # Still not regularized
  lambda_: float = 1.0, # Lambda parameter for V-trace
  entropy_eta: float = 0.2, # Regularization factor for the additional entropy regularization
  gamma: float = 1.0 # Discount factor
):
  """Computes the TD-lambda estimate of the return. Only for the on-policy case.
  (This implementation is esentially V-trace in RNaD without the importance sampling).
  This is designed to work over the entire trajectory without bootstrapping"""
  
  
  
  #[Trajectory, Batch, Player]
  regularization_entropy = -entropy_eta * jnp.sum(sampling_policy * log_sampling_policy, axis=-1)
  
  #[Trajectory, Batch]
  # The MinMaxEnt objective
  both_player_entropy = (regularization_entropy[..., 0]  - regularization_entropy[..., 1])

  #[Trajectory, Batch]
  entropy_reward = reward + both_player_entropy
  #[Trajectory, Batch, Player, 1]
  entropy_reward = jnp.expand_dims(jnp.stack((entropy_reward, -entropy_reward), axis=-1), -1)
  
  #[Trajectory, Batch, Player]
  q_reward = jnp.stack((reward, -reward), axis=-1) + regularization_entropy[..., (0, 1)]
  
  q_reward = jnp.expand_dims(q_reward, -1)
  
  
  
  @chex.dataclass(frozen=True)
  class TDCarry: 
    next_value: chex.Array # Network value in the next timestep 
    delta_v: chex.Array # Propagated delta V in TD-lambda from the next timestep
  
  
  init_carry = TDCarry(
    next_value=jnp.zeros_like(v[-1]),
    delta_v=jnp.zeros_like(v[-1])
  )

  def _v_trace(carry: TDCarry, x) -> tuple[TDCarry, Any]:
    (v, q_reward, entropy_reward,valid, action_oh) = x 
    # reward_uncorrected = reward + gamma * carry.reward_uncorrected + entropy
    # discounted_reward = reward + gamma * carry.reward
    
    delta_v = (entropy_reward + gamma * carry.next_value - v)
    carry_delta_v = delta_v + lambda_ * gamma * carry.delta_v
    
    v_target = v + carry_delta_v
    
    
    q_value = v + action_oh *  (q_reward + gamma * (carry.next_value + carry.delta_v) - v )
    
    next_carry = TDCarry(
      next_value=v,
      delta_v=carry_delta_v
    )
    reset_carry = init_carry
  
    reset_v_target = jnp.zeros_like(v_target)
    reset_q_value = jnp.zeros_like(q_value) 
    
    reset_carry = init_carry
    return tree_where(valid, (next_carry, (v_target, q_value)), (reset_carry, (reset_v_target, reset_q_value)))

  _, (v_target, q_value) = jax.lax.scan(
    f=_v_trace,
    init=init_carry,
    xs=(v, q_reward, entropy_reward, valid, action_oh),
    reverse=True
  )
  return v_target, q_value

class DreamerActorCritic():
  
  def __init__(self, config: ActorCriticConfig, world_model: DreamerMA):
    """A class that has the standard Dreamer
    actor-critic, as described in https://arxiv.org/pdf/2301.04104"""
    self.config = config
    self.world_model = world_model
    self.key = jax.random.key(self.config.seed)
    self.init()

  def init(self):

    self.actions = self.world_model.action_dimension
    #Unlike world model, we operate with rewards defined 
    # as (state, action, next_state) and only care how to
    # act in non-terminal states, hence we end one turn before terminal
    self.non_chance_trajectory_max = self.world_model.non_chance_trajectory_max - 1
    self.num_players = self.world_model.game.num_players()
    self.iset_size = self.world_model.game.information_state_tensor_shape()

    # Use iset for IIGs use iset, otherwise use the model state [deterministic_state, hidden_state]
    self.use_iset = self.world_model.game.information_type() == InformationType.IIG

    self.example_hidden = jnp.zeros(self.world_model.hidden_state_size)
    self.example_categorical = jnp.zeros((self.world_model.config.encoded_classes, self.world_model.config.encoded_categories))
    
    self.rng_key = jax.random.key(self.config.seed)
    rngs = nnx.Rngs(jax.random.key(self.config.network_seed))

    
    
    self.example_timestep = self.default_timestep()
    self.return_range = jnp.array(0)

    self.optimizers = initialize_actor_critic_optimizers(self.config, self.input_features, self.actions, rngs)
    self.cached_step = nnx.cached_partial(self._jit_step_with_model, self.optimizers, self.world_model.optimizers)
    self.learner_steps = 0
    self.policy_switch_steps = 0
  
  
  def default_timestep(self):
    if self.use_iset:
      obs = np.zeros(self.iset_size, dtype=np.float32)
      self.input_features = self.iset_size
    else:
      model_state_features = self.world_model.config.encoded_classes * self.world_model.config.encoded_categories + self.world_model.hidden_state_size + self.num_players
      obs = np.zeros(model_state_features, dtype=np.float32)
      self.input_features = model_state_features

    legal = np.ones((1, self.actions), dtype=np.int8)
    action = np.ones((1, self.actions), dtype=np.float32)
    policy = np.ones((1,self.actions), dtype=np.float32)
    valid = np.array(0, dtype=np.bool)
    reward = np.array(0, dtype=np.float32)
    
    ts = ActorCriticTimeStep(
      valid = valid,
      obs = obs,
      legal = legal,
      action = action, 
      policy = policy,
      reward = reward
    )
    return ts
    
  
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
  def _jit_get_actor(self, actor_network: ActorNetwork, input, legal) -> chex.Array:
    return actor_network(input, legal)
  
  @partial(nnx.jit, static_argnums=(0,))
  def _jit_get_critic(self, critic_network: CriticNetwork, input) ->chex.Array:
    return critic_network(input)
  
  @partial(nnx.jit, static_argnums=(0,))
  def _jit_get_policy(self, actor_network: ActorNetwork, input, legal) -> chex.Array:
    return actor_network(input, legal)[0]
  
  @partial(nnx.jit, static_argnums=(0,))
  def _jit_get_actor_critic(self, actor_network: ActorNetwork, critic_network: CriticNetwork, input, legal):
    """Gets the predictions from both actor and critic networks
    and also reads out the value predicted by critic from the distribution."""
    pi, log_pi, logits = self._jit_get_actor(actor_network, input, legal)
    v_dist_logits = self._jit_get_critic(critic_network, input)
    v = self.get_v_from_dist(v_dist_logits)
    return pi, v, log_pi, logits
  
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
  
  @partial(nnx.jit, static_argnums=0)
  def get_obs(self, p1_decoder:IsetDecoder, p2_decoder: IsetDecoder, hidden_state: chex.Array, deter_state:chex.Array):
    if self.use_iset:
      #TODO: For now, iset decoder is used to create trajectories 
      # trained on the "original" isets. This might be changed later
      p1_iset = p1_decoder(hidden_state, deter_state)
      p2_iset = p2_decoder(hidden_state, deter_state)
      obs = jnp.stack([p1_iset, p2_iset], axis=0)
      return obs
    flat_deter = deter_state.reshape((deter_state.shape[:-2], -1))
    players_oh = jnp.eye(self.num_players)
    players_oh = jnp.reshape(players_oh, (1, ) * (flat_deter.ndim - 1) + players_oh.shape)
    model_state = jnp.concatenate([hidden_state, flat_deter], axis=-1)
    player_model_state = jnp.concatenate([jnp.stack([model_state, model_state], axis=-2), players_oh], axis=-1)
    return player_model_state
  
  #TODO: Is it necessary to pass all the models explicitly like this?
  @partial(nnx.jit, static_argnums=0)
  def sample_trajectories(self, key, starting_points: PredictionStepWithLegal, actor_network: ActorNetwork, sequence_model: SequenceModel, dynamics: DynamicsPredictor,
                        predictor: Predictor, legal_network: LegalActionsNetwork, encoder: JointIsetEncoder, p1_iset_decoder:IsetDecoder,
                        p2_iset_decoder: IsetDecoder) ->ActorCriticTimeStep:
    keys = jax.random.split(key, self.config.batch_size)
    batch_sample_trajectory = nnx.vmap(self.sample_trajectory, in_axes=(0, 0, None, None, None, None, None, None, None, None), out_axes=1) 
    return batch_sample_trajectory(keys, starting_points, actor_network, sequence_model, dynamics, predictor, legal_network, encoder, p1_iset_decoder, p2_iset_decoder)


  #TODO: Is it necessary to pass all the models explicitly like this?
  @partial(nnx.jit, static_argnums=0)
  def sample_trajectory(self, key, starting_point: PredictionStepWithLegal, actor_network: ActorNetwork, sequence_model: SequenceModel, dynamics: DynamicsPredictor,
                        predictor: Predictor, legal_network: LegalActionsNetwork, encoder: JointIsetEncoder, p1_iset_decoder:IsetDecoder,
                        p2_iset_decoder: IsetDecoder) ->ActorCriticTimeStep:
    #init_sample_key, trajectory_key, = jax.random.split(key)
    trajectory_key = jax.random.split(key, self.non_chance_trajectory_max)
  
    
    
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
    def _sample_trajectory(carry: SampleTrajectoryCarry, key , actor_network: ActorNetwork, sequence_model: SequenceModel, dynamics: DynamicsPredictor, 
                        predictor: Predictor, legal_network: LegalActionsNetwork, p1_iset_decoder:IsetDecoder,
                        p2_iset_decoder: IsetDecoder) -> tuple[SampleTrajectoryCarry, chex.Array]:
      
      
      obs = self.get_obs(p1_iset_decoder, p2_iset_decoder, carry.hidden_state, carry.deter_state)
      #get policy 
      pi = self.get_policy_both(actor_network, obs, carry.legal_actions)
      #uniform mix to the policy
      normalization = jnp.sum(carry.legal_actions, axis=-1, keepdims=True)
      # For each player samples a single action
      
      action_sample_key, state_sample_key = jax.random.split(key)
      action_sample_keys = jax.random.split(action_sample_key, self.num_players)
      action, action_oh = vectorized_sample_action(action_sample_keys, pi)
      
      next_hidden = sequence_model(carry.hidden_state, carry.deter_state, action_oh)
      next_stoch = dynamics(next_hidden)
      next_deter = sample_categorical(next_stoch, state_sample_key, sample_threshold=self.config.state_sample_threshold)
      next_reward, next_terminal, next_legal = self.world_model.get_predictor(predictor, legal_network, next_hidden, next_deter, self.config.terminal_threshold, self.config.legal_threshold)
      next_terminal = jnp.logical_or(carry.terminal, next_terminal)
      # Dreamer can produce all actions to be invalid
      # even when one of the players does not act, he always has one legal
      # NOOP action. So, if one of the players has all actions invalid, then
      # the state is not valid
      valid = jnp.logical_and(jnp.logical_not(carry.terminal), jnp.all(normalization > 0))
      timestep = ActorCriticTimeStep(
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
         
      timestep = tree_where(valid, timestep, self.example_timestep)
      return new_carry, timestep
    _, timestep = _sample_trajectory(init_carry, trajectory_key, actor_network, sequence_model, dynamics, predictor, legal_network, p1_iset_decoder, p2_iset_decoder)
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
  

  @partial(nnx.jit, static_argnums=0)
  def dreamer_timestep_to_timestep(self, dreamer_timestep: TimeStep, dreamer_prediction_step: PredictionStepWithLegal) ->ActorCriticTimeStep:
    #Do not forget that the Dreamer timestep rewards and terminal
    # are w.r.t. the current state. We want
    # reward for playing an action in the current state, not for getting to it
    # so, they are shifted by 1 forward in time
    # compared to our desired RNaD timesteps.
    # We also do not want the last step, since that is always a terminal state
    legal = dreamer_timestep.legal[:-1]
    action = dreamer_timestep.action[:-1]
    policy = dreamer_timestep.policy[:-1]
    reward = dreamer_timestep.reward[1:]
    valid = jnp.logical_and(~dreamer_timestep.terminal[:-1], dreamer_timestep.valid[:-1])
    
    #if we shouldnt use infosets we replace obs
    # with the predicted model states
    if self.use_iset:
      obs = dreamer_timestep.obs[:-1]
    else:
      #These are sampled from the encoder produced stochastic states
      flat_deters = jnp.reshape(dreamer_prediction_step.deter_state, (*dreamer_prediction_step.deter_state.shape[-2], -1))  
      players_oh = jnp.eye(self.num_players)
      players_oh = jnp.reshape(players_oh, (1, ) * (flat_deters.ndim - 1) + players_oh.shape)
      model_states = jnp.concatenate([dreamer_prediction_step.hidden_state, flat_deters], axis=-1)
      player_model_states = jnp.concatenate([jnp.stack([model_states, model_states], axis=-2), players_oh], axis=-1)
      obs = player_model_states
    ac_timestep = ActorCriticTimeStep(obs = obs,
                                      legal=legal,
                                      action=action,
                                      policy = policy,
                                      reward = reward,
                                      valid=valid)
    return ac_timestep


  @partial(nnx.jit, static_argnums=(0,))
  def update_parameters_and_model(
    self,
    optimizers: ActorCriticOptimizers,
    world_model_optimizers: DreamerMAOptimizers,
    trajectory_key: chex.Array,
    dreamer_timestep: TimeStep,
    dreamer_prediction_step: PredictionStepWithLegal,
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
    ):
      
      bins = jnp.arange((2 * self.config.bin_range) + 1) - self.config.bin_range
      # Per player vmap
      per_player_actor_apply = nnx.vmap(self._jit_get_actor, in_axes=(None, 0, 0), out_axes=(0))
      per_player_critic_apply = nnx.vmap(self._jit_get_critic, in_axes=(None, 0), out_axes=(0))
      #Per trajectory and batch dimensions
      vectorized_actor_apply = nnx.vmap(nnx.vmap(per_player_actor_apply, in_axes=(None, 0, 0), out_axes=(0)), in_axes=(None, 0, 0), out_axes=(0))
      vectorized_critic_apply = nnx.vmap(nnx.vmap(per_player_critic_apply, in_axes=(None, 0), out_axes=(0)), in_axes=(None, 0), out_axes=(0))

      pi, log_pi, logit = vectorized_actor_apply(actor_network, timestep.obs, timestep.legal)

      v_dist_logits = vectorized_critic_apply(critic_network, timestep.obs)

      v_target_dist_logits = vectorized_critic_apply(target_network, timestep.obs)
       
      v_target = self.get_v_from_dist(v_target_dist_logits)
      
      expanded_valid = jnp.expand_dims(timestep.valid, (-1, -2))

      log_timestep_pi = legal_log_policy(timestep.policy, timestep.legal)
      
      v_train_target, q_value = td_estimate(v_target, expanded_valid, timestep.action, timestep.policy, log_timestep_pi, timestep.reward,
                                        self.config.td_lambda, self.config.eta, self.config.gamma)
      
      q_mask = expanded_valid * timestep.legal
      percentiles = get_percentiles_with_mask(q_value, q_mask, jnp.array([self.config.upper_percentile, self.config.lower_percentile]))
      current_range = (percentiles[0] - percentiles[1])
      new_range = self.config.range_ema_coeff * current_range + (1 - self.config.range_ema_coeff) * return_range
      #v_train_target, q_value = jnp.zeros_like(v), jnp.zeros_like(pi)
      v_loss = -get_bin_log_prob(v_dist_logits, bins, jax.lax.stop_gradient(v_train_target))
      v_loss_value = get_loss_mean_with_mask(v_loss, expanded_valid)    
      
      loss_reinforce = reinforce_loss_with_range(log_pi, pi, q_value, timestep.action, new_range)
      
      # The multiplication by -1 is critical here, otherwise we would
      # be minimizing the neurd term, but we want to maximize it.
      reinforce_loss_value = -get_loss_mean_with_mask(loss_reinforce, expanded_valid)

      return v_loss_value + reinforce_loss_value, new_range

    def imagination_loss(actor_network: ActorNetwork,
      critic_network: CriticNetwork,
      sequence_model: SequenceModel, 
      dynamics: DynamicsPredictor,
      predictor: Predictor, 
      legal_network: LegalActionsNetwork, 
      encoder: JointIsetEncoder, 
      p1_iset_decoder:IsetDecoder,
      p2_iset_decoder: IsetDecoder,
      target_network: RNaDNetwork,
      trajectory_key: chex.Array,
      starting_points: PredictionStepWithLegal,
      return_range: chex.Array,
      beta_imagination: float):
        timestep = self.sample_trajectories(trajectory_key, starting_points, actor_network, 
                                                sequence_model,
                                                dynamics,
                                                predictor,
                                                legal_network,
                                                encoder,
                                                p1_iset_decoder,
                                                p2_iset_decoder)
        loss_val, new_range = actor_critic_loss(timestep, actor_network, critic_network, target_network, return_range) 
        return beta_imagination * loss_val, new_range
    
    def real_loss(actor_network: ActorNetwork,
      critic_network: CriticNetwork,
      target_network: RNaDNetwork,
      timestep: ActorCriticTimeStep,
      return_range: chex.Array,
      beta_real: float):
        loss_val, new_range = actor_critic_loss(timestep, actor_network, critic_network, target_network, return_range)
        return beta_real * loss_val, new_range
      
    starting_key, selector_key, trajectory_key = jax.random.split(trajectory_key, 3)
    #First select the trajectories from which we will be unrolling
    traj_indices = jax.random.randint(selector_key, self.config.batch_size, 0, self.world_model.config.batch_size)
    timestep_for_imagination = tree_index(dreamer_timestep, traj_indices, axis=1)
    predictions_for_imagination = tree_index(dreamer_prediction_step, traj_indices, axis=1)
    #Then select starting points in these trajectories
    starting_key = jax.random.split(starting_key, self.config.batch_size)
    def choose_starting_point(timestep: TimeStep, prediction_step: PredictionStepWithLegal, key):
      """Choose a starting point that is not invalid or terminal in the timestep
      uniformly. Chooses over the trajectory dimension and should be 
      vmaped over the batch dimension"""
      validity_mask = timestep.valid * ~(timestep.terminal)
      normalization = jnp.sum(validity_mask)
      p = validity_mask / (normalization + (normalization == 0))
      chosen_idx = jax.random.choice(key, p.shape[0], p = p)
      sampled_start = jax.tree_util.tree_map(lambda x: jnp.take_along_axis(x, chosen_idx.reshape((1,) * x.ndim), axis=0).squeeze(0), prediction_step)
      #sampled_start = jax.tree_util.tree_map(lambda x: x[0], prediction_step)
      return sampled_start
    vectorized_starting_point = jax.vmap(choose_starting_point, in_axes=(1,1, 0), out_axes=(0))
    starting_points = vectorized_starting_point(timestep_for_imagination, predictions_for_imagination, starting_key)

    img_return, grads = nnx.value_and_grad(imagination_loss, argnums=(0, 1), has_aux=True)(
      optimizers.actor_optimizer.model,
      optimizers.critic_optimizer.model,
      world_model_optimizers.sequence_optimizer.model,
      world_model_optimizers.dynamics_optimizer.model,
      world_model_optimizers.predictor_optimizer.model,
      world_model_optimizers.legal_actions_optimizer.model,
      world_model_optimizers.encoder_optimizer.model,
      world_model_optimizers.p1_decoder_optimizer.model,
      world_model_optimizers.p2_decoder_optimizer.model, 
      optimizers.target_optimizer.model,
      trajectory_key, 
      starting_points,
      return_range,
      self.config.beta_imagination)
    
    img_loss, new_range = img_return
    optimizers.actor_optimizer.update(grads[0])
    optimizers.critic_optimizer.update(grads[1])

    ac_timestep = self.dreamer_timestep_to_timestep(dreamer_timestep, dreamer_prediction_step)             
    r_return, grads = nnx.value_and_grad(real_loss, argnums=(0, 1), has_aux=True)(
      optimizers.actor_optimizer.model,
      optimizers.critic_optimizer.model,
      optimizers.target_optimizer.model,
      ac_timestep,
      new_range,
      self.config.beta_real
    )
    r_loss, new_range = r_return
    optimizers.actor_optimizer.update(grads[0])
    optimizers.critic_optimizer.update(grads[1])

    critic_graphdef, state = nnx.split(optimizers.critic_optimizer.model)
    _, state_target = nnx.split(optimizers.target_optimizer.model)

    #This grad coupled with vanilla SGD optimizer 
    # is equivalent to the EMA formula (1 - alpha) * state_target + alpha * state
    target_grad = jax.tree.map(lambda a, b: a - b, state_target, state)
    optimizers.target_optimizer.update(target_grad)

    return img_loss, r_loss, new_range
  
  @partial(nnx.jit, static_argnums=(0))
  def _jit_step_with_model(self, optimizers:ActorCriticOptimizers, world_model_optimizers:DreamerMAOptimizers,
                trajectory_key, dreamer_timestep: TimeStep, 
                dreamer_prediction_step:PredictionStepWithLegal, return_range:chex.Array):
    #
    img_loss, r_loss, new_range = self.update_parameters_and_model(
      optimizers, world_model_optimizers, trajectory_key, dreamer_timestep, dreamer_prediction_step, return_range
    )
    return img_loss, r_loss, new_range

  
  def step(self, dreamer_timestep: TimeStep, dreamer_prediction_step:PredictionStepWithLegal):
    trajectory_key = self.get_next_rng_key()
    #img_loss, r_loss, self.return_range =  self._jit_step_with_model(self.optimizers, self.world_model.optimizers, trajectory_key, dreamer_timestep, dreamer_prediction_step, self.return_range)
    img_loss, r_loss, self.return_range = self.cached_step(trajectory_key, dreamer_timestep, dreamer_prediction_step, self.return_range)
    self.learner_steps += 1
    return img_loss, r_loss


  def __getstate__(self):
    return {"config": self.config,
            "world_model": self.world_model,
            "optimizers": nnx.state(self.optimizers),
            "steps": self.learner_steps,
            "trajectory_key": self.rng_key,
            "return_range" : self.return_range}
  
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

    self.return_range = state["return_range"]
    # #Necessary to correctly continue to train on the updated state of the optimizers
    # self.cached_step = nnx.cached_partial(self._jit_step_with_model, self.optimizers, self.prev_network, self._prev_network)
    # #Explicitly reinitialize the world model optimizers
    # # to make sure that the deserialization does not make them point to
    # # two different objects.
    # wm_optimizers = DreamerMAOptimizers(sequence_optimizer= self.optimizers.sequence_optimizer,
    #       encoder_optimizer = self.optimizers.encoder_optimizer,
    #       p1_decoder_optimizer = self.optimizers.p1_decoder_optimizer,
    #       p2_decoder_optimizer = self.optimizers.p2_decoder_optimizer,
    #       dynamics_optimizer = self.optimizers.dynamics_optimizer,
    #       predictor_optimizer = self.optimizers.predictor_optimizer,
    #       legal_actions_optimizer = self.optimizers.legal_actions_optimizer)

    # self.world_model.optimizers = wm_optimizers
    #Do not forget to also reset the cached step of the world model
    #self.world_model.cached_step = nnx.cached_partial(self.world_model.world_model_train, self.world_model.optimizers) 
  