import chex
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from functools import partial



from train_utils import *
from distributions import *
from ma_rssm import *
from replay_buffer import ReplayBuffer
from dreamer_actor_critic import DreamerActorCritic
from rnad_dreamer import RNaDDreamer

LATEST_STEP_FILENAME = "latest.txt"



class DreamerMA():
  def __init__(self, config: DreamerMAConfig, buffer_config: BufferConfig,
               ac_config:RNaDConfig| ActorCriticConfig, opt_config: OptimizerConfig
               , game: JaxGame, seed:int):
    """The complete model handling Dreamer world model and actor/critic training

    Args:
        config (DreamerMAConfig): Configuration for the world model
        ac_config (RNaDConfig | ActorCriticConfig): Configuration for the actor-critic algorithm. Either RNaD, or standard Dreamer Reinforce with TD(lambda) estimate
        opt_config (OptimizerConfig): Configuration for the optimizer
        buffer (ReplayBuffer): The replay buffer sampling from the environment
        seed (int): RNG seed for the whole algorithm. Shared for world-model and actor-critic
    """
    self.wm_config = config
    self.ac_config = ac_config
    self.opt_config = opt_config
    self.buffer_config = buffer_config
    self.init_seed = seed
    self.game = game
    self.init()
    
    
  def init(self):
    self.jax_rngs = jax.random.key(self.init_seed)

    self.buffer= ReplayBuffer(self.game, self.buffer_config, self.wm_config, self.init_seed, self.ac_config.state_sample_threshold)

    rngs = nnx.Rngs(jax.random.key(self.init_seed))
    self.optimizer= create_dreamer_optimizer(self.game, self.wm_config, self.ac_config, self.opt_config, rngs)
    ma_rssm = self.optimizer.model
    self.learner_steps = 0
    
    #Assuming that this contains a terminal state as well
    self.trajectory_max = self.game.max_trajectory_length()
    self.non_chance_trajectory_max = self.game.max_trajectory_lenght_no_chance()


    self.action_dimension = self.game.num_distinct_actions()
    self.infoset_size = ma_rssm.infoset_size
    self.latent_infoset_size = ma_rssm.latent_infoset_size
    self.use_real_infoset = ma_rssm.use_real_infoset
    assert self.game.num_players() > 1, f"This implementation of Dreamer assumes a game with at least 2 players not {self.game.num_players()}"
    self.recurrent_state_size = ma_rssm.rec_state_size
    #Vanilla SGD coupled with the gradients
    # we compute manually in actor critic will handle
    # the EMA updates for us.
    target_tx = optax.sgd(self.ac_config.target_network_update)
    if ma_rssm.use_rnad:
      ctor = RNaDDreamer
    else:
      ctor = DreamerActorCritic
    target_optimizer = nnx.Optimizer(model=CriticNetwork(self.infoset_size * self.game.num_players(),
                                                           self.ac_config.bin_range,
                                                           self.ac_config.critic_network_details[0],
                                                           self.ac_config.critic_network_details[1],
                                                           rngs=rngs),
                                                           tx = target_tx)
    self.network_keys = ma_rssm.network_names[:-2]
    self.actor_critic = ctor(self.game, self.ac_config, self.optimizer, target_optimizer)
    #self.wm_cached_train = nnx.cached_partial(self.update_world_model, self.optimizer)
    #Also cache the sampling for the buffer
    self.buffer.cache_sampling(ma_rssm.seq, ma_rssm.enc, ma_rssm.observer, ma_rssm.infoset_network, ma_rssm.actor)
    self.grad_norms = {k: 0 for k in self.network_keys}
    self.metrics = {'dec': 0, 'con': 0, 'leg': 0,  'rew': 0, 'dyn': 0, 'rep': 0,'is_act_dec': 0, 'is_obs_dec': 0, 'is_rec_pred': 0, 'is_deter_pred': 0}
    if self.use_real_infoset:
      assert self.game.information_state_tensor_shape() == self.game.observation_tensor_shape(), "Specification of use_real_infoset is only sound when the environment provides infoset in place of observation!"
      print(f"Using original game infosets of shape {self.infoset_size}")
    else:
      print(f"Using latent infosets of shape {self.infoset_size}")
    
  
  def generate_key(self):
    self.jax_rngs, key = jax.random.split(self.jax_rngs)
    return key

  def generate_keys(self, num_keys):
    split_key = self.generate_key()
    keys = jax.random.split(split_key, num_keys)
    return keys
    
  
  @partial(nnx.jit, static_argnums=(0))
  def update_world_model(self, optimizer: nnx.Optimizer, timestep: TimeStep, rng_key):
    """Compound loss for the entire world model."""
    sample_keys = jax.random.split(rng_key, self.non_chance_trajectory_max * self.wm_config.batch_size)
    sample_keys = sample_keys.reshape((self.non_chance_trajectory_max, self.wm_config.batch_size))
    
    def world_model_loss(ma_rssm: MARSSM):
      l_pred, l_dyn, l_rep = 0, 0, 0
      #[Trajectory, Batch, ...]
      @nnx.scan(in_axes=(nnx.Carry, 0, None), out_axes=(nnx.Carry, 0))
      def _predict_over_timestep(carry, xs, model: MARSSM):
        
        recurrent_state, prev_latent_infoset, timestep = carry
        action, obs, prev_action, cur_key = xs
        #Previous action is used for the infoset 
        # at the current timestep. At first step, 
        # there was no previous action so we zero it out
        prev_action = jnp.where(timestep == 0, 0, prev_action)
        stochastic_state = model.get_encoder_no_jit(recurrent_state, obs)
        stochastic_state = add_uniform_mix(stochastic_state, self.wm_config.uniform_mix)
        deterministic_state = sample_categorical(stochastic_state, cur_key)
        prior_stochastic_state = model.get_dynamics_no_jit(recurrent_state)
        prior_stochastic_state = add_uniform_mix(prior_stochastic_state, self.wm_config.uniform_mix)
        decoded_obs = model.get_decoder_no_jit(recurrent_state, deterministic_state, use_symexp=False)
        reward, done = model.rew(recurrent_state, deterministic_state), model.term(recurrent_state, deterministic_state)
        legal = model.leg(recurrent_state, deterministic_state)
        #Dont use symexp here during training. Otherwise we would be training
        # the symexp outputs to match the symlog inputs.
        new_recurrent = model.get_next_recurrent_no_jit(recurrent_state, deterministic_state, action)
        new_latent_infosets = model.get_next_infoset_all_no_jit(prev_latent_infoset, obs, prev_action)
        infoset_decoded_obs, infoset_decoded_actions = model.get_infoset_decoder_all_no_jit(new_latent_infosets)
        infoset_predicted_recurrent, infoset_predicted_deter = model.infoset_predictor(new_latent_infosets)
        preds = PredictionStepWithLegal(
                                recurrent_state = recurrent_state,
                                repr_state = stochastic_state,
                                deter_state = deterministic_state,
                                decoded_obs = decoded_obs,
                                reward_dist_logit = reward,
                                done_logit = done,
                                legal_logit = legal,
                                dynamics_state = prior_stochastic_state,
                                joint_latent_infoset = new_latent_infosets,
                                infoset_decoded_actions = infoset_decoded_actions,
                                infoset_decoded_obs = infoset_decoded_obs,
                                infoset_predicted_recurrent = infoset_predicted_recurrent,
                                infoset_predicted_deter = infoset_predicted_deter) 
        
        return (new_recurrent, new_latent_infosets, timestep + 1), preds
      
      
      previous_actions = jnp.roll(timestep.action, 1, axis=0)
      xs = (timestep.action, timestep.obs, previous_actions, sample_keys)
      init_recurrent = ma_rssm.get_init_recurrent(self.wm_config.batch_size)
      #Since the scan treats this as a previous infoset to the current
      # infoset, we initialize it to all zeros
      init_latent_infosets = jnp.zeros((self.wm_config.batch_size, self.game.num_players(), self.latent_infoset_size))
      #print(f"Init recur shape {init_recurrent.shape}")
      vectorized_predict = nnx.vmap(_predict_over_timestep, in_axes=((0, 0, None), 1,  None), out_axes=(0, 1))
      _, predictions = vectorized_predict((init_recurrent, init_latent_infosets, 0), xs, ma_rssm) 

      #[Trajectory, Batch, num_players, obs_size]
      reconstruction_loss = -get_normal_log_prob(predictions.decoded_obs, timestep.obs, use_symlog=True)
      dec = get_loss_mean_with_mask(reconstruction_loss, timestep.valid[..., None, None])
      l_pred += dec
      #[Trajectory, Batch, 1]
      #continuation_loss = -get_normal_log_prob(predictions.done_logit, timestep.terminal.astype(jnp.int16))
      continuation_loss = optax.sigmoid_binary_cross_entropy(predictions.done_logit, timestep.terminal[..., None])
      con = get_loss_mean_with_mask(continuation_loss, timestep.valid[..., None])
      l_pred += con
      #[Trajectory, Batch, players, action_dim]
      legal_loss = optax.sigmoid_binary_cross_entropy(predictions.legal_logit, timestep.legal)
      #Legal actions should not be trained in terminal states, as there are no legal actions there
      leg = get_loss_mean_with_mask(legal_loss, timestep.valid[..., None, None] & ~timestep.terminal[..., None, None])
      l_pred += leg
      #[Trajectory, Batch, 2* bin_range + 1]
      bins = jnp.arange((2 * self.wm_config.bin_range) + 1) - self.wm_config.bin_range
      reward_loss = -get_bin_log_prob(predictions.reward_dist_logit, bins, timestep.reward, use_symlog=True)
      #[Trajectory, Batch, 1]
      #reward_loss = -get_normal_log_prob(timestep.reward, timestep.reward)
      rew = get_loss_mean_with_mask(reward_loss, timestep.valid[..., None])
      l_pred += rew

      #Using free bits to clip dynamics and representation losses
      # thus disabling their gradient when they are below free_bits_clip_threshold
      #[Trajectory, Batch, encoded_categories, encoded_classes]
      posterior = nnx.softmax(predictions.repr_state, axis=-1)
      prior = nnx.softmax(predictions.dynamics_state, axis=-1)
      #[Trajectory, Batch]
      dynamics_loss = kl_divergence(jax.lax.stop_gradient(posterior), prior)
      l_dyn += jnp.maximum(self.wm_config.free_bits_clip_threshold, get_loss_mean_with_mask(dynamics_loss, timestep.valid))
      #[Trajectory, Batch]
      repr_loss = kl_divergence(posterior, jax.lax.stop_gradient(prior))
      l_rep += jnp.maximum(self.wm_config.free_bits_clip_threshold, get_loss_mean_with_mask(repr_loss, timestep.valid))
      
      mults = [*(self.wm_config.beta_prediction, ) * 4, self.wm_config.beta_dynamics, self.wm_config.beta_representation, *(self.wm_config.beta_infoset, ) * 4]
      l_infoset = 0
      #Update the latent infosets
      #The action loss predicts the previous action. Which also means we do not
      # compute it for the first step
      previous_valid = jnp.roll(timestep.valid, 1, axis=0)
      previous_non_terminal = ~jnp.roll(timestep.terminal, 1, axis=0)
      is_first = jnp.arange(timestep.action.shape[0]) == 0
      action_loss_mask = (~is_first[..., None]) * previous_valid * previous_non_terminal
      #These are one-hot encoded. We want to maximize the probability
      # of seeing the previous action, hence making sure the infoset retains information about it
      infoset_prev_action_loss = -get_categorical_log_prob(predictions.infoset_decoded_actions, previous_actions)
      is_act = get_loss_mean_with_mask(infoset_prev_action_loss, action_loss_mask[..., None, None])
      #is_act = 0
      l_infoset += is_act
      #Current observation loss, similar intuition as with the previous action
      #Reduces to MSE
      is_obs_loss = -get_normal_log_prob(predictions.infoset_decoded_obs, timestep.obs, use_symlog=True)
      is_obs = get_loss_mean_with_mask(is_obs_loss, timestep.valid[..., None, None])
      l_infoset += is_obs
      # The current recurrent state prediction loss. This together
      # with the current deter state prediction loss serves to force
      # perfect recall by making sure that the union of infosets
      # is enough to get the perfect information state.
      is_rec_loss = -get_normal_log_prob(predictions.infoset_predicted_recurrent, jax.lax.stop_gradient(predictions.recurrent_state), use_symlog=True)
      is_rec = get_loss_mean_with_mask(is_rec_loss, timestep.valid[..., None])
      #is_rec = 0
      l_infoset += is_rec
      is_deter_loss = -get_categorical_log_prob(predictions.infoset_predicted_deter, jax.lax.stop_gradient(predictions.deter_state))
      is_deter = get_loss_mean_with_mask(is_deter_loss, timestep.valid[..., None, None])
      #is_deter = 0
      l_infoset += is_deter

      
      losses = [dec, con, leg, rew, l_dyn, l_rep, is_act, is_obs, is_rec, is_deter]

      wm_keys = self.metrics.keys()
      metrics = {k: v * m for k, v, m in zip(wm_keys, losses, mults)}
      
      compound_loss = sum(l * m for l, m in zip(losses, mults))

      return compound_loss, (predictions, metrics)
  
    grad_norms = self.grad_norms.copy()
    func_data, grad = nnx.value_and_grad(world_model_loss, has_aux=True, argnums=(0))(
                    optimizer.model)
    if self.wm_config.report_gradnorms:
      for k in self.network_keys:
        grad_norms[k] = optax.tree.norm(grad[k], ord=2)
    
    loss, (pred_step, metrics) = func_data
    optimizer.update(grad)
    
    return loss, pred_step, metrics, grad_norms
    

  def train_step(self):
    buffer_key = self.generate_key()
    timestep = self.buffer.mixed_sample(buffer_key)
    wm_key = self.generate_key()
    wm_loss, pred_step, self.wm_metrics, self.grad_norms = self.update_world_model(self.optimizer, timestep, wm_key)
    ac_key = self.generate_key()
    self.actor_critic.step(timestep, pred_step, ac_key)
    self.learner_steps += 1

  def train_model(self, model_save_dir:str, num_steps:int, print_each: int = -1, 
                  save_each: int = -1,
                  save_first: bool = False):
    
    
    print(f"Training model that is saved at {model_save_dir}")
    def save_latest():
      #Save which model file is the latest
      if latest_step > 0:
        latest_step_file = model_save_dir + LATEST_STEP_FILENAME
        with open(latest_step_file, 'w') as f:
          f.write(f"step_{latest_step}.pkl")
    latest_step = -1
    if save_first:
      model_file = model_save_dir + f"step_{self.learner_steps}.pkl"
      latest_step = self.learner_steps
      save_model(self, model_file)
      save_latest()
    
    #Start the training by sampling into the buffer,
    # to ensure that there are distinct data for at least one step
    init_batch_key = self.generate_key()
    self.buffer.add_batch(self.wm_config.batch_size, init_batch_key)
    for i in range(num_steps):
      self.train_step()
      if print_each > 0 and self.learner_steps % print_each == 0:
        print(f"Step {self.learner_steps}, World model losses: {self.wm_metrics}.")
        print(f"Actor critic metrics: {self.actor_critic.metrics}.")
        if self.wm_config.report_gradnorms:
          print(f"World model gradnorms {self.grad_norms}")
        if self.ac_config.report_gradnorms:
          print(f"Actor critic gradnorms {self.actor_critic.grad_norms}")
      if save_each > 0 and self.learner_steps % save_each == 0:
        latest_step = self.learner_steps
        model_file = model_save_dir + f"step_{self.learner_steps}.pkl"
        save_model(self, model_file)
        save_latest()
    self.buffer.store_returns(model_save_dir)
    
   
  def __getstate__(self):
    
    state = {}
    general_state = {
      'wm_config': self.wm_config,
      'ac_config': self.ac_config,
      'opt_config': self.opt_config,
      'buffer_config': self.buffer_config,
      'init_seed': self.init_seed,
      'game': self.game,
      'jax_rngs': self.jax_rngs,
      'optimizer': nnx.state(self.optimizer),
      'steps': self.learner_steps
    }
    state['gen'] = general_state
    actor_critic_state = self.actor_critic.getstate()
    state['ac'] = actor_critic_state
    buffer_state = self.buffer.getstate()
    state['buffer'] = buffer_state
    return state
    
  
  def __setstate__(self, state):
    gen_state = state['gen']
    self.wm_config = gen_state['wm_config']
    self.ac_config = gen_state['ac_config']
    self.opt_config = gen_state['opt_config']
    self.buffer_config = gen_state['buffer_config']
    self.init_seed = gen_state['init_seed']
    self.game = gen_state['game']
    self.init()
    
    
    self.jax_rngs = gen_state["jax_rngs"]
    nnx.update(self.optimizer, gen_state["optimizer"])
    self.learner_steps = gen_state["steps"]
    self.actor_critic.setstate(state['ac'])
    self.buffer.setstate(state['buffer'])
    

    
    