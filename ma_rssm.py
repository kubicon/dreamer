import jax

from functools import partial


from networks import *
from optimizer import make_opt
from distributions import sample_categorical
from train_utils import *
from games.jax_game import JaxGame

f32 = jnp.float32
u8 = jnp.uint8

  

class MARSSM(nnx.Module):
  
  def __init__(self, game: JaxGame, wm_config: DreamerMAConfig, ac_config: RNaDConfig | ActorCriticConfig, rngs: nnx.Rngs):
    """A model encapsulating the entire structure of the Dreamer algorithm for
    2p0s games.

    Args:
        game (JaxGame): The environment to train on
        wm_config (DreamerMAConfig): A full configuration of the Dreamer world model
        ac_config (RNaDConfig | ActorCriticConfig): _description_
    """
    #Outside of IIGs, the concatenation of the latent states (player_id, recurrent_state, deter_state)
    # for each player is sufficient for our iset. Later, when scaling up, we will have a model 
    # for each player and it will also work this way even for IIGs. 
    # But for now, for IIGs we just use the actual in game isets.
    self.use_real_iset = wm_config.use_original_iset
    self.num_actions = game.num_distinct_actions()
    self.num_players = game.num_players()
    rec_state_size = wm_config.sequential_network_details[0]
    #If recurrent state size is unset, set it to the minimal
    # size possible to capture all the context. Eg.
    # the size of the infoset tensor
    if rec_state_size < 1:
      rec_state_size = game.information_state_tensor_shape()
    self.rec_state_size = rec_state_size
    enc_tokens = wm_config.encoder_network_details[0]
    deter_size = wm_config.encoded_categories * wm_config.encoded_classes
    self.encoded_classes = wm_config.encoded_classes
    self.encoded_categories = wm_config.encoded_categories
    if self.use_real_iset:
      self.infoset_size = game.information_state_tensor_shape()
      print(f"Using original game isets of shape {self.infoset_size}")
    else:
      self.infoset_size = (rec_state_size + deter_size)
      print(f"Using the model states of shape {self.infoset_size}")
    #self.wm_config = wm_config
    #self.ac_config = ac_config
    self.sampling_epsilon = ac_config.sampling_epsilon
    self.legal_threshold = ac_config.legal_threshold
    self.terminal_threshold = ac_config.terminal_threshold
    self.state_sample_threshold = ac_config.state_sample_threshold
    self.wm_bin_range = wm_config.bin_range
    self.use_rnad = isinstance(ac_config, RNaDConfig)
    #Always imagine at most 1 less than max trajectory
    # length, since the last step is terminal and we do not train on those
    self.ac_trajectory_len = game.max_trajectory_lenght_no_chance() - 1
    self.seq = SequenceModel(wm_config.encoded_classes,
                              wm_config.encoded_categories,
                              self.num_actions,
                              wm_config.sequential_network_details[1],
                              wm_config.sequential_network_details[2],
                              rec_state_size,
                              rngs)
    self.rew = RewardPredictor(self.num_players,
                               wm_config.bin_range, 
                               wm_config.encoded_classes, 
                               wm_config.encoded_categories,
                               rec_state_size,
                               wm_config.reward_predictor_network_details[0],
                               wm_config.reward_predictor_network_details[1],
                               rngs=rngs)
    self.term = DonePredictor(self.num_players,
                              wm_config.encoded_classes,
                              wm_config.encoded_categories,
                              rec_state_size,
                              wm_config.done_predictor_network_details[0],
                              wm_config.done_predictor_network_details[1],
                              rngs)

    self.leg = LegalActionsNetwork(self.num_players,
                                   self.num_actions,
                                   wm_config.encoded_classes,
                                   wm_config.encoded_categories,
                                   rec_state_size,
                                   wm_config.legal_actions_network_details[0],
                                   wm_config.legal_actions_network_details[1],
                                   rngs)
    self.dyn = DynamicsPredictor(rec_state_size,
                                 wm_config.encoded_classes,
                                 wm_config.encoded_categories,
                                 wm_config.dynamics_network_details[0],
                                 wm_config.dynamics_network_details[1],
                                 rngs=rngs)
    self.enc = Encoder(game.information_state_tensor_shape(),
                                enc_tokens,
                                wm_config.encoder_network_details[1],
                                wm_config.encoder_network_details[2],
                                rngs
                                )
    
    self.dec = Decoder(rec_state_size, 
                          game.information_state_tensor_shape(),
                          wm_config.encoded_classes,
                          wm_config.encoded_categories,
                          wm_config.decoder_network_details[0],
                          wm_config.decoder_network_details[1],
                          rngs)
    self.observer = ObservedPredictor(rec_state_size,
                                      enc_tokens,
                                      wm_config.encoded_classes,
                                      wm_config.encoded_categories,
                                      wm_config.observer_network_details[0],
                                      wm_config.observer_network_details[1],
                                      rngs)
    
    self.network_names = ['dyn', 'seq' 'p1_enc','leg', 'observer', 'dec', 'rew', 'term']

    if self.use_rnad:
      self.actor_critic = RNaDNetwork(self.infoset_size,
                                      self.num_actions,
                                      ac_config.bin_range,
                                      ac_config.rnad_network_details[0],
                                      ac_config.rnad_network_details[1],
                                      rngs)
      self.network_names.append('actor_critic')
    else:
      self.actor = ActorNetwork(self.infoset_size,
                                self.num_actions,
                                ac_config.actor_network_details[0], 
                                ac_config.actor_network_details[1],
                                rngs)
      self.critic = CriticNetwork(self.infoset_size,
                                  ac_config.bin_range,
                                  ac_config.critic_network_details[0],
                                  ac_config.critic_network_details[1],
                                  rngs)
      self.network_names.append('actor')
      self.network_names.append('critic')
      
  def default_ac_timestep(self):
    obs = jnp.zeros((1, self.infoset_size), dtype=f32)
    
    legal = jnp.ones((1, self.num_actions), dtype=u8)
    action = jnp.ones((1, self.num_actions), dtype=f32)
    policy = jnp.ones((1,self.num_actions), dtype=u8)
    valid =jnp.array(0, dtype=f32)
    reward = jnp.array(0, dtype=f32)
    
    ts = ActorCriticTimeStep(
      valid = valid,
      obs = obs,
      legal = legal,
      action = action, 
      policy = policy,
      reward = reward
    )
    return ts

  
  @staticmethod
  def call_net(net: nnx.Module, *args):
    return net(*args)
  
  @staticmethod
  def vmap_over_net(net: nnx.Module, in_axes: list, out_axes: list):
    """A special vmap over a given net, where in_axes and out axes
    are required to be lists, where each element is some sequence
    of axes corresponding to one vmap. Eg. the lenght of the lists
    corresponds to the number of vmaps done. The lists
    are ordered from innermost to the outermost vmap.
    The in axes and out axes are NOT expected to contain
    the leading None for the network itself, that is added by this function.
    The network itself is then bound to the vmapped function, so it should not be put into
    calls of the resulting function."""
    assert len(in_axes) == len(out_axes), f"In axes and out axes need to contain specification for the same amount of vmaps! Got {len(in_axes)} for in axes and {len(out_axes)} for out axes."
    graphdef, state = nnx.split(net)
    def pure_call(state, args_tuple):
      local_net = nnx.merge(graphdef, state)
      return local_net(*args_tuple)
    
    f = pure_call
    for in_ax, out_ax in zip(in_axes, out_axes):
      if isinstance(in_ax, int):
        in_ax = (in_ax, )
      f = jax.vmap(f, in_axes=(None, in_ax), out_axes=out_ax)
    def final_bind(*args):
      return f(state, args)
    return final_bind
  
  def policy_net(self) ->nnx.Module:
    if self.use_rnad:
      return self.actor_critic
    return self.actor
  

  @nnx.jit
  def get_predictor(self, joint_recurrent_state: chex.Array, joint_deterministic_state:chex.Array):
    """Calls the predictor and legal actions networks and 
    passes the reward, done logits and legal action logits through
    appropriate transformations to return the actual values"""
    return self.get_predictor_no_jit(joint_recurrent_state, joint_deterministic_state,
                                    )
  
  def get_predictor_no_jit(self, joint_recurrent_state: chex.Array, joint_deterministic_state:chex.Array):
    """Calls the predictor and legal actions networks and 
    passes the reward, done logits and legal action logits through
    appropriate transformations to return the actual values"""
    #[2* bin_range + 1], [1]
    reward_bin_logits, done_logit = MARSSM.call_net(self.rew, joint_recurrent_state, joint_deterministic_state), MARSSM.call_net(self.term, joint_recurrent_state, joint_deterministic_state)
    legal_logit = MARSSM.call_net(self.leg, joint_recurrent_state, joint_deterministic_state)
    reward = get_value_from_bins(reward_bin_logits, self.wm_bin_range)
    done_prob = nnx.sigmoid(done_logit)
    terminal = done_prob >= self.terminal_threshold
    legal_prob = nnx.sigmoid(legal_logit)
    legal_actions = (legal_prob >= self.legal_threshold).astype(u8)
    return reward[0], terminal[0], legal_actions
  
  @partial(nnx.jit, static_argnums=(3))
  def get_decoder(self, recurrent_state: chex.Array, deterministic_state: chex.Array, use_symexp=True):
    """Calls the decoder network and 
    applies the appropriate transformation to its output.
    Outputs either predicted real observation in single agent setting, or 
    predicted iset for a single player in a multi agent setting. """
    decoder_output = MARSSM.call_net(self.dec, recurrent_state, deterministic_state)
    if use_symexp:
      decoder_output = symexp(decoder_output)
    return decoder_output
  
  @partial(nnx.jit, static_argnums=(3))
  def get_decoder_all(self, joint_recurrent_state: chex.Array, joint_deterministic_state:chex.Array, use_symexp=True):
    return self.get_decoder_all_no_jit(joint_recurrent_state, joint_deterministic_state, use_symexp)

  
  def get_decoder_all_no_jit(self, joint_recurrent_state: chex.Array, joint_deterministic_state:chex.Array, use_symexp=True):
    vectorized_decoder = nnx.vmap(MARSSM.call_net, in_axes=(None, 0, 0), out_axes=0)
    decoder_output = vectorized_decoder(self.dec, joint_recurrent_state, joint_deterministic_state)
    if use_symexp:
      decoder_output = symexp(decoder_output)
    return decoder_output
  
  @nnx.jit
  def get_dynamics(self, recurrent_state:chex.Array):
    return MARSSM.call_net(self.dyn, recurrent_state)
  
  @nnx.jit
  def get_dyn_all(self, joint_recurrent_state: chex.Array):
    return self.get_dyn_all_no_jit(joint_recurrent_state)
  
  def get_dyn_all_no_jit(self, joint_recurrent_state: chex.Array):
    vectorized_dynamics = nnx.vmap(MARSSM.call_net, in_axes=(None, 0), out_axes=0)
    return vectorized_dynamics(self.dyn, joint_recurrent_state)
  
  @nnx.jit
  def get_encoder(self, recurrent_state:chex.Array, obs: chex.Array):
    tokens = MARSSM.call_net(self.enc, obs)
    return MARSSM.call_net(self.observer, recurrent_state, tokens)
  
  def get_encoder_no_jit(self,recurrent_state:chex.Array, obs: chex.Array):
    tokens = MARSSM.call_net(self.enc, obs)
    return MARSSM.call_net(self.observer, recurrent_state, tokens)
  
  @nnx.jit
  def get_enc_all(self, joint_recurrent_state: chex.Array, joint_obs: chex.Array):
    return self.get_dyn_all_no_jit(joint_recurrent_state, joint_obs)
  
  def get_enc_all_no_jit(self, joint_recurrent_state: chex.Array, joint_obs: chex.Array):
    vectorized_enc = nnx.vmap(MARSSM.call_net, in_axes=(None, 0), out_axes=0)
    vectorized_observer = nnx.vmap(MARSSM.call_net, in_axes=(None, 0, 0), out_axes=0)
    tokens = vectorized_enc(self.enc, joint_obs)
    return vectorized_observer(self.observer, joint_recurrent_state, tokens)

  
  @nnx.jit
  def get_next_recurrent(self, recurrent_state:chex.Array, deterministic_state:chex.Array, action:chex.Array):
    return MARSSM.call_net(self.seq, recurrent_state, deterministic_state, action)
  
  def get_next_recurrent_no_jit(self, recurrent_state:chex.Array, deterministic_state:chex.Array, action:chex.Array):
    return MARSSM.call_net(self.seq, recurrent_state, deterministic_state, action)
  
  @nnx.jit
  def get_next_recurrent_all(self, joint_recurrent_state:chex.Array, joint_deterministic_state:chex.Array, joint_action:chex.Array):
    return self.get_next_recurrent_all_no_jit(joint_recurrent_state, joint_deterministic_state, joint_action)
  
  def get_next_recurrent_all_no_jit(self, joint_recurrent_state:chex.Array, joint_deterministic_state:chex.Array, joint_action:chex.Array):
    vectorized_seq = nnx.vmap(MARSSM.call_net, in_axes=(None, 0, 0, 0), out_axes=0)
    return vectorized_seq(self.seq, joint_recurrent_state, joint_deterministic_state, joint_action)
  
  
  @partial(nnx.jit, static_argnums=(1))
  def get_init_recurrent(self, n_starts:int = 0):
    dummy_rec = jnp.zeros(self.rec_state_size)
    dummy_deter = jnp.zeros((self.encoded_classes, self.encoded_categories))
    dummy_action = jnp.zeros((self.num_actions))
    #Both players will start from the zero context
    # we can just tile this instead of calling the network twice
    init_rec =  MARSSM.call_net(self.seq, dummy_rec, dummy_deter, dummy_action)
    init_rec = jnp.tile(init_rec[None, ...], (self.num_players, 1))
    #Just handle 0, or negative value as a special case for only one
    # start, without the leading batch dimension
    if n_starts > 0:
      init_rec = jnp.tile(init_rec[None, ...], (n_starts, 1, 1))
    return init_rec

  
  @nnx.jit
  def get_policy(self, obs, legal) ->chex.Array:
    if self.use_rnad:
      return MARSSM.call_net(self.actor_critic, obs, legal)[0]
    return MARSSM.call_net(self.actor, obs, legal)
  
  
  
  @nnx.jit
  def get_policy_both(self, joint_obs, joint_legal) ->chex.Array:
    return self.get_policy_both_no_jit(joint_obs, joint_legal)
  
  def get_policy_both_no_jit(self, joint_obs, joint_legal) ->chex.Array:
    
    if self.use_rnad:
      net = self.actor_critic
    else:
      net = self.actor
    vectorized_actor = nnx.vmap(MARSSM.call_net, in_axes=(None, 0, 0), out_axes=0)
    return vectorized_actor(net, joint_obs, joint_legal)[0]
  
  @nnx.jit
  def imagine_trajectories(self, key, starting_points: PredictionStepWithLegal) ->ActorCriticTimeStep:
    batch_size = starting_points.done_logit.shape[0]
    keys = jax.random.split(key, batch_size)
    batch_sample_trajectory = nnx.vmap(self.imagine_trajectory, in_axes=(0, 0, None), out_axes=1) 
    return batch_sample_trajectory(keys, starting_points, self)


  @nnx.jit
  def get_iset(self, joint_recurrent_state: chex.Array, joint_deter_state:chex.Array):
    return self.get_iset_no_jit(joint_recurrent_state, joint_deter_state)
  
  def get_iset_no_jit(self, joint_recurrent_state: chex.Array, joint_deter_state:chex.Array):
    if self.use_real_iset:
      return self.get_decoder_all_no_jit(joint_recurrent_state, joint_deter_state)
    flat_deter = joint_deter_state.reshape((*joint_deter_state.shape[:-2], -1))
    model_state = jnp.concatenate([joint_recurrent_state, flat_deter], axis=-1)
    return model_state


  
  def imagine_trajectory(self, key, starting_point: PredictionStepWithLegal, ma_rssm) ->ActorCriticTimeStep:
    #init_sample_key, trajectory_key, = jax.random.split(key)
    trajectory_key = jax.random.split(key, self.ac_trajectory_len)
    ac_default = self.default_ac_timestep()
  
    
    
    @chex.dataclass(frozen=True)
    class SampleTrajectoryCarry:
      joint_recurrent_state:chex.Array
      joint_deter_state: chex.Array
      legal_actions: chex.Array
      terminal: bool
      
    init_carry = SampleTrajectoryCarry(
      joint_recurrent_state = starting_point.joint_recurrent_state,
      joint_deter_state = starting_point.joint_deter_state, #TODO: Take the one that the world model sampled, or sample anew?
      legal_actions = (nnx.sigmoid(starting_point.legal_logit) >= self.legal_threshold).astype(u8), 
      terminal = (nnx.sigmoid(starting_point.done_logit) >= self.terminal_threshold)[0]
    )
    
    
    def choice_wrapper(key, p):
      action = jax.random.choice(key, self.num_actions, p=p)
      action_oh = jax.nn.one_hot(action, self.num_actions)
      return action, action_oh

    vectorized_sample_action = nnx.vmap(choice_wrapper, in_axes=(0, 0), out_axes=0)

    

    @nnx.scan(in_axes = (nnx.Carry, 0, None), out_axes=(nnx.Carry, 0))
    def _imagine_trajectory(carry: SampleTrajectoryCarry, key, ma_rssm: MARSSM) -> tuple[SampleTrajectoryCarry, chex.Array]:
      
      obs = ma_rssm.get_iset_no_jit(carry.joint_recurrent_state, carry.joint_deter_state)

      #get policy 
      pi = ma_rssm.get_policy_both_no_jit(obs, carry.legal_actions)
      #uniform mix to the policy
      normalization = jnp.sum(carry.legal_actions, axis=-1, keepdims=True)
      uniform_pi = carry.legal_actions / (normalization + (normalization == 0))
      pi = self.sampling_epsilon * uniform_pi + (1 - self.sampling_epsilon) * pi
      # For each player samples a single action
      
      action_sample_key, state_sample_key = jax.random.split(key)
      action_sample_keys = jax.random.split(action_sample_key, self.num_players)
      action, action_oh = vectorized_sample_action(action_sample_keys, pi)

      
      
      next_joint_recurrent = ma_rssm.get_next_recurrent_all_no_jit(carry.joint_recurrent_state, carry.joint_deter_state, action_oh)
      next_joint_stoch = ma_rssm.get_dyn_all_no_jit(next_joint_recurrent)
      next_joint_deter = sample_categorical(next_joint_stoch, state_sample_key, sample_threshold=ma_rssm.state_sample_threshold)
      next_reward, next_terminal, next_legal = ma_rssm.get_predictor(next_joint_recurrent, next_joint_deter)
      next_terminal = jnp.logical_or(carry.terminal, next_terminal)
      # The world model can produce all actions to be invalid
      # even when one of the players does not act, he always has one legal
      # NOOP action. So, if one of the players has all actions invalid, then
      # the state is not valid
      valid = jnp.logical_and(jnp.logical_not(carry.terminal), jnp.all(normalization > 0))
      timestep = ActorCriticTimeStep(
        obs = obs,
        legal = carry.legal_actions.astype(u8),
        action = action_oh.astype(u8),
        policy = pi,
        reward = next_reward,
        valid = valid
      )
      new_carry = SampleTrajectoryCarry(
        joint_recurrent_state = next_joint_recurrent,
        joint_deter_state = next_joint_deter,
        legal_actions=jnp.where(next_terminal, ac_default.legal, next_legal),
        terminal = jnp.logical_or(next_terminal, jnp.logical_not(valid)),
      )
         
      timestep = tree_where(timestep.valid, timestep, ac_default)
      return new_carry, timestep
    _, timestep = _imagine_trajectory(init_carry, trajectory_key, ma_rssm)
    #[Trajectory, ...]
    return timestep
      
def create_dreamer_optimizer(game: JaxGame, wm_config: DreamerMAConfig, ac_config: RNaDConfig | ActorCriticConfig,
                              opt_config:OptimizerConfig,  rngs: nnx.Rngs,
                             return_tx = False):
  """Create a single optimizer for the entire MARSSM

  Args:
      game (JaxGame): The environment to train on
      wm_config (DreamerMAConfig): Config of the world model
      ac_config (RNaDConfig | ActorCriticConfig): Config of the actor-critic algorithm. Can be either RNaD or the standard Dreamer combination of Reinforce and TD(lambda)
      seed (int): Seed used to initialize all the networks
  """
  ma_rssm = MARSSM(game, wm_config, ac_config, rngs)
  opt_tx = make_opt(opt_config)
  optimizer = nnx.Optimizer(model=ma_rssm, tx=opt_tx)
  if not return_tx:
    return optimizer
  return optimizer, opt_tx


    
  
  