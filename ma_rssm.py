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
    # for each player is sufficient for our infoset. Later, when scaling up, we will have a model 
    # for each player and it will also work this way even for IIGs. 
    # But for now, for IIGs we just use the actual in game infosets.
    self.use_real_infoset = wm_config.use_original_infoset
    self.num_actions = game.num_distinct_actions()
    self.num_players = game.num_players()
    rec_state_size = wm_config.sequential_network_details[0]
    latent_infoset_size = wm_config.infoset_network_details[0]
    #If recurrent state size is unset, set it to the minimal
    # size possible to capture all the context. Eg.
    # the size of the infoset tensor
    if rec_state_size < 1:
      rec_state_size = self.num_players * game.information_state_tensor_shape()
    #Similarly for the latent infoset
    if latent_infoset_size < 1:
      latent_infoset_size = game.information_state_tensor_shape()
    self.rec_state_size = rec_state_size
    self.latent_infoset_size = latent_infoset_size
    enc_tokens = wm_config.encoder_network_details[0]
    self.encoded_classes = wm_config.encoded_classes
    self.encoded_categories = wm_config.encoded_categories
    if self.use_real_infoset:
      self.infoset_size = game.information_state_tensor_shape()
    else:
      self.infoset_size = latent_infoset_size
    self.observation_size = game.observation_tensor_shape()
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
                              self.num_players,
                              self.num_actions,
                              wm_config.sequential_network_details[1],
                              wm_config.sequential_network_details[2],
                              rec_state_size,
                              rngs)
    self.rew = RewardPredictor(wm_config.bin_range, 
                               wm_config.encoded_classes, 
                               wm_config.encoded_categories,
                               rec_state_size,
                               wm_config.reward_predictor_network_details[0],
                               wm_config.reward_predictor_network_details[1],
                               rngs=rngs)
    self.term = DonePredictor(wm_config.encoded_classes,
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
    self.enc = Encoder(self.num_players,
                                self.observation_size,
                                enc_tokens,
                                wm_config.encoder_network_details[1],
                                wm_config.encoder_network_details[2],
                                rngs
                                )
    
    self.dec = Decoder(self.num_players,
                          rec_state_size, 
                          self.observation_size,
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
    
    self.network_names = ['dyn', 'seq', 'enc','leg', 'observer', 'dec', 'rew', 'term', 'infoset_network', 'infoset_decoder', 'infoset_predictor', 'actor', 'critic']

    self.infoset_network = InfosetModel(self.observation_size,
                                      self.num_actions,
                                      latent_infoset_size,
                                      wm_config.infoset_network_details[1],
                                      wm_config.infoset_network_details[2],
                                      rngs)
    self.infoset_decoder = InfosetDecoder(self.observation_size,
                                        self.num_actions,
                                        self.latent_infoset_size,
                                        wm_config.infoset_decoder_details[0],
                                        wm_config.infoset_decoder_details[1],
                                        rngs)
    self.infoset_predictor = InfosetPredictor(self.num_players,
                                            self.latent_infoset_size,
                                            self.rec_state_size,
                                            wm_config.encoded_classes,
                                            wm_config.encoded_categories,
                                            wm_config.infoset_predictor_details[0],
                                            wm_config.infoset_predictor_details[1],
                                            rngs)

    self.actor = ActorNetwork(self.infoset_size,
                              self.num_actions,
                              ac_config.actor_network_details[0], 
                              ac_config.actor_network_details[1],
                              rngs)
    self.critic = CriticNetwork(self.infoset_size * game.num_players(),
                                ac_config.bin_range,
                                ac_config.critic_network_details[0],
                                ac_config.critic_network_details[1],
                                rngs)
      
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
  

  @nnx.jit
  def get_predictor(self, recurrent_state: chex.Array, deterministic_state:chex.Array):
    """Calls the predictor and legal actions networks and 
    passes the reward, done logits and legal action logits through
    appropriate transformations to return the actual values"""
    return self.get_predictor_no_jit(recurrent_state, deterministic_state)
  
  def get_predictor_no_jit(self, recurrent_state: chex.Array, deterministic_state:chex.Array):
    """Calls the predictor and legal actions networks and 
    passes the reward, done logits and legal action logits through
    appropriate transformations to return the actual values"""
    #[2* bin_range + 1], [1]
    reward_bin_logits, done_logit = MARSSM.call_net(self.rew, recurrent_state, deterministic_state), MARSSM.call_net(self.term, recurrent_state, deterministic_state)
    legal_logit = MARSSM.call_net(self.leg, recurrent_state, deterministic_state)
    reward = get_value_from_bins(reward_bin_logits, self.wm_bin_range)
    done_prob = nnx.sigmoid(done_logit)
    terminal = done_prob >= self.terminal_threshold
    legal_prob = nnx.sigmoid(legal_logit)
    legal_actions = (legal_prob >= self.legal_threshold).astype(u8)
    return reward[0], terminal[0], legal_actions
  
  @partial(nnx.jit, static_argnums=(2))
  def get_decoder(self, recurrent_state: chex.Array, deter_state: chex.Array, use_symexp=True):
    self.get_decoder_no_jit(recurrent_state, deter_state, use_symexp)
  
  def get_decoder_no_jit(self, recurrent_state: chex.Array, deter_state: chex.Array, use_symexp=True):
    """Calls the decoder network and 
    applies the appropriate transformation to its output.
    Outputs either predicted real observation in single agent setting, or 
    predicted infoset for a single player in a multi agent setting. """
    decoder_output = MARSSM.call_net(self.dec, recurrent_state, deter_state)
    if use_symexp:
      decoder_output = symexp(decoder_output)
    return decoder_output
  
  @nnx.jit
  def get_dynamics(self, recurrent_state:chex.Array):
    return MARSSM.call_net(self.dyn, recurrent_state)
  
  def get_dynamics_no_jit(self, recurrent_state:chex.Array):
    return MARSSM.call_net(self.dyn, recurrent_state)
  
  @partial(nnx.jit, static_argnums=3)
  def get_encoder(self, recurrent_state:chex.Array, obs: chex.Array, use_symlog=True):
    return self.get_encoder_no_jit(recurrent_state, obs, use_symlog)
  
  def get_encoder_no_jit(self,recurrent_state:chex.Array, obs: chex.Array, use_symlog=True):
    if use_symlog:
      obs = symlog(obs)
    tokens = MARSSM.call_net(self.enc, obs)
    return MARSSM.call_net(self.observer, recurrent_state, tokens)

  
  @nnx.jit
  def get_next_recurrent(self, recurrent_state:chex.Array, deterministic_state:chex.Array, action:chex.Array):
    return MARSSM.call_net(self.seq, recurrent_state, deterministic_state, action)
  
  def get_next_recurrent_no_jit(self, recurrent_state:chex.Array, deterministic_state:chex.Array, action:chex.Array):
    return MARSSM.call_net(self.seq, recurrent_state, deterministic_state, action)
  
  def get_next_infoset_all_no_jit(self, joint_latent_infoset: chex.Array, joint_cur_obs:chex.Array, joint_action:chex.Array, use_symlog=True):
    """Update latent infoset for both players when they observe new observation cur_obs after playing an action"""
    if use_symlog:
      joint_cur_obs = symlog(joint_cur_obs)
    vectorized_get_infosets = MARSSM.vmap_over_net(self.infoset_network, in_axes=[(0, 0, 0)], out_axes=([0]))
    return vectorized_get_infosets(joint_latent_infoset, joint_cur_obs, joint_action)
  
  @partial(nnx.jit, static_argnums=4)
  def get_next_infoset_all(self, joint_latent_infoset: chex.Array, joint_cur_obs:chex.Array, joint_action:chex.Array, use_symlog=True):
    return self.get_next_infoset_all_no_jit(joint_latent_infoset, joint_cur_obs, joint_action, use_symlog)
  
  def get_infoset_decoder_all_no_jit(self, joint_latent_infoset:chex.Array, use_symexp=False):
    vectorized_infoset_decoder = MARSSM.vmap_over_net(self.infoset_decoder, in_axes=[(0, )], out_axes=[(0, 0)])
    output = vectorized_infoset_decoder(joint_latent_infoset)
    if use_symexp:
      output = symexp(output)
    return output
  
  @partial(nnx.jit, static_argnums=2)
  def get_infoset_decoder_all(self, joint_latent_infoset:chex.Array, use_symexp=False):
    return self.get_infoset_decoder_all_no_jit(joint_latent_infoset, use_symexp)
  
  
  @partial(nnx.jit, static_argnums=1)
  def get_init_recurrent(self, n_starts:int = 0):
    dummy_rec = jnp.zeros(self.rec_state_size)
    dummy_deter = jnp.zeros((self.encoded_classes, self.encoded_categories))
    dummy_action = jnp.zeros((self.num_players, self.num_actions))
    init_rec =  MARSSM.call_net(self.seq, dummy_rec, dummy_deter, dummy_action)
    #Just handle 0, or negative value as a special case for only one
    # start, without the leading batch dimension
    if n_starts > 0:
      init_rec = jnp.tile(init_rec[None, ...], (n_starts, 1))
    return init_rec
  
  @partial(nnx.jit)
  def get_init_infoset(self, init_obs: chex.Array):
    """Get initial infosets for a batch of initial observations."""
    dummy_infoset = jnp.zeros((1, 1, self.latent_infoset_size))
    dummy_actions = jnp.zeros((1, self.num_actions))
    #We vmap only over the observations, not over the dummy infoset and
    # actions, hence the None
    vectorized_init_infosets = MARSSM.vmap_over_net(self.infoset_network, in_axes=[(None, 0, None), (None, 0, None)], out_axes=[0, 0])
    init_infosets = vectorized_init_infosets(dummy_infoset, init_obs, dummy_actions)
    return init_infosets

  
  @partial(nnx.jit, static_argnums=3)
  def get_policy(self, obs, legal, use_symlog=True) ->chex.Array:
    if use_symlog:
      obs = symlog(obs)
    return MARSSM.call_net(self.actor, obs, legal)
  
  
  
  @partial(nnx.jit, static_argnums=3)
  def get_policy_both(self, joint_obs, joint_legal, use_symlog=True) ->chex.Array:
    return self.get_policy_both_no_jit(joint_obs, joint_legal, use_symlog)
  
  def get_policy_both_no_jit(self, joint_obs, joint_legal, use_symlog=True) ->chex.Array:
    if use_symlog:
      joint_legal = symlog(joint_obs)
    vectorized_actor = nnx.vmap(MARSSM.call_net, in_axes=(None, 0, 0), out_axes=0)
    return vectorized_actor(self.actor, joint_obs, joint_legal)[0]
  
  @nnx.jit
  def imagine_trajectories(self, key, starting_points: PredictionStepWithLegal) ->ActorCriticTimeStep:
    batch_size = starting_points.done_logit.shape[0]
    keys = jax.random.split(key, batch_size)
    batch_sample_trajectory = nnx.vmap(self.imagine_trajectory, in_axes=(0, 0, None), out_axes=1) 
    return batch_sample_trajectory(keys, starting_points, self)


  @nnx.jit
  def get_infoset(self, recurrent_state: chex.Array, deter_state:chex.Array, joint_latent_infoset: chex.Array):
    return self.get_infoset_no_jit(recurrent_state, deter_state, joint_latent_infoset)
  
  def get_infoset_no_jit(self, recurrent_state: chex.Array, deter_state:chex.Array, joint_latent_infoset:chex.Array):
    if self.use_real_infoset:
      return self.get_decoder(recurrent_state, deter_state)
    return joint_latent_infoset


  
  def imagine_trajectory(self, key, starting_point: PredictionStepWithLegal, ma_rssm) ->ActorCriticTimeStep:
    #init_sample_key, trajectory_key, = jax.random.split(key)
    trajectory_key = jax.random.split(key, self.ac_trajectory_len)
    ac_default = self.default_ac_timestep()
  
    
    
    @chex.dataclass(frozen=True)
    class SampleTrajectoryCarry:
      recurrent_state:chex.Array
      deter_state: chex.Array
      joint_latent_infoset:chex.Array
      legal_actions: chex.Array
      terminal: bool
      
    init_carry = SampleTrajectoryCarry(
      recurrent_state = starting_point.recurrent_state,
      deter_state = starting_point.deter_state, #TODO: Take the one that the world model sampled, or sample anew?
      joint_latent_infoset = starting_point.joint_latent_infoset,
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
      
      #Use the infoset decoders here. We want to learn the actor/critic on
      # the latent infosets, so we just pass it through an additional layer here, 
      # to allow easy testing in the real game. However, it still requires good latent infosets.
      decoded_obs, _ = ma_rssm.get_infoset_decoder_all_no_jit(carry.joint_latent_infoset, use_symexp=False)
      #decoded_obs = ma_rssm.get_decoder_no_jit(carry.recurrent_state, carry.deter_state, use_symexp=False)
      
      obs = carry.joint_latent_infoset if not ma_rssm.use_real_infoset else decoded_obs

      #get policy 
      pi = ma_rssm.get_policy_both_no_jit(obs, carry.legal_actions, use_symlog=False)
      #uniform mix to the policy
      normalization = jnp.sum(carry.legal_actions, axis=-1, keepdims=True)
      uniform_pi = carry.legal_actions / (normalization + (normalization == 0))
      pi = self.sampling_epsilon * uniform_pi + (1 - self.sampling_epsilon) * pi
      # For each player samples a single action
      
      action_sample_key, state_sample_key = jax.random.split(key)
      action_sample_keys = jax.random.split(action_sample_key, self.num_players)
      action, action_oh = vectorized_sample_action(action_sample_keys, pi)

      
      
      next_recurrent = ma_rssm.get_next_recurrent_no_jit(carry.recurrent_state, carry.deter_state, action_oh)
      next_stoch = ma_rssm.get_dynamics_no_jit(next_recurrent)
      next_deter = sample_categorical(next_stoch, state_sample_key, sample_threshold=ma_rssm.state_sample_threshold)

      #We need the centralized decoder here. Since we are asking 
      # about the observation AFTER playing the action. So, this is actually
      # what we need to pass to the infoset network to produce our next latent infoset.
      next_obs = ma_rssm.get_decoder_no_jit(next_recurrent, next_deter, use_symexp=False)

      next_latent_infoset = ma_rssm.get_next_infoset_all_no_jit(carry.joint_latent_infoset, next_obs, action_oh, use_symlog=False)

      next_reward, next_terminal, next_legal = ma_rssm.get_predictor(next_recurrent, next_deter)
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
        recurrent_state = next_recurrent,
        deter_state = next_deter,
        joint_latent_infoset = next_latent_infoset,
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


    
  
  