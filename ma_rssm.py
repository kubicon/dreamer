import jax

from functools import partial


from networks import *
from optimizer import make_opt
from distributions import sample_categorical
from train_utils import *
from games.jax_game import JaxGame, InformationType

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
    self.is_iig = game.information_type() == InformationType.IIG or wm_config.use_original_iset
    self.num_actions = game.num_distinct_actions()
    self.num_players = game.num_players()
    rec_state_size = wm_config.sequential_network_details[0]
    if rec_state_size < 1:
      rec_state_size = self.num_players * game.information_state_tensor_shape()
    self.rec_state_size = rec_state_size
    enc_tokens = wm_config.encoder_network_details[0]
    deter_size = wm_config.encoded_categories * wm_config.encoded_classes
    self.encoded_classes = wm_config.encoded_classes
    self.encoded_categories = wm_config.encoded_categories
    if self.is_iig:
      self.infoset_size = game.information_state_tensor_shape()
      print(f"Using original game isets of shape {self.infoset_size}")
    else:
      self.infoset_size = (self.num_players + rec_state_size + deter_size)
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
                              self.num_players,
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
    #Watch out! The encoder always receives the
    # joint infoset from the game, never the model infosets.
    # Decoder also decodes the original infosets
    self.enc = JointIsetEncoder(game.information_state_tensor_shape(), 
                                self.num_players,
                                enc_tokens,
                                wm_config.encoder_network_details[1],
                                wm_config.encoder_network_details[2],
                                rngs
                                )
    
    self.p1_dec = Decoder(rec_state_size, 
                          game.information_state_tensor_shape(),
                          wm_config.encoded_classes,
                          wm_config.encoded_categories,
                          wm_config.decoder_network_details[0],
                          wm_config.decoder_network_details[1],
                          rngs)
    self.p2_dec = Decoder(rec_state_size, 
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
    
    self.network_names = ['dyn', 'enc', 'leg', 'observer', 'p1_dec', 'p2_dec', 'rew', 'seq', 'term']

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
    The network itself is then bound through partial, so it should not be put into
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
  def get_predictor(self, recurrent_state: chex.Array, deterministic_state:chex.Array):
    """Calls the predictor and legal actions networks and 
    passes the reward, done logits and legal action logits through
    appropriate transformations to return the actual values"""
    return self.get_predictor_no_jit(recurrent_state, deterministic_state,
                                    )
  
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
  
  @partial(nnx.jit, static_argnums=(3))
  def get_decoder(self, recurrent_state: chex.Array, deterministic_state: chex.Array, player:int):
    """Calls the decoder network and 
    applies the appropriate transformation to its output.
    Outputs either predicted real observation in single agent setting, or 
    predicted iset for a single player in a multi agent setting. """
    dec = self.p1_dec if player == 0 else self.p2_dec
    decoder_output_untransformed = MARSSM.call_net(dec, recurrent_state, deterministic_state)
    decoder_output = symexp(decoder_output_untransformed)
    #decoder_output = decoder_output_untransformed
    return decoder_output
  
  @nnx.jit
  def get_dynamics(self, recurrent_state:chex.Array):
    return MARSSM.call_net(self.dyn, recurrent_state)
  
  @nnx.jit
  def get_encoder(self, recurrent_state:chex.Array, obs: chex.Array):
    tokens = MARSSM.call_net(self.enc, obs)
    return MARSSM.call_net(self.observer, recurrent_state, tokens)
  
  def get_encoder_no_jit(self,recurrent_state:chex.Array, obs: chex.Array):
    tokens = MARSSM.call_net(self.enc, obs)
    return MARSSM.call_net(self.observer, recurrent_state, tokens)
  
  @nnx.jit
  def get_next_recurrent(self, recurrent_state:chex.Array, deterministic_state:chex.Array, joint_action:chex.Array):
    return MARSSM.call_net(self.seq, recurrent_state, deterministic_state, joint_action)
  
  def get_next_recurrent_no_jit(self, recurrent_state:chex.Array, deterministic_state:chex.Array, joint_action:chex.Array):
    return MARSSM.call_net(self.seq, recurrent_state, deterministic_state, joint_action)
  
  @partial(nnx.jit, static_argnums=(1))
  def get_init_recurrent(self, n_starts:int = 0):
    dummy_rec = jnp.zeros(self.rec_state_size)
    dummy_deter = jnp.zeros((self.encoded_classes, self.encoded_categories))
    dummy_joint_action = jnp.zeros((self.num_players, self.num_actions))
    init_rec =  MARSSM.call_net(self.seq, dummy_rec, dummy_deter, dummy_joint_action)
    #Just handle 0, or negative value as a special case for only one
    # start, without the leading batch dimension
    if n_starts > 0:
      init_rec = jnp.tile(init_rec[None, ...], (n_starts, 1))
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
  def get_obs(self, recurrent_state: chex.Array, deter_state:chex.Array):
    return self.get_obs_no_jit(recurrent_state, deter_state)
  
  def get_obs_no_jit(self, recurrent_state: chex.Array, deter_state:chex.Array):
    if self.is_iig:
      #TODO: For now, iset decoder is used to create trajectories 
      # trained on the "original" isets. This might be changed later
      p1_iset = MARSSM.call_net(self.p1_dec, recurrent_state, deter_state)
      p2_iset = MARSSM.call_net(self.p2_dec, recurrent_state, deter_state)
      obs = jnp.stack([p1_iset, p2_iset], axis=0)
      return obs
    flat_deter = deter_state.reshape((*deter_state.shape[:-2], -1))
    players_oh = jnp.eye(self.num_players)
    players_oh = jnp.reshape(players_oh, (1, ) * (flat_deter.ndim - 1) + players_oh.shape)
    model_state = jnp.concatenate([recurrent_state, flat_deter], axis=-1)
    player_model_state = jnp.concatenate([jnp.stack([model_state, model_state], axis=-2), players_oh], axis=-1)
    return player_model_state


  
  def imagine_trajectory(self, key, starting_point: PredictionStepWithLegal, ma_rssm) ->ActorCriticTimeStep:
    #init_sample_key, trajectory_key, = jax.random.split(key)
    trajectory_key = jax.random.split(key, self.ac_trajectory_len)
    ac_default = self.default_ac_timestep()
  
    
    
    @chex.dataclass(frozen=True)
    class SampleTrajectoryCarry:
      recurrent_state:chex.Array
      deter_state: chex.Array
      legal_actions: chex.Array
      terminal: bool
      
    init_carry = SampleTrajectoryCarry(
      recurrent_state = starting_point.recurrent_state,
      deter_state = starting_point.deter_state, #TODO: Take the one that the world model sampled, or sample anew?
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
      
      obs = ma_rssm.get_obs_no_jit(carry.recurrent_state, carry.deter_state)

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
      
      
      next_hidden = ma_rssm.seq(carry.recurrent_state, carry.deter_state, action_oh)
      next_stoch = ma_rssm.dyn(next_hidden)
      next_deter = sample_categorical(next_stoch, state_sample_key, sample_threshold=ma_rssm.state_sample_threshold)
      next_reward, next_terminal, next_legal = ma_rssm.get_predictor(next_hidden, next_deter)
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
        recurrent_state = next_hidden,
        deter_state = next_deter,
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


    
  
  