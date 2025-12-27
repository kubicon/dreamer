import optax
import jax.numpy as jnp
import jax

from train_utils import OptimizerConfig


"""CODE TAKEN FROM THE OFFICIAL DREAMER_V3 IMPLEMENTATION.
AT https://github.com/danijar/dreamerv3"""
i32 = jnp.int32
f32 = jnp.float32

def _clip_by_agc(clip=0.3, pmin=1e-3):

  def init_fn(params):
    return ()

  def update_fn(updates, state, params=None):
    def fn(param, update):
      unorm = jnp.linalg.norm(update.flatten(), 2)
      pnorm = jnp.linalg.norm(param.flatten(), 2)
      upper = clip * jnp.maximum(pmin, pnorm)
      return update * (1 / jnp.maximum(1.0, unorm / upper))
    updates = jax.tree.map(fn, params, updates) if clip else updates
    return updates, ()

  return optax.GradientTransformation(init_fn, update_fn)


def _scale_by_rms(beta=0.999, eps=1e-8):

  def init_fn(params):
    nu = jax.tree.map(lambda t: jnp.zeros_like(t, f32), params)
    step = jnp.zeros((), i32)
    return (step, nu)

  def update_fn(updates, state, params=None):
    step, nu = state
    step = optax.safe_int32_increment(step)
    nu = jax.tree.map(
        lambda v, u: beta * v + (1 - beta) * (u * u), nu, updates)
    nu_hat = optax.bias_correction(nu, beta, step)
    updates = jax.tree.map(
        lambda u, v: u / (jnp.sqrt(v) + eps), updates, nu_hat)
    return updates, (step, nu)

  return optax.GradientTransformation(init_fn, update_fn)


def _scale_by_momentum(beta=0.9, nesterov=False):

  def init_fn(params):
    mu = jax.tree.map(lambda t: jnp.zeros_like(t, f32), params)
    step = jnp.zeros((), i32)
    return (step, mu)

  def update_fn(updates, state, params=None):
    step, mu = state
    step = optax.safe_int32_increment(step)
    mu = optax.update_moment(updates, mu, beta, 1)
    if nesterov:
      mu_nesterov = optax.update_moment(updates, mu, beta, 1)
      mu_hat = optax.bias_correction(mu_nesterov, beta, step)
    else:
      mu_hat = optax.bias_correction(mu, beta, step)
    return mu_hat, (step, mu)

  return optax.GradientTransformation(init_fn, update_fn)

def make_opt(
    config: OptimizerConfig
):
  """Initialize an optax optimizer. By default the same LaProp optimizer
  that was used in the DreamerV3 reference implementation."""
  chain = []
  chain.append(_clip_by_agc(config.agc))
  chain.append(_scale_by_rms(config.beta2, config.eps))
  chain.append(_scale_by_momentum(config.beta1, config.nesterov))
  assert config.anneal > 0 or config.schedule == 'const'
  if config.schedule == 'const':
    sched = optax.constant_schedule(config.lr)
  elif config.schedule == 'linear':
    sched = optax.linear_schedule(config.lr, 0.1 * config.lr, config.anneal - config.warmup)
  elif config.schedule == 'cosine':
    sched = optax.cosine_decay_schedule(config.lr, config.anneal - config.warmup, 0.1 * config.lr)
  else:
    raise NotImplementedError(config.schedule)
  if config.warmup:
    ramp = optax.linear_schedule(0.0, config.lr, config.warmup)
    sched = optax.join_schedules([ramp, sched], [config.warmup])
  chain.append(optax.scale_by_learning_rate(sched))
  return optax.chain(*chain)

"""END OF CODE TAKEN FROM https://github.com/danijar/dreamerv3"""


