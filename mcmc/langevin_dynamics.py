import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
import jax.tree as jt
import equinox as eqx

from collections import namedtuple

from typing import NamedTuple, Tuple
from jaxtyping import Float, Array

adam_state = namedtuple('AdamState', ['eps', 'beta1', 'beta2', 'm', 'v', 't', 'stability_eps', 'mask', 'c1', 'c2'])
LangevinState = Tuple[Float, Float, adam_state]  # lens, grad, opt_state

def initialize_adam_state(init_state, nsurfaces, eps=1e-3, beta1=0.9, beta2=0.999, mask=None):
    m = jt.map(jnp.zeros_like, init_state)
    v = jt.map(jnp.zeros_like, init_state)
    t = 0
    stability_eps = 1e-4

    if mask is None:
        mask = jnp.ones_like(init_state, dtype=jnp.bool_)
        mask = mask.at[:, 2].set(False) # don't change the IORs
        mask = mask.at[:, 3].set(False) # don't change the semidiam if you can prevent it
        mask = mask.at[nsurfaces:, :].set(False) # don't change the extraneous state

    # constants for diminishing adaptation
    c1 = 1
    c2 = 1

    return adam_state(eps, beta1, beta2, m, v, t, stability_eps, mask, c1, c2)


@jax.jit
def adam_update(opt_state, new_grad):
    that = opt_state.t + 1
    mhat = jt.map(lambda m, g : opt_state.beta1 * m + (1 - opt_state.beta1) * g, opt_state.m, new_grad)
    vhat = jt.map(lambda v, g : opt_state.beta2 * v + (1 - opt_state.beta2) * jnp.square(g), opt_state.v, new_grad)

    new_opt_state = opt_state._replace(m=mhat, v=vhat, t=that)

    # bias correction
    mcc = jt.map(lambda m : m / (1 - opt_state.beta1 ** that), mhat)
    vcc = jt.map(lambda v : v / (1 - opt_state.beta2 ** that), vhat)
    aux = (mcc, vcc)

    return new_opt_state, aux


def multi_gauss_pdf(x : jnp.ndarray, mu : jnp.ndarray, sigma : jnp.ndarray, mask : jnp.ndarray):
    '''Independent multivariate normal (unnormalized) pdf
        x, mu, and var (and mask) should be all the same shape
        the pdf ignores the dimension, it assumes that you are taking a ratio
    '''
    # if sigma val is 0,a then we assume a dirac delta function
    dx = x - mu
    dx_mask = jnp.where(mask, dx, jnp.array(0.0))
    sigma_mask = jnp.where(mask, sigma, 1.0)

    logpdf = -jnp.sum(dx_mask * dx_mask / sigma_mask) / 2
    det = jnp.prod(sigma_mask)

    return jnp.exp(logpdf) / jnp.sqrt(det)


def multi_gauss_sample(key, mu, sigma, mask):
    '''Independent multivariate normal (unnormalized) pdf
        x, mu, and var should be all the same shape
        the pdf ignores the dimension, it assumes that you are taking a ratio
    '''
    dx = jr.normal(key, mu.shape)
    dx = jnp.where(mask, dx, 0.0)
    return mu + dx * jnp.sqrt(sigma)


@jax.jit
def langevin_step(cur_state : LangevinState, key):
    '''We assume that the we are not changing thee IORS of the lens'''
    lens, lens_grad, opt_state = cur_state

    eps = opt_state.eps
    mask = opt_state.mask
    grad = jnp.where(mask, lens_grad, 0.0) # don't change the IORs
    # grad = lens_grad.at[~mask].set(0.0) # don't change the IORs

    new_os, (mcc, vcc) = adam_update(opt_state, grad) # update adam state

    M0 = 1 / (jnp.sqrt(vcc) / (new_os.t**new_os.c1) + opt_state.stability_eps)
    m = (mcc / (new_os.t**new_os.c2)) + grad

    mu = lens + 0.5 * eps * (M0 * m)
    sigma = eps * M0
    new_state = multi_gauss_sample(key, mu, sigma, new_os.mask)

    return new_state, new_os


def langevin_transition_pdf(x, cur_state : LangevinState) -> Float:
    lens, grad, opt_state = cur_state

    eps = opt_state.eps

    new_os, (mcc, vcc) = adam_update(opt_state, grad)

    M0 = 1 / (jnp.sqrt(vcc) / (new_os.t**new_os.c1) + opt_state.stability_eps)
    m = (mcc / (new_os.t**new_os.c2)) + grad

    mu = lens + 0.5 * eps * (M0 * m)
    var = M0 * eps

    pdf = multi_gauss_pdf(x, mu, var, opt_state.mask)
    return pdf


@jax.jit
def langevin_transition_ratio(cur_state : LangevinState, prop_state : LangevinState) -> Float:
    new_lens, new_grad, new_opt_state = prop_state
    cur_lens, cur_grad, cur_opt_state = cur_state

    prop_to_cur = langevin_transition_pdf(cur_lens, prop_state)
    cur_to_prop = langevin_transition_pdf(new_lens, cur_state)
    return prop_to_cur / cur_to_prop


def adam_step(lens, lens_grad, opt_state):
    '''We assume that the we are not changing thee IORS of the lens'''

    # take a gradient step
    grad = lens_grad.at[:, 2:].set(0.0) # don't change the IORs

    new_os, (mcc, vcc) = adam_update(opt_state, grad) # update adam state
    M0 = 1 / jnp.sqrt(vcc + opt_state.stability_eps)

    new_state = lens + 0.5 * opt_state.eps * (M0 * mcc)

    return new_state, new_os


def adam_transition_ratio(x, cur_state):
    return 1.0
