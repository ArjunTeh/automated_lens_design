import os
import sys
from pathlib import Path

# Add the root directory to Python path for importing dlt and mcmc modules
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))


import jax
import jax.numpy as jnp 
import jax.random as jr
import equinox as eqx

from typing import NamedTuple, Tuple, Callable
from jaxtyping import Int, Float, Array, Key

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import tqdm

from dlt.constants import FLOAT_BIG_NUMBER, FLOAT_EPSILON
from dlt import system_matrix
from dlt import zemax_loader
from dlt import opt_utils
from dlt import plot_utils
from dlt import optimize
from dlt import primary_sample_space as pss
from dlt import experiment_utils as eutils

from mcmc import chainstate, mutation
from mcmc import gradient_restore as gr
from mcmc import langevin_dynamics as ld
from mcmc import chainstate as cs


# RJ-MCMC mutation functions
# The convention is to have the rng key as the first argument if the function uses randomness


def jump(key : Key, loss_and_grad : Callable, cur_state : cs.ChainState, params) -> Tuple[cs.ChainState, Float, dict]:
    cur_nlens = cur_state.cur_state
    kchar, tchar = cur_nlens.characteristic_curvature, cur_nlens.characteristic_thickness
    cur_lens = pss.normalized2lens(cur_nlens)
    lens, aux = mutation.full_lens_mutation(key, cur_lens, params)
    nlens = pss.lens2normalized(lens, kchar, tchar)

    (prop_loss, prop_aux), prop_grad = loss_and_grad(nlens, key)

    transition_ratio = aux['transition_ratio'] * aux['jacdet']
    opt_state = ld.initialize_adam_state(nlens.data, nlens.nsurfaces)
    new_state = cs.make_chain_state_from_lens(nlens, prop_loss, prop_grad, prop_aux, opt_state)
    return new_state, transition_ratio, aux



def perturb(key : Key, loss_and_grad : Callable, cur_state : cs.ChainState, params) -> Tuple[cs.ChainState, Float, dict]:
    langevin_key, loss_key = jr.split(key, 2)
    lens = cur_state.cur_state
    grad = cur_state.ray_log_grad
    opt_state = cur_state.opt_state
    cur_langevin_state = (lens.data, grad.data, opt_state)
    prop_state, prop_os = ld.langevin_step(cur_langevin_state, langevin_key)
    prop_nlens = lens._replace(data=prop_state)

    # Need to project the lens back to the valid set after langevin step (this is a biased step)
    prop_nlens = gr.retract_lens(prop_nlens, params)

    (prop_loss, prop_aux), prop_grad = loss_and_grad(prop_nlens, loss_key)
    prop_langevin_state = (prop_nlens.data, prop_grad.data, prop_os)
    transition_ratio = ld.langevin_transition_ratio(cur_langevin_state, prop_langevin_state)

    new_state = cs.make_chain_state_from_lens(prop_nlens, prop_loss, prop_grad, prop_aux, prop_os)
    return new_state, transition_ratio, prop_aux


def rjmcmc_step(key : Key, loss_and_grad : Callable, cur_state : cs.ChainState, params):

    mutate_key, mh_key = jr.split(key, 2)

    coin = jr.uniform(mutate_key)
    split_prob = params['mutate_prob']
    if coin < split_prob:
        new_state, transition_ratio, aux = jump(mutate_key, loss_and_grad, cur_state, params)
        aux['mutate_type'] = 'jump'
    else:
        new_state, transition_ratio, aux = perturb(mutate_key, loss_and_grad, cur_state, params)
        aux['mutate_type'] = 'perturb'

    # MH acceptance step
    log_accept_ratio = (new_state.ray_log_den - cur_state.ray_log_den) + jnp.log(transition_ratio)
    accept_coin = jr.uniform(mh_key)

    if log_accept_ratio >= jnp.log(accept_coin):
        # accept
        aux['rejected_state'] = None
        new_state = new_state._replace(accept=True)
    else:
        # reject
        aux['rejected_state'] = new_state
        new_state = cur_state._replace(accept=False)

    return new_state, aux


def rjmcmc(key : Key, loss_and_grad : Callable, init_state : cs.ChainState, params : dict, nsteps : int):
    state = init_state
    states = []
    aux_list = []
    
    for i in tqdm.tqdm(range(nsteps)):
        step_key, key = jr.split(key, 2)
        state, aux = rjmcmc_step(step_key, loss_and_grad, state, params)
        states.append(state)
        aux_list.append(aux)

    return states, aux_list


if __name__ == "__main__":
    jnp.set_printoptions(linewidth=200)
    jax.config.update('jax_enable_x64', True)

    parser = eutils.load_config_parser()
    parser.add_argument('--mutate_prob', type=float, default=0.01, help='Probability of performing a jump mutation versus a perturbation')
    args = parser.parse_args()

    nsteps = args.niters
    nrays = args.nrays
    post_process_iters = args.post_process_iters
    render_key = jr.PRNGKey(args.render_seed)
    noise_key = jr.PRNGKey(args.noise_seed)
    save_reservoir_hist = args.save_reservoir_hist

    folder_name = args.out_folder
    os.makedirs(folder_name, exist_ok=True)

    # save the config file
    config_pathname = os.path.join(folder_name, 'config.yaml')
    parser.write_config_file(args, [config_pathname])

    print('objective weights')
    print('spot:', args.spot_weight)
    print('thru:', args.thru_weight)
    print('thickness_penalty:', args.thick_weight)
    print('focal:', args.focal_weight)
    print('nelements:', args.nelements_weight)
    print('track:', args.track_weight)
    print('no_project:', args.no_projection)

    params = gr.restore_params_from_args(args)
    params['mutate_prob'] = args.mutate_prob

    render_params = dict(
        focal_length=args.focal_length,
        resolution=800,
        sensor_height_ratio=0.2,
        nrenders=0,
    )

    args.zemax_reservoir_name = '50mm'
    init_zemax_files = gr.get_zemax_reservoir_files(args.zemax_reservoir_name)

    max_surfaces = args.max_elements * 3
    init_lens = eutils.load_lens_file(init_zemax_files[0], params['max_elements'])

    init_normlens = pss.lens2normalized(init_lens, params['kchar'], params['tchar'])
    init_normlens = gr.retract_lens(init_normlens, params)
    init_lens = pss.normalized2lens(init_normlens)

    retracted_lens = eutils.retract_normal_lens(init_normlens,
                                                max_curvature=params['max_curvature'], 
                                                max_semidiam=params['max_semidiam'], 
                                                min_distance=params['min_distance'], 
                                                expand_semidiam=params['expand_semidiam'])

    if jnp.sum(jnp.abs(retracted_lens.data - init_normlens.data)) > FLOAT_EPSILON:
        raise ValueError('Initial lens does not meet the constraints specified')

    if jnp.sum(jnp.abs(init_lens.data - pss.normalized2lens(init_normlens).data)) > FLOAT_EPSILON:
        raise ValueError('Initial lens does not meet the constraints specified')


    param_eps = jnp.ones_like(init_lens.data)
    opt_epsilon = args.gradient_eps
    noise_global_scale = args.noise_eps
    noise_epsilon = noise_global_scale * param_eps

    print('Running gradient restore with the following parameters:')
    print(args)
    # config_args = experiment_utils.load_config(config_pathname)
    obj_fun = eutils.create_objective_function_from_args(args, init_lens)

    def eval_obj_fun(x : pss.NormalizedLens, key) -> Tuple[Float, dict]:
        loss, aux = obj_fun(x, key)
        return loss, aux

    loss_fun = jax.jit(eval_obj_fun)
    loss_and_grad = jax.jit(eqx.filter_value_and_grad(eval_obj_fun, has_aux=True))
    (loss_init, aux_init), loss_grad_init = loss_and_grad(init_normlens, render_key)
    init_state = cs.make_chain_state_from_lens(init_normlens, loss_init, loss_grad_init, aux_init)

    state_hist, aux_hist = rjmcmc(render_key, loss_and_grad, init_state, params, nsteps)

    loss_hist = jnp.array([ -s.ray_log_den for s in state_hist])
    accept_hist = jnp.array([s.accept for s in state_hist])

    # grab the top 5 unique lenses
    accepted_losses = loss_hist[accept_hist]
    sorted_accept_indices = jnp.argsort(accepted_losses, descending=True)
    accepted_states = [s for s, a in zip(state_hist, accept_hist) if a]
    reservoir = [accepted_states[idx] for idx, _ in zip(sorted_accept_indices, range(5))]
    gr.log_reservoir(reservoir)
    gr.plot_lens(reservoir[0].cur_state, title='Best Lens in RJ-MCMC Reservoir')

    # grab the top 5 best lenses
    sorted_indices = jnp.argsort(loss_hist)
    best_indices = sorted_indices[:5]
    for i, idx in enumerate(best_indices):
        state = state_hist[idx]
        fname = os.path.join(folder_name, f'rjmcmc_best_lens_{i}_iter_{idx}_loss_{-state.ray_log_den:.4f}.npz')
        cs.save_chain_state(fname, state)


    # plot the loss history
    # plot the acceptance ratio?
    accepted_losses = loss_hist[accept_hist]
    accepted_times = jnp.arange(len(loss_hist))[accept_hist]




