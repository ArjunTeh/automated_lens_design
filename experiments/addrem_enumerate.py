import os
import sys
from pathlib import Path

# Add the root directory to Python path for importing dlt and mcmc modules
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import jax
import jax.numpy as jnp
import jax.random as jr

import csv
import time
import configargparse

import equinox as eqx

import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go

from typing import Callable, Iterable

from dlt import primary_sample_space as pss
from dlt import optical_properties
from dlt import opt_utils
from dlt import plot_utils
from dlt import zemax_loader
from dlt import experiment_utils
from dlt.zemax_loader import ZemaxSurfaceType

from mcmc import chainstate
from mcmc import gradient_restore as gr
from mcmc import mutation

# import experiment_plot_utils as eplot
# import render_lens


def optimize_with_optax_step(opt_step, lens : pss.NormalizedLens, opt, niters, key, opt_params, fixed_key):
    init_opt_state = opt.init(lens.data)
    lens_hist, result_hist, times = experiment_utils.optimize_lens_with_optax_step(opt_step, 
                                                                         lens, 
                                                                         init_opt_state, 
                                                                         niters, 
                                                                         key, 
                                                                         opt_params, 
                                                                         fixed_key=fixed_key)

    aux_hist = [aux for (_, aux), _ in result_hist]
    loss_hist = [l for (l, aux), _ in result_hist]
    grad_hist = [g for _, g in result_hist]

    results = {}
    results['times'] = times
    results['loss_hist'] = loss_hist
    results['lens_hist'] = lens_hist
    results['aux_hist'] = aux_hist
    results['grad_hist'] = grad_hist
    return results


def load_lens_from_zemax_file(fname, config_args):
    raw_lens, lens_type_list = zemax_loader.load_zemax_file(fname, info_list=True)
    sani_lens, lens_type_list = zemax_loader.sanitize_zemax_lens(raw_lens, lens_type_list)
    # sani_lens = raw_lens
    
    lens_idx = pss.get_lens_idx(sani_lens)
    if len(lens_idx) > config_args.max_elements:
        raise ValueError(f'Lens has too many elements for config: {len(lens_idx)} > {config_args.max_elements}')

    sani_lens = jnp.array(sani_lens)
    sani_lens = optical_properties.clip_semidiameter_to_curvature(sani_lens)

    orig_lens = jnp.array(sani_lens)
    lens_data = pss.homogenize_lens(orig_lens, config_args.max_elements*3)
    lens = pss.Lens(data=lens_data, nsurfaces=orig_lens.shape[0])
    return lens


def save_opt_info(expname, opt_info):
    loss_hist = opt_info['loss_hist']

    if len(loss_hist) == 0:
        return # no optimization was performed

    final_lens = opt_info['lens_hist'][-1]
    final_aux = opt_info['aux_hist'][-1]
    final_loss = loss_hist[-1]
    final_grad = opt_info['grad_hist'][-1]
    state = gr.make_chain_state(final_lens, final_loss, final_grad, final_aux, opt_state=None)
    chainstate.save_chain_state(expname + '_final_state.npz', state)
    jnp.save(expname + '_loss_hist.npy', loss_hist)


if __name__ == '__main__':

    jnp.set_printoptions(linewidth=200)
    jax.config.update('jax_enable_x64', True)
    # jax.config.update('jax_disable_jit', True)
    # jax.config.update('jax_debug_nans', True)

    parser = experiment_utils.load_config_parser()
    args = parser.parse_args()

    lens_files = gr.get_zemax_reservoir_files(args.zemax_reservoir_name)

    out_folder = args.out_folder
    os.makedirs(out_folder, exist_ok=True)

    # save the config file
    config_pathname = os.path.join(out_folder, 'config.yaml')
    parser.write_config_file(args, [config_pathname])

    post_process_iters = args.post_process_iters
    eps = args.gradient_eps
    key = jr.key(args.render_seed)
    max_semidiam = args.max_semidiam

    lens = load_lens_from_zemax_file(lens_files[0], args)
    correct_lens = lens
    lens_idx = pss.get_lens_idx(lens.data)

    init_obj_fun = experiment_utils.create_objective_function_from_args(args, lens)

    @eqx.filter_jit
    def obj_fun(*args):
        loss, aux = init_obj_fun(*args)
        return loss / 1.0, aux

    obj_fun_and_grad = jax.jit(eqx.filter_value_and_grad(obj_fun, has_aux=True))

    k_char = 0.1
    t_char = 10.0

    optimizer : str = args.optimizer
    niters = args.niters // (2 * lens_idx.shape[0])

    opt_params = dict(
        ior_air=1.0,
        ior_glass=1.6,
        max_semidiam=max_semidiam,
        min_thickness=0.1,
        min_distance=args.min_thickness,
        max_curvature=args.max_curvature,
        expand_semidiam=args.expand_semidiam,
        singlet_curv_distribution=(0.0, 0.1 / args.focal_length),
        singlet_dist_distribution=(0.5, 5.0),
        use_projection=False,
        optimizer=optimizer,
    )

    normlens = pss.lens2normalized(lens, k_char, t_char)

    init_loss, init_aux = obj_fun(normlens, key)
    print(init_loss)
    print(init_aux)

    opt_step, opt = experiment_utils.create_optax_step(obj_fun_and_grad, eps, opt_params, obj_fun, optimizer=opt_params['optimizer'])

    final_opt_step, final_opt = experiment_utils.create_optax_step(obj_fun_and_grad, eps, opt_params, obj_fun, optimizer='lbfgs')

    if jnp.all(init_aux['valid']):
        raise ValueError('Beam is too small to evaluate throughput')

    lens_idx = pss.get_lens_idx(lens.data)

    def opt_lens_and_save(lens, name):
        norm_lens = pss.lens2normalized(lens, k_char, t_char)
        init_norm_lens = experiment_utils.retract_normal_lens(norm_lens, opt_params['max_curvature'], opt_params['min_distance'], opt_params['max_semidiam'], opt_params['expand_semidiam'])
        first_opt_info = optimize_with_optax_step(opt_step, init_norm_lens, opt, niters, key, opt_params, fixed_key=False)
        second_opt_info = optimize_with_optax_step(final_opt_step, first_opt_info['lens_hist'][-1], final_opt, post_process_iters, key, opt_params, fixed_key=True)

        print('opt loss', first_opt_info['loss_hist'][-1])
        save_opt_info(os.path.join(out_folder, name), first_opt_info)
        save_opt_info(os.path.join(out_folder, name + '_post'), second_opt_info)


    for i in range(len(lens_idx)):
        success, addlens, _ = mutation.add_singlet(key, lens, lens_idx[i], opt_params)
        opt_lens_and_save(addlens, f'add_{i}')

        success, remlens, _ = mutation.remove_singlet(key, lens, lens_idx[i], opt_params)
        opt_lens_and_save(remlens, f'rem_{i}')


    chain_states = []
    for i in range(lens_idx.shape[0]):
        fname = os.path.join(out_folder, f"add_{i}_final_state.npz")
        chain_state = chainstate.load_chain_state(fname)
        chain_states.append(chain_state)

        fname = os.path.join(out_folder, f"rem_{i}_final_state.npz")
        chain_state = chainstate.load_chain_state(fname)
        chain_states.append(chain_state)

    gr.log_reservoir(chain_states)

    # plot the best lens from the enumeration
    sorted_states = sorted(chain_states, key=lambda s: -s.ray_log_den, reverse=False)
    gr.plot_lens(sorted_states[0].cur_state, title='Best Lens from Add-Remove Enumeration')
    print('add rem enumerate done')
