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

from jaxtyping import Float
from typing import Tuple

from dlt import constants
from dlt import primary_sample_space as pss
from dlt import experiment_utils

from mcmc import gradient_restore as gr

# run the gradient restore optimization and save the results
if __name__ == '__main__':
    jnp.set_printoptions(linewidth=200)
    jax.config.update('jax_enable_x64', True)

    parser = experiment_utils.load_config_parser()
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

    render_params = dict(
        focal_length=args.focal_length,
        resolution=800,
        sensor_height_ratio=0.2,
        nrenders=0,
    )

    init_zemax_files = gr.get_zemax_reservoir_files(args.zemax_reservoir_name)

    max_surfaces = args.max_elements * 3
    init_lens = experiment_utils.load_lens_file(init_zemax_files[0], params['max_elements'])

    init_normlens = pss.lens2normalized(init_lens, params['kchar'], params['tchar'])
    init_normlens = gr.retract_lens(init_normlens, params)
    init_lens = pss.normalized2lens(init_normlens)

    retracted_lens = experiment_utils.retract_normal_lens(init_normlens,
                                                         max_curvature=params['max_curvature'], 
                                                         max_semidiam=params['max_semidiam'], 
                                                         min_distance=params['min_distance'], 
                                                         expand_semidiam=params['expand_semidiam'])

    if jnp.sum(jnp.abs(retracted_lens.data - init_normlens.data)) > constants.FLOAT_EPSILON:
        raise ValueError('Initial lens does not meet the constraints specified')

    if jnp.sum(jnp.abs(init_lens.data - pss.normalized2lens(init_normlens).data)) > constants.FLOAT_EPSILON:
        raise ValueError('Initial lens does not meet the constraints specified')


    param_eps = jnp.ones_like(init_lens.data)
    opt_epsilon = args.gradient_eps
    noise_global_scale = args.noise_eps
    noise_epsilon = noise_global_scale * param_eps

    print('Running gradient restore with the following parameters:')
    print(args)
    # config_args = experiment_utils.load_config(config_pathname)
    obj_fun = experiment_utils.create_objective_function_from_args(args, init_lens)

    def eval_obj_fun(x : pss.NormalizedLens, key) -> Tuple[Float, dict]:
        loss, aux = obj_fun(x, key)
        return loss, aux
    loss_fun = jax.jit(eval_obj_fun)
    loss_and_grad = jax.jit(eqx.filter_value_and_grad(eval_obj_fun, has_aux=True))
    (loss_init, aux_init), loss_grad_init = loss_and_grad(init_normlens, render_key)

    print(pss.normalized2lens(init_normlens).toarray())
    print('init loss', loss_init)
    print('init aux', aux_init['spot_errors'])

    loss_hist, tour_hist, (chain_hist, reservoir_hist, time_hist), reservoir = gr.jump_restore(loss_and_grad,
                                                                                 init_normlens, 
                                                                                 epsilon=opt_epsilon, 
                                                                                 noise_scale=noise_epsilon, 
                                                                                 nsteps=nsteps,
                                                                                 loss_fun=loss_fun,
                                                                                 tour_key=render_key, 
                                                                                 noise_key=noise_key, 
                                                                                 params=params)

    regenerate_times = [int(t['iteration']) for t in tour_hist if t['step_type'] == 'regenerate']
    regenerate_types = [t['jump_type'] for t in tour_hist if 'jump_type' in t]

    print('num regens', len([t for t in tour_hist if t['step_type'] == 'regenerate']))

    jnp.save(os.path.join(folder_name, 'regenerate_types.npy'), regenerate_types)
    jnp.save(os.path.join(folder_name, 'tour_hist.npy'), jnp.array(regenerate_times))
    jnp.save(os.path.join(folder_name, 'loss_hist.npy'), jnp.array(loss_hist))
    jnp.save(os.path.join(folder_name, 'time_hist.npy'), jnp.array(time_hist))

    chain_hist_folder = os.path.join(folder_name, 'chain_hist')
    os.makedirs(chain_hist_folder, exist_ok=True)
    gr.save_chain_hist(chain_hist_folder, chain_hist)

    if save_reservoir_hist:
        for i, r in enumerate(reservoir_hist):
            res_hist_folder = os.path.join(folder_name, f'reservoir_hist/r_{i:04d}')
            os.makedirs(res_hist_folder, exist_ok=True)
            gr.save_reservoir(res_hist_folder, r)

    sres = sorted(reservoir, key=lambda x: -x.ray_log_den)
    # optimize the lenses in sres
    sres_opt = gr.post_process_reservoir(loss_and_grad, loss_fun, sres, post_process_iters, epsilon=opt_epsilon, render_key=render_key, params=params)

    final_res_folder = os.path.join(folder_name, 'final_reservoir')
    os.makedirs(final_res_folder, exist_ok=True)
    gr.save_reservoir(final_res_folder, sres_opt)

    # print('regen_times', regenerate_times)
    # print('regen types', [t['regen_type'] for t in tour_hist])
    print('RESTORE')
    print(f'{"iter":<5} {"nsurf":<5} {"ray_log_den":>15}{"spot":>15}{"thru":>15}{"cost":>15}{"focal":>15}')
    print(f'{"init":<5} {init_lens.nsurfaces:<5} {loss_init:>15.5f}{aux_init["spot"]:>15.5f}{aux_init["thru"]:>15.5f}{aux_init["thickness_penalty"]:>15.5f}{aux_init["focal"]:>15.5f}')
    for i, r in enumerate(sres):
        print(f'{i:<5} {r.cur_state.nsurfaces:<5} {-r.ray_log_den:>15.5f}{r.loss_aux["spot"]:>15.5f}{r.loss_aux["thru"]:>15.5f}{r.loss_aux["thickness_penalty"]:>15.5f}{r.loss_aux["focal"]:>15.5f}')

    print('Opt')
    print(f'{"iter":<5} {"nsurf":<5} {"ray_log_den":>15}{"spot":>15}{"thru":>15}{"cost":>15}{"focal":>15}')
    print(f'{"init":<5} {init_lens.nsurfaces:<5} {loss_init:>15.5f}{aux_init["spot"]:>15.5f}{aux_init["thru"]:>15.5f}{aux_init["thickness_penalty"]:>15.5f}{aux_init["focal"]:>15.5f}')
    for i, r in enumerate(sres_opt):
        print(f'{i:<5} {r.cur_state.nsurfaces:<5} {-r.ray_log_den:>15.5f}{r.loss_aux["spot"]:>15.5f}{r.loss_aux["thru"]:>15.5f}{r.loss_aux["thickness_penalty"]:>15.5f}{r.loss_aux["focal"]:>15.5f}')

    gr.render_chain_states(folder_name, sres, render_params)
    gr.plot_lens(sres[0].cur_state, title='Best Lens using Gradient Jump Restore')
    
    print('done')