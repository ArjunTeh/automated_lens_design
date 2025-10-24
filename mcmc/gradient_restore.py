import os
import time
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from tqdm import tqdm

from typing import NamedTuple, Tuple, Iterable, Union
from jaxtyping import Float, Int
import plotly.graph_objects as go

from dlt import (
    constants,
    plot_utils,
    optical_properties,
    primary_sample_space as pss,
    zemax_loader,
    experiment_utils,
    render_lens
)

from mcmc import mutation
from mcmc.chainstate import ChainState, load_chain_state, save_chain_state

OptaxOptimizer = Union[optax.GradientTransformation, optax.GradientTransformationExtraArgs]

def norm2(x : jnp.ndarray):
    return jnp.sum(x**2)


def sample_singlet_lens(key, params) -> pss.NormalizedLens:
    '''Sample a new lens at random'''
    element_key, sample_key = jr.split(key)
    nelements = jr.choice(element_key, params['max_elements'] - params['min_elements'] + 1) + params['min_elements']
    newlens = mutation.sample_set_of_singlets(element_key, nelements, params)
    return pss.lens2normalized(newlens, params['kchar'], params['tchar'])


def mu0_sample(key, state : pss.NormalizedLens, params) -> pss.NormalizedLens:
    '''Sample a new lens from the reservoir'''
    curv_bounds = params['curvature_bounds']
    thic_bounds = params['thickness_bounds']
    dist_bounds = params['distance_bounds']

    tot_surfs = state.data.shape[0]
    curvs = jr.uniform(key, (tot_surfs,), minval=curv_bounds[0], maxval=curv_bounds[1])
    thics = jr.uniform(key, (tot_surfs,), minval=thic_bounds[0], maxval=thic_bounds[1])
    dists = jr.uniform(key, (tot_surfs,), minval=dist_bounds[0], maxval=dist_bounds[1])

    nair = state.data[-1, 2]
    
    tvals = jnp.where(state.data[:, 2] > nair, thics, dists)
    lens = jnp.stack([curvs, tvals, state.data[:, 2], state.data[:, 3]], axis=1)
    lens = lens.at[state.nsurfaces-1, 1].set(40.0)

    return state._replace(data=lens)


def reservoir_sample(key, reservoir : list[ChainState], cur_iter, params) -> Tuple[ChainState, Int]:
    # simply sample uniformly from the reservoir based on the target_distribution
    temperature = params['temperature_schedule'](cur_iter)

    if params['reservoir_greedy_sample']:
        # greedy sampling, bias towards the better states in the reservoir
        weights = jnp.array([jnp.exp(c.ray_log_den / temperature) for c in reservoir])
    else:
        weights = jnp.ones(len(reservoir))
    choice = jr.choice(key, len(reservoir), p=weights)
    return reservoir[choice], choice


def reservoir_update(key, reservoir : list[ChainState], new_state : ChainState, params, reservoir_idx : Int = -1):
    '''Update the reservoir with a new state to keep track of the best states'''
    if reservoir_idx >= 0 and reservoir_idx < len(reservoir):
        reservoir[reservoir_idx] = new_state
        return reservoir

    if len(reservoir) < params['reservoir_size']:
        reservoir.append(new_state)
    else:
        # check to see if the topological lens exists already
        for i, r in enumerate(reservoir):
            if pss.topological_equals(r.cur_state, new_state.cur_state):
                # if it exists, just update the state
                if new_state.ray_log_den > reservoir[i].ray_log_den:
                    # update the state
                    reservoir[i] = new_state
                return reservoir

        # otherwise find the state with the worst performance and replace it if the new state is better
        min_idx = min(range(len(reservoir)), key=lambda i: reservoir[i].ray_log_den)
        if new_state.ray_log_den > reservoir[min_idx].ray_log_den:
            reservoir[min_idx] = new_state
    return reservoir


@eqx.filter_jit
def mu_pdf(state : pss.NormalizedLens, reservoir, params):
    curv_bounds = params['curvature_bounds']
    thic_bounds = params['thickness_bounds']
    dist_bounds = params['distance_bounds']

    lens = pss.normalized2lens(state)

    curv_area = (curv_bounds[1] - curv_bounds[0])
    dist_area = (dist_bounds[1] - dist_bounds[0])
    thic_area = (thic_bounds[1] - thic_bounds[0])

    nair = lens.data[-1, 2]
    nelems = jnp.count_nonzero(lens.data[:, 2] > nair)
    ntot = lens.data.shape[0]
    thic_ratio = nelems / ntot
    pdf = (1 / curv_area) * ((1 - thic_ratio) / dist_area) * (thic_ratio / thic_area)

    return pdf


def make_chain_state(lens : pss.NormalizedLens, loss, grad, aux, opt_state) -> ChainState:
    cur_state = ChainState(accept=True, 
                           cur_state=lens, 
                           ray_log_den=-loss, 
                           ray_log_grad=grad, 
                           opt_state=opt_state, 
                           loss_aux=aux)
    return cur_state


def retract_lens(lens : pss.NormalizedLens, params) -> pss.NormalizedLens:
    lens = experiment_utils.retract_normal_lens(lens, 
                                                max_curvature=params['max_curvature'], 
                                                max_semidiam=params['max_semidiam'], 
                                                min_distance=params['min_distance'], 
                                                expand_semidiam=params['expand_semidiam'])
    return lens


def regenerate_lens(state : ChainState, reservoir, key, cur_iter, params, loss_and_grad, optimizer) -> Tuple[ChainState, str, Int]:
    # The t=0 distribution is the uniform distribution over the space of lenses
    # sample probability of choosing from the reservoir vs the uniform distribution
    a = params['discrete_measure_dominance']
    if a < 0.0:
        reservoir_prob = 0.0
    else:
        reservoir_prob = len(reservoir) / (len(reservoir) + a)

    key, coinkey = jr.split(key)
    if jr.uniform(coinkey) < reservoir_prob:
        new_lens_state, reservoir_idx = reservoir_sample(key, reservoir, cur_iter, params)
        sample_type = 'reservoir'
    else:
        new_lens = mu0_sample(key, state.cur_state, params)
        (loss, aux), grad = loss_and_grad(new_lens, key)
        new_opt_state = optimizer.init(new_lens.data)
        new_lens_state = make_chain_state(new_lens, loss, grad, aux, new_opt_state)

        sample_type = 'mu0'
        reservoir_idx = len(reservoir)

    return new_lens_state, sample_type, reservoir_idx


@eqx.filter_jit
def sample_killing_rate(state : ChainState, next_state : ChainState, reservoir, cur_iter, params) -> Float:
    '''Calculate the killing rate for the process.
        This is calculated by the ratio of the target distribution at next_state and state (the state that took a gradient step to next_state)
    '''
    temp_schedule = params['temperature_schedule']
    temperature = temp_schedule(cur_iter)

    kplus = params['expected_tour_lifetime'] * temperature
    muval = mu_pdf(next_state.cur_state, reservoir, params)

    kpartial = jnp.exp((state.ray_log_den - next_state.ray_log_den) / temperature) - 1
    regen_partial = muval / jnp.exp(next_state.ray_log_den / temperature)

    kr = kpartial + kplus * regen_partial
    aux = dict(
        mu=muval,
        kpartial=kpartial,
        pi=jnp.exp(next_state.ray_log_den / temperature),
        regen_partial=regen_partial,
    )
    return kr, aux


def save_chain_hist(dirname : str, chain_hist : Iterable[ChainState]):
    for i, chain_state in enumerate(chain_hist):
        fname = os.path.join(dirname, f'chain_{i:05d}.npz')
        save_chain_state(fname, chain_state)


def save_reservoir(dirname : str, reservoir : Iterable[ChainState]):
    for i, chain_state in enumerate(reservoir):
        fname = os.path.join(dirname, f'reservoir_{i:05d}.npz')
        save_chain_state(fname, chain_state)


def render_chain_states(dirname : str, chain_hist : Iterable[ChainState], render_params):
    for i, chain_state in enumerate(chain_hist):
        render_name = os.path.join(dirname, f'chain_{i:05d}_render.png')
        focal_length = render_params['focal_length']
        resolution = render_params['resolution']
        sensor_height_ratio = render_params['sensor_height_ratio']
        if i < render_params['nrenders']:
            lens = pss.normalized2lens(chain_state.cur_state)
            render_lens.render_lens(lens, render_name, focal_length, resolution, rng_seed=0, sensor_height_ratio=sensor_height_ratio)


def jump_lens(normlens : pss.NormalizedLens, key, params) -> Tuple[bool, pss.NormalizedLens, Float, dict]:
    '''Jump to a new lens state'''
    '''We might consider using the version of the jump that uses the reservoir instead'''
    # new_lens, transition_ratio, aux = mutation.full_lens_mutation(key, lens, params)

    lens = pss.normalized2lens(normlens)

    kchar, tchar = normlens.characteristic_curvature, normlens.characteristic_thickness
    # new_lens, aux = mutation.full_lens_mutation_new(key, lens, params)
    new_lens, aux = mutation.full_lens_mutation(key, lens, params)
    transition_ratio = 1.0
    if not aux['mutate_success']:
        new_lens = lens
        transition_ratio = 1.0

    success = aux['mutate_success']
    new_normlens = pss.lens2normalized(new_lens, kchar, tchar)
    return success, new_normlens, transition_ratio, aux


def create_optimizer_step(optimizer, loss_fun, epsilon, params):
    if optimizer == 'adam':
        opt = optax.adam(epsilon)
    elif optimizer == 'adam_linesearch':
        opt = optax.chain(
            optax.adam(epsilon),
            optax.scale_by_zoom_linesearch(10)
        )
    elif optimizer == 'lbfgs':
        opt = optax.lbfgs(epsilon)
    else:
        raise ValueError('Unknown optimizer: {}'.format(optimizer))

    @jax.jit
    def single_opt_step(lens : pss.NormalizedLens, lens_grad : pss.NormalizedLens, lens_loss, opt_state, key):
        def val_fun(xdat):
            lens_update = lens._replace(data=xdat)
            return loss_fun(lens_update, key)[0]

        # project the gradient since projection depends on opt parameters
        grad = experiment_utils.grad_project(lens, lens_grad, expand_semidiam=params['expand_semidiam'])

        if optimizer == 'lbfgs' or optimizer == 'adam_linesearch':
            updates, new_opt_state = opt.update(
                grad.data, opt_state, lens.data, 
                value=lens_loss, grad=grad.data, value_fn=val_fun) #type:ignore
        elif optimizer == 'adam':
            updates, new_opt_state = opt.update(grad.data, opt_state, lens.data)
        else:
            raise ValueError(f'Invalid optimizer: {optimizer}')

        new_lens_dat = optax.apply_updates(lens.data, updates)
        newlens = lens._replace(data=new_lens_dat)
        newlens = retract_lens(newlens, params)
        return newlens, new_opt_state

    return opt, single_opt_step


def jump_restore(loss_and_grad, lens : pss.NormalizedLens, epsilon : Float, noise_scale : Float, nsteps : int, loss_fun, tour_key, noise_key, params, log_callback=None):
    # Main function for the gradient restore MCMC sampler

    if params['fixed_local_iterations'] is not None:
        print('NOTE: using fixed local iterations:', params['fixed_local_iterations'])

    opt, single_opt_step = create_optimizer_step(params['optimizer'], loss_fun, epsilon, params)
    opt_state = opt.init(lens.data)

    render_key, tour_key = jr.split(tour_key)
    lens = retract_lens(lens, params)
    (new_ray_loss, ray_aux), ray_grad = loss_and_grad(lens, render_key)
    cur_state = make_chain_state(lens, new_ray_loss, ray_grad, ray_aux, opt_state)

    @eqx.filter_jit
    def opt_step(key, cur_state : ChainState):
        # render_key, tour_key = jr.split(key)

        state = cur_state
        for i in range(params['local_step_iterations']):
            lens = state.cur_state
            cur_loss = -state.ray_log_den
            cur_grad = state.ray_log_grad
            opt_state = state.opt_state
    
            newlens, new_opt_state = single_opt_step(lens, cur_grad, cur_loss, opt_state, key)
    
            (new_ray_loss, ray_aux), new_ray_grad = loss_and_grad(newlens, key)
            state = make_chain_state(newlens, new_ray_loss, new_ray_grad, ray_aux, new_opt_state)

        # Replace the for loop with lax.fori_loop
        new_state = state
        return new_state

    reservoir = load_reservoir(render_key, loss_and_grad, opt, params)
    cur_state, reservoir_idx = reservoir_sample(tour_key, reservoir, 0, params)
    best_state = cur_state
    tour_hist = []
    loss_hist = [-cur_state.ray_log_den]
    chain_hist = [cur_state]
    reservoir_hist = [reservoir.copy()]
    ideal_tour_constant = jnp.array(1e-6)
    num_negative_regen_rates = 0
    last_regen_iter = 0
    start_time = time.time()
    time_hist = []
    for i in tqdm(range(nsteps)):

        new_state = opt_step(tour_key, cur_state)

        tour_key, killkey, subkey = jr.split(tour_key, 3)
        killing_rate, kr_aux = sample_killing_rate(cur_state, new_state, reservoir, i, params)
        k = jr.exponential(killkey) / killing_rate
        t = jr.exponential(subkey)

        cur_ideal_tour_constant = jnp.abs(kr_aux['kpartial'] / kr_aux['regen_partial'])
        ideal_tour_constant = jnp.maximum(ideal_tour_constant, cur_ideal_tour_constant)

        if killing_rate < constants.FLOAT_EPSILON:
            # print('killing rate is negative:', killing_rate)
            # print('need tour rate to be: ', cur_ideal_tour_constant)
            num_negative_regen_rates = num_negative_regen_rates + 1
            killing_rate = jnp.clip(killing_rate, min=constants.FLOAT_EPSILON)
            k = 1.0
            t = 0.0

        if params['fixed_local_iterations'] is not None:
            time_since_last_regen = i - last_regen_iter
            # force the local dynamics when iteration count hasn't been reached
            if time_since_last_regen >= params['fixed_local_iterations']:
                t = 1.0
                k = 0.0
                last_regen_iter = i
            else:
                t = 0.0
                k = 1.0

        tour_step_info = dict(
            killing_rate=killing_rate,
            iteration=i,
        )

        if t > k:
            # save the current state in the reservoir and regenerate
            noise_key, noise_subkey, jumpkey, addkey = jr.split(noise_key, 4)
            reservoir = reservoir_update(addkey, reservoir, cur_state, params, reservoir_idx)
            regen_state, regen_sample_type, reservoir_idx = regenerate_lens(cur_state, reservoir, noise_subkey, i, params, loss_and_grad, opt)
            next_state = regen_state

            # store info
            tour_step_info['step_type'] = 'regenerate'
            tour_step_info['regen_type'] = regen_sample_type
            tour_step_info['tdelta'] = k

            if not params['no_mutation']:
                jumpkey, coin_key = jr.split(jumpkey)
                mutate_prob = params['temperature_schedule'](i) / params['temperature_schedule'](0)
                if jr.uniform(coin_key) > mutate_prob:
                    tour_step_info['jump_type'] = 'nomutate'
                else:
                    regen_lens = regen_state.cur_state
                    jump_success, lens, transition_ratio, mutate_aux = jump_lens(regen_lens, jumpkey, params)

                    if pss.get_lens_idx(lens.data).shape[0] > params['max_elements']:
                        raise ValueError('Too many elements added')
                
                    if lens.nsurfaces > params['max_elements'] * 3:
                        raise ValueError('Too many surfaces added')

                    tour_step_info['jump_type'] = mutate_aux['mutate_type']
                    tour_step_info['jump_aux'] = mutate_aux
                    if jump_success:
                        reservoir_idx = -1

                    lens = retract_lens(lens, params)
                    (new_ray_loss, new_ray_aux), new_ray_grad = loss_and_grad(lens, render_key)
                    opt_state = opt.init(lens.data)
                    next_state = make_chain_state(lens, new_ray_loss, new_ray_grad, new_ray_aux, opt_state)

            chain_hist.append(cur_state)
            reservoir_hist.append(reservoir.copy())
            opt_state = opt.init(lens.data)
        else:
            tour_step_info['step_type'] = 'local'
            tour_step_info['tdelta'] = t
            next_state = new_state

        cur_state = next_state
        tour_hist.append(tour_step_info)
        time_hist.append(time.time() - start_time)
        loss_hist.append(-cur_state.ray_log_den)

        if cur_state.ray_log_den > best_state.ray_log_den:
            best_state = cur_state

        if log_callback:
            log_callback(i, cur_state, tour_step_info, best_state)

    reservoir = reservoir_update(noise_key, reservoir, cur_state, params)
    reservoir_hist.append(reservoir.copy())
    print('ideal tour constant:', ideal_tour_constant)
    print('num_negative_regen_rates', num_negative_regen_rates)

    return loss_hist, tour_hist, (chain_hist, reservoir_hist, time_hist), reservoir


def load_reservoir(render_key, loss_fn, opt : OptaxOptimizer, params):
    if params['load_reservoir_folder'] is not None:
        print(f'Loading reservoir from {params["load_reservoir_folder"]}')
        return load_chain_reservoir(params['load_reservoir_folder'], opt, params)

    if params['random_lens_initialization']:
        normlens = sample_singlet_lens(render_key, params)
        init_normlens = retract_lens(normlens, params)

        opt_state = opt.init(init_normlens.data)
        (new_ray_loss, ray_aux), ray_grad = loss_fn(init_normlens, render_key)
        cur_state = make_chain_state(init_normlens, new_ray_loss, ray_grad, ray_aux, opt_state)
        return [cur_state]
    
    return load_zemax_reservoir(render_key, loss_fn, opt, params)


def load_chain_reservoir(reservoir_folder : str, opt : OptaxOptimizer, params) -> list[ChainState]:
    reservoir = []
    for fname in os.listdir(reservoir_folder):
        if not fname.endswith('.npz'):
            continue
        data = load_chain_state(os.path.join(reservoir_folder, fname))
        lens = data.cur_state
        grad = data.ray_log_grad
        lens_data_pad = pss.homogenize_lens(lens.data, params['max_surfaces'])
        grad_data_pad = pss.homogenize_lens(grad.data, params['max_surfaces'])
        lens = lens._replace(data=lens_data_pad)
        grad = grad._replace(data=grad_data_pad)
        data = data._replace(cur_state=lens, 
                             ray_log_grad=grad,
                             opt_state=opt.init(lens.data))
        reservoir.append(data)

    if len(reservoir) == 0:
        raise ValueError('No reservoir data found')

    return reservoir


def load_zemax_reservoir(render_key, loss_fn, opt : OptaxOptimizer, params):
    zemax_reservoir_name = params.get('zemax_reservoir_name', '50mm')
    lens_fnames = get_zemax_reservoir_files(zemax_reservoir_name)

    reservoir = []
    for fname in lens_fnames:
        lens_data, lens_type_list = zemax_loader.load_zemax_file(fname, info_list=True)
        lens_data, lens_type_list = zemax_loader.sanitize_zemax_lens(lens_data, lens_type_list)
        nsurfaces = lens_data.shape[0]
        lens_data = pss.homogenize_lens(jnp.array(lens_data), params['max_surfaces'])
        init_lens_data = pss.homogenize_lens(lens_data, params['max_surfaces'])
        init_lens = pss.Lens(data=init_lens_data, nsurfaces=nsurfaces)
        init_normlens = pss.lens2normalized(init_lens, params['kchar'], params['tchar'])
        init_normlens = retract_lens(init_normlens, params)

        opt_state = opt.init(init_normlens.data)
        (new_ray_loss, ray_aux), ray_grad = loss_fn(init_normlens, render_key)
        cur_state = make_chain_state(init_normlens, new_ray_loss, ray_grad, ray_aux, opt_state)

        reservoir.append(cur_state)

    return reservoir


def get_zemax_reservoir_files(zemax_reservoir_name):
    print('Reservoir: Loading', zemax_reservoir_name)
    if zemax_reservoir_name == '50mm':
        lens_fnames = [
            'data/double_gauss_50mm/US03560079.zmx',
        ]
    else:
        if os.path.isdir(zemax_reservoir_name):
            lens_fnames = [os.path.join(zemax_reservoir_name, f) for f in os.listdir(zemax_reservoir_name)]
        else:
            raise ValueError('Unknown lens reservoir name: {}'.format(zemax_reservoir_name))

    return lens_fnames


def plot_lens(result : pss.NormalizedLens, title='Lens Visualization'):
    res_unnorm = pss.normalized2lens(result)
    lens = res_unnorm.data[:result.nsurfaces]

    angles = [jnp.atan2(a, 50.0) for a in [0.0, 8.0, 12.0]]
    traces = plot_utils.visualize_state_and_rays(lens, angle_list=angles, focal_length=50.0, nrays=15, show_invalid_rays=False, ignore_L=True)

    fig = go.Figure()
    fig.add_traces(traces)
    fig.update_layout(title=title)
    fig.show()


def plot_focus_curves(result : pss.NormalizedLens, angles):
    res_unnorm = pss.normalized2lens(result)
    lens = res_unnorm.toarray()

    fig = go.Figure()
    for ang in angles:
        tan, sag, chief = optical_properties.focus_curves(lens, ang, 100, 100)
        fig.add_trace(go.Scatter(y=tan[0], x=tan[1], name=f"Tangential {jnp.rad2deg(ang):.1f}"))
        fig.add_trace(go.Scatter(y=sag[0], x=sag[1], name=f"Saggital {jnp.rad2deg(ang):.1f}"))
    fig.show()


def post_process_reservoir(loss_and_grad, loss_fun, lens_states : Iterable[ChainState], niters, epsilon, render_key, params, fixed_key=True) -> Iterable[ChainState]:
    if niters <= 0:
        # just return the lens states before attempting to initialize the optimization
        return lens_states

    opt_step, opt = experiment_utils.create_optax_step(loss_and_grad, epsilon, params, loss_fun, optimizer='adam')

    sres_opt = []
    for i, s in enumerate(lens_states):
        print(f'Optimizing lens {i}')
        opt_state = opt.init(s.cur_state.data)
        lens_hist, ray_loss, times = experiment_utils.optimize_lens_with_optax_step(opt_step, s.cur_state, opt_state, niters, key=render_key, params=params, fixed_key=fixed_key)
        print('improvement', ray_loss[0][0][0] - ray_loss[-1][0][0])

        new_lens = lens_hist[-1]
        (newloss, newaux), newgrad = loss_and_grad(new_lens, render_key)
        newchain = make_chain_state(new_lens, newloss, newgrad, newaux, s.opt_state)
        sres_opt.append(newchain)

    return sres_opt


def log_reservoir(reservoir):
    print(f'{"iter":<5} {"nsurf":<5} {"ray_log_den":>15}{"spot":>15}{"thru":>15}{"cost":>15}{"focal":>15}')
    for i, r in enumerate(reservoir):
        print(f'{i:<5} {r.cur_state.nsurfaces:<5} {-r.ray_log_den:>15.5f}{r.loss_aux["spot"]:>15.5f}{r.loss_aux["thru"]:>15.5f}{r.loss_aux["thickness_penalty"]:>15.5f}{r.loss_aux["focal"]:>15.5f}')


def restore_params_from_args(args):
    kmax = args.max_curvature
    temperature = args.temperature
    corrected_tour_lifetime = args.tour_lifetime
    # cooling_schedule = optax.linear_schedule(temperature, 0.1, nsteps, nsteps//2)
    # cooling_schedule = optax.exponential_decay(temperature, 20, 0.8, nsteps//2, end_value=0.1*temperature)
    cooling_schedule = optax.constant_schedule(temperature)


    params = dict(
        curvature_bounds=(-kmax, kmax),
        thickness_bounds=(0.1, 10.0),
        distance_bounds=(0.01, 10.0),
        reservoir_size=args.reservoir_size,
        reservoir_greedy_sample=args.reservoir_greedy_sample,
        expected_tour_lifetime=corrected_tour_lifetime,
        discrete_measure_dominance=args.discrete_measure_dominance,
        split_prob=args.split_prob,
        cement_split_prob=0.5,
        min_elements=args.min_elements,
        max_elements=args.max_elements,
        max_surfaces=args.max_elements * 3,
        singlet_curv_distribution=(0.0, 0.005),
        singlet_dist_distribution=(0.5, 2.0),
        cemented_curv_distribution=(0.0, 0.01),
        cemented_dist_distribution=(1.0,  1.0),
        ior_air=1.0,
        ior_glass=1.6,
        ior_glass_cement=1.7,
        max_semidiam=args.max_semidiam,
        min_distance=0.1,
        max_curvature=kmax,
        focal_length=args.focal_length,
        mutation_types=['glue'],
        no_mutation=args.no_mutation,
        use_projection=not args.no_projection,
        temperature=temperature,
        temperature_schedule=cooling_schedule,
        load_reservoir_folder=args.load_reservoir_folder,
        random_lens_initialization=args.random_lens_initialization,
        kchar=0.1,
        tchar=10.0,
        optimizer=args.optimizer,
        local_step_iterations=args.local_step_iters,
        fixed_local_iterations=args.fixed_iteration_count,
        expand_semidiam=args.expand_semidiam,
        zemax_reservoir_name=args.zemax_reservoir_name,
    )
    return params

