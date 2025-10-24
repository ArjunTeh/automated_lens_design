import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax

import configargparse

from jaxtyping import Key, Float
from typing import Callable, Tuple, Dict, Union

import time
from tqdm import tqdm

from dlt import opt_utils
from dlt import system_matrix
from dlt import optical_properties
from dlt import primary_sample_space as pss
from dlt import zemax_loader


def load_lens_file(filename: str, max_elements) -> pss.Lens:
    max_surfaces = max_elements * 3
    lens_data, lens_type_list = zemax_loader.load_zemax_file(filename, info_list=True)
    lens_data, lens_type_list = zemax_loader.sanitize_zemax_lens(lens_data, lens_type_list)

    if lens_data.shape[0] > max_surfaces:
        raise ValueError(f'Lens has too many surfaces: {lens_data.shape[0]} > {max_surfaces}')

    nsurfaces = lens_data.shape[0]
    lens_data = pss.homogenize_lens(jnp.array(lens_data), max_surfaces)
    lens = pss.Lens(data=lens_data, nsurfaces=nsurfaces)
    return lens


def load_config_parser():
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')
    parser.add_argument('--out_folder', type=str, default='results/restore_reservoir_test', help='output folder')
    parser.add_argument('--spot_weight', type=float, default=1.0, help='weight for spot size')
    parser.add_argument('--thru_weight', type=float, default=1.0, help='weight for throughput')
    parser.add_argument('--thick_weight', type=float, default=1.0, help='weight for thickness penalty')
    parser.add_argument('--track_weight', type=float, default=0.0, help='weight for track length')
    parser.add_argument('--volume_weight', type=float, default=0.0, help='weight for volume')
    parser.add_argument('--nelements_weight', type=float, default=0.0, help='weight for number of elements')
    parser.add_argument('--focal_weight', type=float, default=0.001, help='weight for focal loss')
    parser.add_argument('--focal_length', type=float, default=50.0, help='focal length of the lens')
    parser.add_argument('--sensor_half_width', type=float, default=21.0, help='half width of the sensor')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature for the objective')
    parser.add_argument('--nrays', type=int, default=50, help='number of rays to use')
    parser.add_argument('--use_leggauss', action='store_true', help='use the leggauss weights for the different rays')
    parser.add_argument('--source_type', type=str, default='rectangle', help='shape of the source beam')
    parser.add_argument('--niters', type=int, default=200, help='number of iterations')
    parser.add_argument('--optimizer', type=str, default='lbfgs', help='optimizer to use for local step')
    parser.add_argument('--local_step_iters', type=int, default=2, help='number of iterations for each local step')
    parser.add_argument('--post_process_iters', type=int, default=20, help='number of iterations for post processing')
    parser.add_argument('--gradient_eps', type=float, default=1e-4, help='epsilon for the optimizer')
    parser.add_argument('--noise_eps', type=float, default=0.0, help='epsilon for the noise')
    parser.add_argument('--tour_lifetime', type=float, default=8.0, help='expected tour regeneration rate')
    parser.add_argument('--noise_seed', type=int, default=1, help='seed for the added noise')
    parser.add_argument('--render_seed', type=int, default=1, help='seed for the rendering')
    parser.add_argument('--split_prob', type=float, default=0.5, help='split probability')
    parser.add_argument('--expand_semidiam', action='store_true', help='expand the semi-diameter of the lens based on curvature')
    parser.add_argument('--max_semidiam', type=float, default=25.0, help='maximum semi-diameter')
    parser.add_argument('--max_curvature', type=float, default=0.08, help='maximum semi-diameter')
    parser.add_argument('--max_elements', type=int, default=7, help='maximum number of elements')
    parser.add_argument('--min_elements', type=int, default=2, help='minimum number of elements')
    parser.add_argument('--min_thickness', type=float, default=1.6, help='minimum allowable thickness of glass element')
    parser.add_argument('--max_thickness', type=float, default=100.0, help='maximum allowable thickness of glass element')
    parser.add_argument('--max_distance', type=float, default=100.0, help='maximum allowable distance between elements')
    parser.add_argument('--target_max_elements', type=int, default=5, help='maximum number of elements before nelements penalty')
    parser.add_argument('--volume_normalization', type=float, default=15000.0, help='target volume for lens')
    parser.add_argument('--spot_normalization', type=float, default=0.001, help='the minimum acceptable spot size')
    parser.add_argument('--track_normalization', type=float, default=90.0, help='baseline track length')
    parser.add_argument('--thru_normalization_strategy', type=str, default='max', help='how to calculate the throughput normalization (max or init thru)')
    parser.add_argument('--no_mutation', action='store_true', help='do not use mutations during regeneration')
    parser.add_argument('--no_projection', action='store_true', help='do not use projection during regeneration')
    parser.add_argument('--reservoir_size', type=int, default=5, help='size of the reservoir')
    parser.add_argument('--reservoir_greedy_sample', action='store_true', help='use greedy sampling for the reservoir')
    parser.add_argument('--discrete_measure_dominance', type=float, default=0.1, help='dominance threshold for sampling the reservoir, (negative value means no discrete sampling)')
    parser.add_argument('--load_reservoir_folder', type=str, default=None, help='folder to load the reservoir from')
    parser.add_argument('--fixed_iteration_count', type=int, default=None, help='fixed iteration count for the optimization instead of restore')
    parser.add_argument('--random_lens_initialization', action='store_true', help='use random lens initialization for reservoir')
    parser.add_argument('--zemax_reservoir_name', type=str, default=None, help='zemax fils to load the reservoir from')
    parser.add_argument('--save_reservoir_hist', action='store_true', help='save the reservoir history')
    return parser


def load_config(config_file):
    '''From a config file return the args that config parser would return'''
    parser = load_config_parser()
    args = parser.parse_args(['--config', config_file])
    return args


def create_objective_function_from_args(args, lens=None):
    max_elements = args.max_elements
    nsurfaces = max_elements * 3

    if lens is None:
        lens = pss.Lens(nsurfaces=1, data=jnp.ones((nsurfaces, 4)))

    obj_weights = dict(
        spot=args.spot_weight,
        thru=args.thru_weight,
        thick=args.thick_weight,
        track=args.track_weight,
        volume=args.volume_weight,
        nelements=args.nelements_weight,
        focal=args.focal_weight
    )

    ymax, premat, aper_idx = system_matrix.aperture_from_transfer_matrix(lens.toarray())
    appx_fpp = lens.data[:aper_idx + 1, 1].sum()

    if args.thru_normalization_strategy == 'max':
        thru_normalization = jnp.pi * args.max_semidiam**2
    elif args.thru_normalization_strategy == 'init':
        thru_normalization = 2.0 * jnp.pi * ymax**2
    else:
        raise ValueError(f'Invalid throughput normalization strategy: {args.thru_normalization_strategy}')

    focal_length = args.focal_length
    lens_type_list = [zemax_loader.ZemaxSurfaceType.STANDARD] * nsurfaces

    target_sensor_points = jnp.linspace(0, 1, 4) * args.sensor_half_width

    kwargs = dict(
        volume_normalization=args.volume_normalization, 
        spot_normalization=args.spot_normalization,
        track_normalization=args.track_normalization,
        thru_normalization=thru_normalization,
        max_semidiam=args.max_semidiam,
        focal_length=focal_length, 
        target_min_thickness=args.min_thickness,
        target_max_thickness=args.max_thickness,
        target_max_distance=args.max_distance,
        target_max_elements=args.target_max_elements,
        beam_size=1.0,
        nrays=args.nrays,
        front_principal_plane=appx_fpp,
        gaussian_weights=args.use_leggauss,
        target_sensor_points=target_sensor_points,
        source_type=args.source_type,
    )

    obj_fun = create_objective(lens, lens_type_list, obj_weights, **kwargs)
    return obj_fun


def create_trace_function(lens : pss.Lens, lens_type_list, focal_length, **kwargs):
    # should generate a jit function that is able to calculate the gradient and loss of a lens with different number of lenses
    nrays = kwargs.get('nrays', 100)
    beam_size = kwargs.get('beam_size', 1.0)
    target_sensor_points = kwargs.get('target_sensor_points', [0.0, 8.0, 12.0])
    angle_list = [jnp.atan2(a, focal_length) for a in target_sensor_points]
    biased = kwargs.get('biased', False)
    max_semidiam = kwargs.get('max_semidiam', lens.data[0, 3])
    fpp_appx = kwargs.get('front_principal_plane', 0.0) # default is the front of the lens
    use_leggauss = kwargs.get('gaussian_weights', False)
    source_type = kwargs.get('source_type', 'rectangle')

    loss_params = dict(
        loss_type='spot_error_unbiased',
        focal_loss=True,
        focal_loss_weight=0.0,
        focal_length=focal_length,
        nrays=nrays,
        angle_list=angle_list,
        asphere=False,
        object_distance=jnp.inf,
        stochastic=True,
        log_loss=False,
        use_leggauss=use_leggauss,
        beam_size=beam_size,
        max_semidiam=max_semidiam,
        surface_type_list=lens_type_list,
        biased=biased,
        front_principal_plane=fpp_appx,
        source_type=source_type,
    )


    def l2_loss_fn(x):
        return jnp.sum(x[:, 1:]**2, axis=-1)
    
    trace_fun = opt_utils.create_2d_tracing_loss(lens, l2_loss_fn, **loss_params)

    def loss_fun(lens : pss.Lens, k) -> Tuple[Float, Float, Float, Dict]:
        lam_thru = 0.0
        n = lens.nsurfaces
        lens_dat = lens.data
        _, aux = trace_fun(lens_dat, lam_thru, n, k)

        spot = aux['spot_error']
        thru = aux['throughput']
        focal = aux['focal_loss']

        return spot, thru, focal, aux

    return loss_fun


def material_cost(lens : pss.Lens):
    '''Calculate the total volume of the lens -> this is a metric that can be messed with to penalize complicated designs'''
    #TODO: might consider using something like track length or something else
    mask = jnp.arange(lens.data.shape[0]-1) < lens.nsurfaces

    # the total volume of glass is proportional to cost
    volsum = optical_properties.glass_appx_volumes(lens.data).sum(where=mask)

    # total_cost = jnp.cbrt(volsum)
    total_cost = volsum
    return total_cost


def lens_thickness_penalty(lens : pss.Lens, target_min_thickness : Float, max_thickness : Float):
    # really thin elements are bad as well
    # really thick elements are also bad
    mask = jnp.arange(lens.data.shape[0]-1) < lens.nsurfaces
    glass = (lens.data[:-1, 2] > lens.data[-1, 2]) | (lens.data[:-1, 2] > 1.001)
    glass_mask = glass & mask

    d = lens.data[:-1, 1]
    d_cost : Float = jnp.where(d < target_min_thickness, (target_min_thickness - d)**2, 0.0)
    max_cost : Float = jnp.where(d > max_thickness, (d - max_thickness)**2, 0.0)
    dist_cost = d_cost.sum(where=glass_mask)
    max_cost = max_cost.sum(where=glass_mask)
    return dist_cost + max_cost


def lens_distance_penalty(lens : pss.Lens, target_max_distance : Float):
    # really thin elements are bad as well
    # really thick elements are also bad
    mask = jnp.arange(lens.data.shape[0]-1) < lens.nsurfaces
    glass = (lens.data[:-1, 2] > lens.data[-1, 2]) | (lens.data[:-1, 2] > 1.001)
    air_mask = (~glass) & mask

    d = lens.data[:-1, 0]
    dist_cost : Float = jnp.where(d > target_max_distance, (d - target_max_distance)**2, 0.0)
    dist_cost = dist_cost.sum(where=air_mask)
    return dist_cost


def nelements_penalty(lens : pss.Lens, target_max_elements):
    '''The number of glass elements in the design, this isn't differentiable'''
    # really thin elements are bad as well
    mask = jnp.arange(lens.data.shape[0]-1) < lens.nsurfaces
    glass = (lens.data[:-1, 2] > lens.data[-1, 2]) | (lens.data[:-1, 2] > 1.001)
    glass_mask = glass & mask

    n = jnp.sum(glass_mask)
    val = jnp.where(n > target_max_elements, (n - target_max_elements)**2, 0.0)
    return val



def create_objective(lens : pss.Lens, lens_type_list, objective_weights, focal_length, **kwargs):
    trace_fun = create_trace_function(lens, lens_type_list, focal_length, **kwargs)
    lam_spot = objective_weights['spot']
    lam_thru = objective_weights['thru']
    lam_focal = objective_weights['focal']
    lam_track = objective_weights['track']
    lam_volume = objective_weights['volume']
    lam_thick = objective_weights['thick']
    lam_nelements = objective_weights['nelements']

    # normalize the lam weights
    # don't normalize the penalty weights
    lam_sum = lam_spot + lam_thru + lam_focal + lam_track + lam_volume# + lam_thick + lam_nelements
    lam_spot /= lam_sum
    lam_thru /= lam_sum
    lam_focal /= lam_sum
    lam_track /= lam_sum
    lam_volume /= lam_sum
    # lam_thick /= lam_sum
    # lam_nelements /= lam_sum

    max_semidiam = kwargs['max_semidiam']
    volume_normalization = kwargs['volume_normalization']
    spot_normalization = kwargs['spot_normalization']
    track_normalization = kwargs['track_normalization']
    thru_normalization = kwargs['thru_normalization']
    min_thickness = kwargs['target_min_thickness']
    max_thickness = kwargs['target_max_thickness']
    max_distance = kwargs['target_max_distance']
    target_max_elements = kwargs['target_max_elements']

    track_length_appx = track_normalization
    T0_appx = thru_normalization
    V0_appx = volume_normalization
    spot0_appx = spot_normalization # what we consider the best spot size given the version of the 

    def objective(nlens : pss.NormalizedLens, k):
        lens = pss.normalized2lens(nlens)
        spot, thru, focal, aux = trace_fun(lens, k)
        spot_safe : Float = jnp.where(thru < 1e-6, 1e6, spot)
        focal_safe : Float = jnp.where(thru < 1e-6, 1e6, focal)
        thru_safe : Float = jnp.maximum(thru, 1e-6)
        volume = material_cost(lens)
        track_length = jnp.sum(lens.data[:, 1], where=jnp.arange(lens.data.shape[0]) < lens.nsurfaces)
        thickness_penalty = (
            lens_thickness_penalty(lens, target_min_thickness=min_thickness, max_thickness=max_thickness) +
            lens_distance_penalty(lens, target_max_distance=max_distance)
        )
        nelem = nelements_penalty(lens, target_max_elements=target_max_elements)

        scaled_spot = lam_spot * (spot_safe / spot0_appx)
        # scaled_thru = lam_thru * (- jnp.log(thru_safe / T0_appx))
        scaled_thru = lam_thru * (1 - thru / T0_appx)
        scaled_thick = lam_thick * thickness_penalty
        scaled_volume = lam_volume * (volume / V0_appx)
        scaled_focal = lam_focal * focal_safe
        scaled_track = lam_track * (track_length / track_length_appx)
        scaled_nelements = lam_nelements * nelem

        loss = (
            scaled_spot + 
            scaled_thru + 
            scaled_thick + 
            scaled_volume +
            scaled_focal +
            scaled_nelements +
            scaled_track
        )

        aux['spot'] = spot
        aux['thru'] = thru
        aux['focal'] = focal
        aux['track'] = track_length
        aux['thick'] = thickness_penalty
        aux['volume'] = volume
        aux['nelements'] = nelem
        aux['thickness_penalty'] = thickness_penalty
        aux['scaled_spot'] = scaled_spot
        aux['scaled_thru'] = scaled_thru
        aux['scaled_thick'] = scaled_thick
        aux['scaled_volume'] = scaled_volume
        aux['scaled_focal'] = scaled_focal
        aux['scaled_track'] = scaled_track
        aux['scaled_nelements'] = scaled_nelements
        aux['loss'] = loss
        return loss, aux

    return objective


@eqx.filter_jit
def grad_project(lens : pss.LensType, v : pss.LensType, expand_semidiam) -> pss.LensType:
    projv = v.data
    projv = projv.at[lens.nsurfaces, 0].set(0.0)
    projv = projv.at[:, 2].set(0.0)
    projv = jnp.where(expand_semidiam, projv.at[:, 3].set(0.0), projv) # don't change semidiameter if it is being determined by the curvature
    return v._replace(data=projv)


@eqx.filter_jit
def retract_lens(lens : pss.Lens, max_curvature, min_distance, max_semidiam):
    '''Retract the lens to the feasible set'''
    lens_data = lens.data
    lens_data = lens_data.at[:, 0].set(jnp.clip(lens_data[:, 0], -max_curvature, max_curvature))
    lens_data = lens_data.at[:, 1].set(jnp.maximum(lens_data[:, 1], min_distance))
    lens_data = lens_data.at[:, 2].set(jnp.maximum(lens_data[:, 2], 1.0))
    lens_data = lens_data.at[:, 3].set(max_semidiam)
    lens_data = optical_properties.clip_semidiameter_to_curvature(lens_data)
    return lens._replace(data=lens_data)


@jax.jit
def retract_normal_lens(normlens : pss.NormalizedLens, max_curvature, min_distance, max_semidiam, expand_semidiam):
    '''Retract the lens to the feasible set'''
    lens = pss.normalized2lens(normlens)
    lens_data = lens.data
    lens_data = lens_data.at[:, 0].set(jnp.clip(lens_data[:, 0], -max_curvature, max_curvature))
    lens_data = lens_data.at[:, 1].set(jnp.maximum(lens_data[:, 1], min_distance))
    lens_data = lens_data.at[:, 2].set(jnp.maximum(lens_data[:, 2], 1.0))
    lens_data = jnp.where(expand_semidiam, lens_data.at[:, 3].set(max_semidiam), lens_data)
    lens_data = optical_properties.clip_semidiameter_to_curvature(lens_data)
    new_lens = lens._replace(data=lens_data)
    return pss.lens2normalized(new_lens, normlens.characteristic_curvature, normlens.characteristic_thickness)


def optimize_lens_optax(loss_and_grad : Callable, init_lens : pss.NormalizedLens, niters : int, epsilon : float, key : Key, params, obj_fun : Callable, optimizer='adam', fixed_key=False):
    '''Optimize a lens using optax'''
    opt_step, opt = create_optax_step(loss_and_grad, epsilon, params, obj_fun, optimizer=optimizer)
    init_opt_state = opt.init(init_lens.data)
    return optimize_lens_with_optax_step(opt_step, init_lens, init_opt_state, niters, key, params, fixed_key=fixed_key)


def optimize_lens_with_optax_step(opt_step : Callable, init_lens : pss.NormalizedLens, init_opt_state, niters : int, key : Key, params, fixed_key=False):
    '''Optimize a lens using optax'''
    lens = init_lens
    # eps_schedule = optax.exponential_decay(init_value=1.0, transition_steps=niters//2, decay_rate=0.8, end_value=0.001)
    # opt = optax.polyak_sgd(epsilon, scaling=eps_schedule)
    opt_state = init_opt_state

    ray_loss = 10000.0
    ray_losses = []
    lens_hist : list[pss.NormalizedLens] = [lens]
    times = []

    lens = retract_normal_lens(normlens=lens, 
                               max_curvature=params['max_curvature'], 
                               min_distance=params['min_distance'], 
                               max_semidiam=params['max_semidiam'],
                               expand_semidiam=params['expand_semidiam'])
    init_time = time.time()
    for i in tqdm(range(niters)):
        if not fixed_key:
            key, subkey = jr.split(key)

        lens, opt_state, ray_loss, ray_grad, ray_aux = opt_step(lens, opt_state, key)
        ray_losses.append(((ray_loss, ray_aux), ray_grad))
        lens_hist.append(lens)
        times.append(time.time() - init_time)

        if jnp.any(lens.data[:, 1] < 0.0):
            raise ValueError('Negative distance')

    return lens_hist, ray_losses, jnp.array(times)


def create_optax_step(loss_and_grad : Callable, epsilon, params, obj_fun : Callable, optimizer='adam'):
    '''Optimize a lens using optax'''
    # eps_schedule = optax.exponential_decay(init_value=1.0, transition_steps=niters//2, decay_rate=0.8, end_value=0.001)
    # opt = optax.polyak_sgd(epsilon, scaling=eps_schedule)
    if optimizer == 'adam':
        opt = optax.adam(epsilon)
    elif optimizer == 'lbfgs':
        # opt = optax.lbfgs(epsilon, linesearch=optax.scale_by_backtracking_linesearch(15))
        opt = optax.lbfgs(epsilon)
    elif optimizer == 'adam_linesearch':
        opt = optax.chain(
            optax.adam(epsilon),
            optax.scale_by_zoom_linesearch(10)
        )
    else:
        raise ValueError(f'Invalid optimizer: {optimizer}')

    @eqx.filter_jit
    def opt_step(lens, opt_state, key):
        (new_ray_loss, ray_aux), ray_grad = loss_and_grad(lens, key)
        ray_grad = grad_project(lens, ray_grad, params['expand_semidiam'])

        def val_fun(xdat):
            lens_update = lens._replace(data=xdat)
            return obj_fun(lens_update, key)[0]
        if optimizer == 'lbfgs' or optimizer == 'adam_linesearch':
            updates, opt_state = opt.update(
                ray_grad.data, opt_state, lens.data, 
                value=new_ray_loss, grad=ray_grad.data, value_fn=val_fun) #type:ignore
        elif optimizer == 'adam':
            updates, opt_state = opt.update(ray_grad.data, opt_state, lens.data)
        else:
            raise ValueError(f'Invalid optimizer: {optimizer}')

        new_lens_dat = optax.apply_updates(lens.data, updates)
        lens = lens._replace(data=new_lens_dat)
        lens = retract_normal_lens(lens, max_curvature=params['max_curvature'], min_distance=params['min_distance'], max_semidiam=params['max_semidiam'], expand_semidiam=params['expand_semidiam'])
        return lens, opt_state, new_ray_loss, ray_grad, ray_aux
    
    return opt_step, opt