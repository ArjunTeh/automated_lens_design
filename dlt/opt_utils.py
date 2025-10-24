import numpy as np
import jax.numpy as jnp
import jax

from typing import Collection
from jaxtyping import Array

from . import zemax_loader
from . import aspheric
from . import curvature_sphere
from . import primary_sample_space as pss
from . import numerical_aperture
from . import sources
from . import system_matrix
from . import tracing
from . import zemax



def target_positions(nangles, target_focal_length):
    target_heights = [target_focal_length * jnp.tan(ang) for ang in nangles]
    return target_heights


def leggauss_source_chief_rays(lens, focal_length, angle_list, nrings, nangles, max_semidiam=None):
    ymax, premat, aper_idx = system_matrix.aperture_from_transfer_matrix(lens)
    appx_fpp = lens[:aper_idx + 1, 1].sum()
    if max_semidiam is None:
        max_semidiam = lens[0, 3]
    sensor_dist = lens[:, 1].sum()

    xs, vs = [], []
    ws = []
    targets, areas = [], []
    for angle in angle_list:
        cx, cv = system_matrix.chief_ray_from_pre_mat(angle, premat)
        pos, w = sources.leggauss_rings(max_semidiam, nrings, nangles)

        x = pos + cx
        v = cv + jnp.zeros_like(pos)
        xs.append(x)
        vs.append(v)
        ws.append(w)
        areas.append(jnp.ones_like(w) * jnp.pi * max_semidiam**2)

        target = jnp.tile(jnp.array([sensor_dist, -focal_length * jnp.tan(angle), 0.0]), (x.shape[0], 1))
        targets.append(target)

    xs = jnp.concatenate(xs, axis=0)
    vs = jnp.concatenate(vs, axis=0)
    ws = jnp.concatenate(ws, axis=0)
    targets = jnp.concatenate(targets, axis=0)
    areas = jnp.concatenate(areas, axis=0)
    L = jnp.ones(xs.shape[0])
    return xs, vs, L, targets, ws, areas


def create_source_chief_rays(lens, focal_length, angle_list, nrings=10, nangles=6, max_semidiam=None):
    """
    Create a source generator for chief rays using Legendre-Gauss quadrature.
    
    Parameters:
    - lens: The lens system to trace rays through.
    - focal_length: The focal length of the lens system.
    - angle_list: List of angles for which to generate chief rays.
    - nrings: Number of rings in the Legendre-Gauss quadrature.
    - nangles: Number of angles in the quadrature.
    
    Returns:
    A function that generates rays based on the specified parameters.
    """
    x, v, L, targets, w, areas = leggauss_source_chief_rays(lens, focal_length, angle_list, nrings, nangles, max_semidiam=max_semidiam)
    def generate_rays(nrays, width, height, key):
        return x, v, L, targets, w, areas

    return generate_rays


def create_source(line_range, focal_length, object_distance=jnp.inf, front_principal_plane=0.0, angle_list=[0.0], source_type='line', nrays=100, stochastic=False, key=jax.random.PRNGKey(0)):
    angles = angle_list
    if source_type == 'line':
        if object_distance == jnp.inf:
            if stochastic:
                line_generators = [sources.line_source_stratified_random(line_range, a) for a in angles]
            else:
                line_generators = [sources.line_source_quadrature(line_range, a) for a in angles]
        else:
            line_generators = [sources.point_source_quadrature(line_range, a, object_distance) for a in angles]
    elif source_type == 'leggauss':
        line_generators = [sources.leggaus_circular_quadrature(line_range, rings=nrays, angles=6, src_angle=a, fpp=front_principal_plane) for a in angles]
    elif source_type == 'rectangle':
        if stochastic:
            line_generators = [sources.square_source_random_stratified(2*line_range, 2*line_range, a, front_principal_plane) for a in angles]
        else:
            line_generators = [sources.square_source_quadrature(2*line_range, 2*line_range, a, front_principal_plane) for a in angles]
    else:
        raise ValueError(f"source type not recognized: {source_type}")

    source_weights = jnp.linspace(1, 0.3, len(angles))
    # source_weights = jnp.ones_like(source_weights)
    def generate_rays(nrays, width, height, key):
        rays = [lg(nrays//len(angles), width, height, key) for lg in line_generators]
        x = jnp.concatenate([r[0] for r in rays], axis=0)
        v = jnp.concatenate([r[1] for r in rays], axis=0)
        L = jnp.concatenate([r[2] for r in rays], axis=0)
        w = jnp.concatenate([s * r[3] for s, r in zip(source_weights, rays)], axis=0)
        area_of_sources = jnp.concatenate([r[4] for r in rays], axis=0)

        target_heights = target_positions(angles, target_focal_length=focal_length)
        targets = jnp.zeros(x.shape)
        for i, th in enumerate(target_heights):
            step = x.shape[0] // len(angles)
            targets = targets.at[i*step:(i+1)*step, 1].set(th)
            # targets[i*step:(i+1)*step, 1] = target_heights[i]

        return x, v, L, jnp.asarray(targets), w, area_of_sources

    return generate_rays


def create_angle_source(line_range, focal_length, object_distance=np.inf):
    line_generator = sources.line_source_stratified_random_angle(line_range)

    def generate_rays(nrays, angle, key):
        rays = line_generator(nrays, angle, key)
        target_height = target_positions([angle], target_focal_length=focal_length)[0]
        x, v, L, w, rad = rays
        targets = jnp.zeros(x.shape)
        targets = targets.at[:, 1].set(target_height)

        return x, v, L, targets, w, jnp.abs(rad)
    return generate_rays


def create_surface_function_list(surf_desc, asphere_type='standard', no_list=True):
    #TODO(ateh): add support for planes (sensors and stops)
    if no_list:
        if zemax_loader.ZemaxSurfaceType.EVENASPH in surf_desc:
            return aspheric.functionSuite(asphere_type=asphere_type)
        else:
            return curvature_sphere.functionSuite()
    
    fn_list = []
    for sd in surf_desc:
        if sd == zemax_loader.ZemaxSurfaceType.EVENASPH:
            fn_list.append(aspheric.functionSuite(asphere_type='standard'))
        elif sd == zemax_loader.ZemaxSurfaceType.STANDARD:
            fn_list.append(curvature_sphere.functionSuite())
        elif sd == zemax_loader.ZemaxSurfaceType.STOP:
            fn_list.append(curvature_sphere.functionSuite())

    # add sensor plane
    fn_list.append(curvature_sphere.functionSuite())
    return fn_list


def create_source_from_params(lens, **params):
    focal_length = params.get('focal_length', 20.0)
    angle_list = params.get('angle_list', [0.0])
    source_type = params.get('source_type', 'line')
    beam_size = params.get('beam_size', 1.0)
    line_range = params.get('max_semidiam', lens[0, 3]) * beam_size
    object_distance = params.get('object_distance', np.inf)
    stochastic = params.get('stochastic', False)
    fpp_appx = params.get('front_principal_plane', 0.0)
    nrays = params.get('nrays', 100)
    ray_generator = create_source(line_range, 
                                  angle_list=angle_list, 
                                  focal_length=focal_length, 
                                  object_distance=object_distance, 
                                  front_principal_plane=fpp_appx,
                                  source_type=source_type, 
                                  nrays=nrays,
                                  stochastic=stochastic)
    return ray_generator


def create_2d_tracing_loss(lens : pss.Lens, blur_loss_fn, **params):
    nrays = params.get('nrays', 100)
    loss_fn = create_2d_tracing_loss_rays(**params)

    beam_size = params.get('beam_size', 1.0)
    max_semidiam = params.get('max_semidiam', lens.data[0, 3])
    params.setdefault('source_type', 'rectangle')

    if params['source_type'] == 'leggauss':
        generate_rays = create_source_chief_rays(lens.toarray(), params['focal_length'], params['angle_list'], nrings=nrays, nangles=nrays//2, max_semidiam=max_semidiam)
    else:
        generate_rays = create_source_from_params(lens.toarray(), **params)

    def loss(state, na_exp, nsurfaces, rng_key):
        key, subkey = jax.random.split(rng_key)
        line_range = 2 * max_semidiam * beam_size
        rays = generate_rays(nrays, line_range, line_range, key=subkey)
        return loss_fn(state, na_exp, rays, nsurfaces)

    return loss

def create_2d_tracing_loss_rays(**params):
    focal_length = params.get('focal_length', 20.0)
    angle_list = params.get('angle_list', [0.0])
    object_distance = params.get('object_distance', np.inf)
    surface_type_list = params.get('surface_type_list', [])
    nrays = params.get('nrays', 100)
    return_path = params.get('return_path', False)
    additive_loss = params.get('additive_loss', True)
    use_focal_loss = params.get('focal_loss', False)
    focal_loss_weight = params.get('focal_loss_weight', 1.0)
    biased = params.get('biased', False)
    use_leggauss = params.get('use_leggauss', False)


    surf_funs = create_surface_function_list(surface_type_list)
    if biased:
        vtracer = jax.vmap(tracing.trace_scan_no_warp, (None, None, 0, 0, 0, None, None))
        stop_idx = 0

    elif zemax_loader.ZemaxSurfaceType.STOP in surface_type_list:
        stop_idx = surface_type_list.index(zemax_loader.ZemaxSurfaceType.STOP)
        vtracer = jax.vmap(tracing.trace_scan_with_stop_check, (None, None, 0, 0, 0, None, None))
    else:
        stop_idx = 0
        vtracer = jax.vmap(tracing.trace_scan_no_stop, (None, None, 0, 0, 0, None, None))

    def loss(state, na_exp, rays : Collection[Array], nsurfaces):
        surfs = zemax.zemax_spherical_to_jax_array(state, focal_length=focal_length, object_distance=object_distance)

        x, v, L, targets, weights, source_areas = rays
        (valid, xp, vp, Lp), (Tx, wargmin) = vtracer(surf_funs, surfs, x, v, L, stop_idx, nsurfaces)
        xp = xp.swapaxes(0, 1)
        Lp = Lp.swapaxes(0, 1)
        vp = vp.swapaxes(0, 1)

        xp_valid = jnp.where(valid[:, None], xp[nsurfaces+1], 0.0)
        targets = jnp.where(valid[:, None], targets, 0.0)

        if not use_leggauss:
            weights = jnp.ones_like(weights)

        valid = jax.lax.stop_gradient(valid)
        change_of_meas : Array = jnp.where(valid, jnp.linalg.det(Tx), 0.0) # type:ignore
        weights_sum = weights.sum()
        weighted_change_of_meas = change_of_meas * weights / weights.sum()

        throughput_integrand = source_areas * weighted_change_of_meas

        throughput = jnp.sum(throughput_integrand)

        spot_thrus = []
        spot_errors = []
        spot_centers = []
        nrays = x.shape[0]
        for ang in range(len(angle_list)):
            idx_slice = slice(ang*nrays//len(angle_list), (ang+1)*nrays//len(angle_list))
            source_area = source_areas[idx_slice].mean()
            ray_view = xp_valid[idx_slice, :]
            spot_thruput = jnp.where(jnp.any(valid[idx_slice]), jnp.sum(weighted_change_of_meas[idx_slice]), 1.0)
            # spot_thruput = jnp.where(jnp.any(valid[idx_slice]), jnp.sum(throughput_integrand[idx_slice]), 1.0)
            spot_center = jnp.sum(ray_view * weighted_change_of_meas[idx_slice][:, None], axis=0, keepdims=True) / jnp.where(jnp.any(valid[idx_slice]), spot_thruput, 1.0)
            spot_error_ray = weighted_change_of_meas[idx_slice] * ((ray_view[:, 1:] - spot_center[:, 1:])**2).sum(axis=-1)
            spot_error = source_area * jnp.sum(jnp.where(valid[idx_slice], spot_error_ray, 0.0)) / nrays
            spot_errors.append(spot_error)
            spot_centers.append(spot_center)
            spot_thrus.append(spot_thruput * source_area)
        spot_error = sum(spot_errors)

        if additive_loss:
            final_loss = spot_error - na_exp*throughput
        else:
            final_loss = spot_error / (throughput**na_exp)

        focal_loss = 0.0
        focal_losses = []
        if use_focal_loss:
            # ignore the zero angle for focal loss, since it is unstable?
            for ang, spot_center in zip(angle_list[1:], spot_centers[1:]):
                spot_dist2 = jnp.sum(jnp.square(spot_center[:, 1:]))
                focal_dist = jnp.tan(ang) * focal_length
                focal_ang_loss = 0.5 * jnp.sum(jnp.square(spot_dist2 - focal_dist**2))
                focal_losses.append(focal_ang_loss)
                final_loss += focal_loss_weight * focal_loss
            focal_loss = sum(focal_losses) / jnp.maximum(len(focal_losses), 1)

        aux_dict = dict(
            throughput=throughput,
            spot_centers=spot_centers,
            spot_thrus=spot_thrus,
            spot_error=spot_error,
            spot_errors=spot_errors,
            focal_loss=focal_loss,
            focal_losses=focal_losses,
            valid=valid
        )
    
        if return_path:
            aux_dict['path'] = xp

        return final_loss, aux_dict

    return loss


def spot_error(valid, xp, vp, Lp, radius, change_of_meas=1.0):
    '''calculate spot error for multiple views based on nangs'''
    ray_view = xp[valid]
    spot_thruput = jnp.where(jnp.any(valid), jnp.sum(radius * change_of_meas), 1.0)
    spot_center = jnp.sum(ray_view * (radius * change_of_meas)[:, None], axis=0, keepdims=True) / jnp.where(jnp.any(valid), spot_thruput, 1.0)
    spot_error_ray = 2 * np.pi * radius * change_of_meas *((ray_view[:, 1:] - spot_center[:, 1:])**2).sum(axis=-1)
    spot_error = jnp.sum(jnp.where(valid, spot_error_ray, 0.0)) / valid.shape[0]
    return spot_error, spot_center, spot_thruput


def list_of_dict_transpose(list_of_dict):
    return {k: np.stack([d[k] for d in list_of_dict]) for k in list_of_dict[0].keys()}


def lens_f_number(lens, focal_length):
    boundary, aux = numerical_aperture.calc_zemax_na(lens)
    diameter = 2 * boundary
    return jnp.abs(focal_length / diameter)


@jax.jit
def approximate_focal_length(lens):
    mat, mat_list, _ = system_matrix.ray_transfer_matrix_list(lens)

    mat = jnp.eye(2)
    for m in mat_list[:-1]:
        mat = jnp.matmul(m, mat)

    return 1 / mat[0, 1]


@jax.jit
def approximate_f_number(lens):
    focal_length = approximate_focal_length(lens)
    return lens_f_number(lens, focal_length)


def export_lens(fname, lens, lens_desc_generic=None, wavelengths=None):

    if lens_desc_generic is None:
        lens_type_list = ['STANDARD']*lens.shape[0]
    else:
        lens_type_list = [s['surf_type'] for s in lens_desc_generic[1:-1]]

    if wavelengths is None:
        wavelengths = [486.1, 587.56, 656.3]

    iors = [
        [s['ior_fn'](w) 
         for s in lens_desc_generic[1:-1]]
        for w in wavelengths
    ]
    iors = np.array(iors)
    
    right_vertex = lens[:, 1].sum()
    left_vertex = 0

    mat, mat_list, _ = system_matrix.ray_transfer_matrix_list(lens)

    focal_length = 1 / mat[0, 1]
    back_plane_offset = focal_length * (mat[1, 1] - 1)
    front_plane_offset = focal_length * (1 - mat[0, 0])

    dict_out = dict(
        lens=lens,
        iors=iors,
        wavelengths=wavelengths,
        effective_focal_length=focal_length,
        back_plane=right_vertex + back_plane_offset,
        front_plane=left_vertex + front_plane_offset,
        lens_desc=np.array(lens_type_list)
    )

    np.savez(fname, **dict_out)