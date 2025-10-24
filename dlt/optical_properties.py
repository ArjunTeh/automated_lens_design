import jax
import jax.numpy as jnp

from jaxtyping import Float

from . import zemax_loader
from . import aspheric
from . import system_matrix
from . import curvature_sphere
from . import sources
from . import zemax
from . import tracing
from . import constants


def load_state(lens_desc, wavelength):
    return zemax_loader.desc_state_to_array(lens_desc, wavelength=wavelength)


def trace_lens(lens_desc, wavelength=587.6, reverse=False, beam_size=1.0):
    lens = load_state(lens_desc, wavelength=wavelength)[0][:, :4]
    return trace_state(lens, wavelength=wavelength, reverse=reverse, beam_size=beam_size)


def trace_state(lens, wavelength=587.6, reverse=False, beam_size=1.0, asphere=False, nrays=20, angle : Float=0.0):
    raygen = sources.line_source_quadrature(beam_size * lens[:, 3].max(), angle)
    x, v, L, *ray_aux = raygen(nrays, None, None, None)

    surfs = zemax.zemax_state_to_jax_surfaces(lens)
    if asphere:
        surf_funs = [aspheric.functionSuite('standard')] * len(surfs)
    else:
        surf_funs = [curvature_sphere.functionSuite()] * len(surfs)

    if reverse:
        surf_funs = surf_funs[::-1]
        surfs = surfs[::-1]
        x = x.at[:, 0].set(lens[:, 1].sum() + 5.0)
        v = -v
    surfs = jnp.array(surfs)

    vtracer = jax.vmap(tracing.trace_scan_no_warp, (None, None, 0, 0, 0, None, None))
    (valid, xp, vp, Lp), (Tx, wargmin) = vtracer(surf_funs[0], surfs, x, v, L, 0, lens.shape[0])

    return valid, xp, vp, Lp


def trace_state_rays(lens, x, v, L, reverse=False):
    '''NOTE: Only supports curvature sphere'''
    surfs = zemax.zemax_state_to_jax_surfaces(lens)
    surf_funs = [curvature_sphere.functionSuite()] * len(surfs)

    if reverse:
        surf_funs = surf_funs[::-1]
        surfs = surfs[::-1]

    surfs = jnp.array(surfs)
    vtracer = jax.vmap(tracing.trace_scan_no_warp, (None, None, 0, 0, 0, None, None))
    (valid, xp, vp, Lp), (Tx, wargmin) = vtracer(surf_funs[0], surfs, x, v, L, 0, lens.shape[0])
    return valid, xp, vp, Lp


def effective_focal_length(lens_desc, wavelength=587.6):
    ivalid, img_xp, img_vp, img_Lp = trace_lens(lens_desc, beam_size=0.1, wavelength=wavelength, reverse=True)
    ovalid, obj_xp, obj_vp, obj_Lp = trace_lens(lens_desc, beam_size=0.1, wavelength=wavelength, reverse=False)

    img_plane_samples = (obj_xp[0][ovalid, 1] / obj_vp[-1][ovalid, 1]) * (-obj_vp[-1][ovalid, 0])
    obj_plane_samples = (img_xp[0][ivalid, 1] / img_vp[-1][ivalid, 1]) * (img_vp[-1][ivalid, 0])


    effective_back_focal_length = img_plane_samples.mean()
    effective_front_focal_length = obj_plane_samples.mean()
    return effective_front_focal_length, effective_back_focal_length


def effective_focal_length_from_state(lens_state, wavelength=587.6, asphere=False):
    ivalid, img_xp, img_vp, img_Lp = trace_state(lens_state, beam_size=0.1, wavelength=wavelength, reverse=True, asphere=asphere)
    ovalid, obj_xp, obj_vp, obj_Lp = trace_state(lens_state, beam_size=0.1, wavelength=wavelength, reverse=False, asphere=asphere)

    img_plane_samples = (obj_xp[0][:, 1] / obj_vp[-1][:, 1]) * (-obj_vp[-1][:, 0])
    obj_plane_samples = (img_xp[0][:, 1] / img_vp[-1][:, 1]) * (img_vp[-1][:, 0])

    effective_back_focal_length = jnp.sum(jnp.where(ovalid, img_plane_samples, 0.0)) / jnp.maximum(jnp.count_nonzero(ovalid), 1.0)
    effective_front_focal_length = jnp.sum(jnp.where(ivalid, obj_plane_samples, 0.0)) / jnp.maximum(jnp.count_nonzero(ivalid), 1.0)
    return effective_front_focal_length, effective_back_focal_length


def effective_focal_length_zoom_state(lens_state, surf_ids, distances, wavelength=587.6, asphere=False):
    efls = []
    ebls = []
    for d in distances:
        lens_state = lens_state.at[surf_ids, 1].set(d)
        efl, ebl = effective_focal_length_from_state(lens_state, wavelength=wavelength, asphere=asphere)
        efls.append(efl)
        ebls.append(ebl)
    return efls, ebls


def effective_focal_length_magnification_from_state(lens_state, wavelength=587.6):
    # m = x / f where m is magnification, x is extension from image plane, f is focal length

    od = 1000
    extension = (od * 50) / (od - 50) - 50
    # extension = 2.63 # about focus for 1 meter away
    lens_modded = jnp.array(lens_state)
    lens_modded = lens_modded.at[-1, 1].add(extension)
    surfs = zemax.zemax_state_to_jax_surfaces(lens_modded)

    surf_funs = [curvature_sphere.functionSuite()] * len(surfs)

    sample_height = 0.01
    ray_samples = 10
    sample_heights = jnp.linspace(0, sample_height, ray_samples+1)[1:]
    x = jnp.zeros((ray_samples, 3)).at[:, 0].set(-5.0)
    x = x.at[:, 1].set(sample_heights)
    v = jnp.zeros_like(x).at[:, 0].set(1.0)
    L = jnp.ones((ray_samples,))

    vtracer = jax.vmap(tracing.trace_scan_no_stop, (None, None, 0, 0, 0, None))
    (valid, xp, vp, Lp), (Tx, wargmin) = vtracer(surf_funs[0], surfs, x, v, L, 0, lens_state.shape[0])
    magnification = -xp[:, -1, 1] / sample_heights

    efl = extension / magnification
    return efl.mean()


def front_principal_plane(lens, nrays=10):
    rev_valid, rev_xp, rev_vp, rev_Lp = trace_state(lens, reverse=True, beam_size=0.5, asphere=False, nrays=nrays, angle=0.0)

    # guess that the aperture size is the size of the first element
    aperture_rad = lens[0, 3]

    # find where all of these rays intersect the optical axis
    oa_intersect = rev_xp[:, -1, 0] + ((rev_xp[:, 0, 1]-rev_xp[:, -1, 1]) / rev_vp[:, -1, 1]) * (rev_vp[:, -1, 0])
    front_principal_plane = oa_intersect[rev_valid].mean()
    return front_principal_plane


def chief_ray_by_trace(lens, angle, nrays):
    rev_valid, rev_xp, rev_vp, rev_Lp = trace_state(lens, reverse=True, beam_size=0.5, asphere=False, nrays=nrays, angle=0.0)

    # guess that the aperture size is the size of the first element
    aperture_rad = lens[0, 3]

    # find where all of these rays intersect the optical axis
    oa_intersect = rev_xp[rev_valid, -1, 0] + ((rev_xp[rev_valid, 0, 1]-rev_xp[rev_valid, -1, 1]) / rev_vp[rev_valid, -1, 1]) * (rev_vp[rev_valid, -1, 0])
    front_principal_plane = oa_intersect.mean()

    # find the chief ray
    chief_x = jnp.array([
        -5.0,
        (front_principal_plane+5.0) * jnp.tan(angle),
        0.0
    ])

    chief_v = jnp.array([1.0, -jnp.tan(angle), 0.0])
    chief_v = chief_v / jnp.linalg.norm(chief_v)

    return chief_x, chief_v, aperture_rad


def focus_curves(lens, angle, nrays_tan, nrays_sag):
    '''First find the chief ray based on the angle then find the sagittal and tangential rays'''
    # reverse trace the lens to find the chief ray
    # chief_x, chief_v, aperture_rad = chief_ray_by_trace(lens, angle, 10)
    chief_x, chief_v, aperture_rad = system_matrix.chief_ray_from_transfer_matrix(lens, angle)

    aperture_rad = aperture_rad * 1.5
    x_tan = jnp.tile(chief_x, (nrays_tan, 1))
    x_tan = x_tan.at[:, 1].add(jnp.linspace(-aperture_rad, aperture_rad, nrays_tan))
    v_tan = jnp.tile(chief_v, (nrays_tan, 1))
    L_tan = jnp.ones((nrays_tan,))


    x_sag = jnp.tile(chief_x, (nrays_sag, 1))
    x_sag = x_sag.at[:, 2].add(jnp.linspace(-aperture_rad, aperture_rad, nrays_sag))
    v_sag = jnp.tile(chief_v, (nrays_sag, 1))
    L_sag = jnp.ones((nrays_sag,))


    # create tracer
    surf_funs = curvature_sphere.functionSuite()
    surfs = zemax.zemax_state_to_jax_surfaces(lens)
    surfs = jnp.array(surfs)
    vtracer = jax.vmap(lambda x, v, L : tracing.trace_scan_no_warp(surf_funs, surfs, x, v, L, 0, lens.shape[0]))


    # trace the chief ray
    (valid, chief_xp, vp, Lp), (Tx, wargmin) = vtracer(chief_x[None], chief_v[None], jnp.array([1.0]))
    chief_xf = chief_xp[:, -1]

    # trace the tangential rays
    (valid, tan_xp, vp, Lp), (Tx, wargmin) = vtracer(x_tan, v_tan, L_tan)
    tan_xf = tan_xp[valid, -1]
    tan_xi = tan_xp[valid, 0, 1]

    if jnp.all(valid):
        print(f"All tan rays are valid with range {tan_xi.max()} and {tan_xi.min()}")

    # trace the sag rays
    (valid, sag_xp, vp, Lp), (Tx, wargmin) = vtracer(x_sag, v_sag, L_sag)
    sag_xf = sag_xp[valid, -1]
    sag_xi = sag_xp[valid, 0, 2]

    if jnp.all(valid):
        print(f"All sag rays are valid with range {sag_xi.max()} and {sag_xi.min()}")

    tan_focus = jnp.sum(jnp.square(tan_xf - chief_xf), axis=-1)
    sag_focus = jnp.sum(jnp.square(sag_xf - chief_xf), axis=-1)
    return (tan_focus, tan_xi), (sag_focus, sag_xi), chief_xp


def distortion_curve(lens, focal_length, sensor_height, nsamples):
    # reverse trace from the sensor plane
    fpp = front_principal_plane(lens)

    max_angle = jnp.arctan(sensor_height / focal_length)
    angles = jnp.linspace(0, max_angle, nsamples)

    x = jnp.zeros((nsamples, 3)).at[:, 0].set(-5.0)
    x = x.at[:, 1].set((5.0+fpp) * jnp.tan(angles))

    v = jnp.zeros_like(x)
    v = v.at[:, 0].set(jnp.cos(angles))
    v = v.at[:, 1].set(jnp.sin(-angles))

    L = jnp.ones((nsamples,))

    valid, xp, vp, Lp =trace_state_rays(lens, x, v, L, reverse=False)

    end_pos = xp[:, -1, 1]
    target_pos = -focal_length * jnp.tan(angles)
    distortion = end_pos - target_pos
    return distortion, angles


def rms_spot_size(lens, angles, nrays):
    (aper_rad, premat, aper_idx) = system_matrix.aperture_from_transfer_matrix(lens)
    uniform_grid = jnp.meshgrid(*[jnp.linspace(-aper_rad, aper_rad, nrays) for _ in range(2)], indexing='ij')
    x0 = jnp.stack([u.flatten() for u in uniform_grid], axis=-1)
    x0 = jnp.pad(x0, ((0, 0), (0, 1)), constant_values=0.0)
    source_area = (aper_rad * 2) ** 2

    @jax.jit
    def tracer(angle):
        chief_x, chief_v = system_matrix.chief_ray_from_pre_mat(angle, premat)

        x = x0 + chief_x[None, :]
        v = jnp.tile(chief_v, (x.shape[0], 1))
        L = jnp.ones((x.shape[0],))
        return trace_state_rays(lens, x, v, L)

    rms_spots = []
    thrus = []
    for angle in angles:
        valid, xp, vp, Lp = tracer(angle)
        end_pos = xp[valid, -1]
        if end_pos.shape[0] == 0:
            rms_spots.append(jnp.array(0.0))
            continue
        error = end_pos.var(axis=0)
        rms_spot_size = jnp.sqrt(jnp.mean(error))
        rms_spots.append(rms_spot_size)
        thrus.append(source_area * jnp.sum(valid) / valid.shape[0])

    return jnp.array(rms_spots), jnp.array(thrus)



def move_sensor_plane(lens_state, wavelength=587.56):
    if lens_state.shape[1] > 4:
        raise ValueError("lens_state should only have 4 columns, asphere not supported yet")

    surfs = zemax.zemax_state_to_jax_surfaces(lens_state)
    surf_funs = curvature_sphere.functionSuite() # this means only curvature sphere for now

    sample_height = 0.5*lens_state[0, 3]
    ray_samples = 100
    sample_heights = jnp.linspace(0, sample_height, ray_samples+1)[1:]
    x = jnp.zeros((ray_samples, 3)).at[:, 0].set(-5.0)
    x = x.at[:, 1].set(sample_heights)
    v = jnp.zeros_like(x).at[:, 0].set(1.0)
    L = jnp.ones((ray_samples,))

    vtracer = jax.vmap(tracing.trace_scan_no_stop, (None, None, 0, 0, 0, None))
    (valid, xp, vp, Lp), (Tx, wargmin) = vtracer(surf_funs, surfs, x, v, L, 0, lens_state.shape[0])

    # take the last directions and positions and intersect the optical axis
    weights = 1 / sample_heights[valid]
    end_dir = vp[valid, -2, :]
    end_pos = xp[valid, -2, :]

    def sensor_pos(d):
        plane_intersect = end_pos + ((d - end_pos[:, 0:1]) / end_dir[:, 0:1]) * end_dir
        blur_error = jnp.sum(plane_intersect[:, 1:]**2) / end_pos.shape[0]
        return blur_error

    eval_focus = jax.jit(sensor_pos)
    grad_focus = jax.jit(jax.grad(eval_focus))

    d0 = lens_state[:, 1].sum()
    curf = eval_focus(d0)
    print('start focus', curf)
    for i in range(100):
        curf = eval_focus(d0)
        gradf = grad_focus(d0)
        del_d = curf / gradf
        if (jnp.abs(del_d) < constants.NEWTON_EPSILON):
            break
        d0 = d0 - del_d

    print('optimized focus', curf)
    d_result = d0
    new_lens = jnp.array(lens_state).at[-1, 1].set(d_result)
    return new_lens


@jax.jit
def safe_div(a, b):
    bsafe = jnp.where(jnp.isclose(b, 0), 1e-6, b)
    return jnp.where(jnp.isclose(b, 0), constants.FLOAT_BIG_NUMBER, a / bsafe)


def effective_semidiameter(k1 : jnp.ndarray, k2 : jnp.ndarray, th : jnp.ndarray):
    ''' calculate the largest diameter that this lens can have
        return two values, max height for k1 and for k2
    '''
    no_intersect = jnp.logical_and(k1 < 0, k2 > 0)
    no_intersect = jnp.logical_or(no_intersect, k1 < k2)

    square_term = -th*(k1*th - 2)*(k2*th + 2)*(k1*k2*th + 2*k1 - 2*k2)
    denom = k1*k2*th + k1 - k2

    invalid = jnp.logical_or(square_term < 0, jnp.isclose(denom, 0))

    square_term = jnp.where(invalid, constants.FLOAT_BIG_NUMBER, square_term)
    denom = jnp.where(invalid, 1e-6, denom)

    max_height = (1/2) * jnp.sqrt(square_term) / denom
    max_height = jnp.where(no_intersect, constants.FLOAT_BIG_NUMBER, jnp.abs(max_height))

    # check if the intersection point is past the hemisphere of the lens
    maxk1 = jnp.abs(safe_div(1, k1))
    maxk2 = jnp.abs(safe_div(1, k2))
    xstar = jnp.where(invalid, constants.FLOAT_BIG_NUMBER, k1 * th * (2 + k2*th) / (2*denom))
    past1 = jnp.abs(xstar) > maxk1
    past2 = jnp.abs(xstar) > maxk2
    max_k1_height = jnp.where(past1, maxk1, max_height)
    max_k2_height = jnp.where(past2, maxk2, max_height)

    return jnp.abs(max_k1_height), jnp.abs(max_k2_height)


@jax.jit
def clip_semidiameter_to_curvature(lens):
    ''' clip the semidiameter to the curvature of the lens'''

    k1 = lens[:-1, 0]
    k2 = lens[1:, 0]
    th = lens[:-1, 1]

    old_height = lens[:, 3]
    max_height_k1, max_height_k2 = effective_semidiameter(k1, k2, th)
    proj_h1 = jnp.minimum(max_height_k1, lens[:-1, 3])
    lens = lens.at[:-1, 3].set(proj_h1)

    proj_h2 = jnp.minimum(max_height_k2, lens[1:, 3])
    lens = lens.at[1:, 3].set(proj_h2)

    # if jnp.any(proj_h1 / old_height[:-1] < 0.9):
    #     raise ValueError("Projection seems unstable")
    # if jnp.any(proj_h2 / old_height[1:] < 0.9):
    #     raise ValueError("Projection seems unstable")

    return lens


def max_curvature_for_semidiameter(k1, k2, th, semidiam):
    ap = semidiam

    hsquare_term = -th*(k1*th - 2)*(k2*th + 2)*(k1*k2*th + 2*k1 - 2*k2)
    hdenom = k1*k2*th + k1 - k2
    invalid = jnp.logical_or(hsquare_term < 0, jnp.isclose(hdenom, 0))

    ap_square_term = jnp.maximum(1 - (ap*k2)**2, 0.0)
    invalid = jnp.logical_or(invalid, (ap_square_term < 0))

    # num = 2*(-th*jnp.sqrt(ap_square_term)*(k2*th + 2) + (k2*th + 1)*(2*ap**2*k2 + k2*th**2 + 2*th))
    num = 2*(th*jnp.sqrt(ap_square_term)*(k2*th + 2) + (k2*th + 1)*(2*ap**2*k2 + k2*th**2 + 2*th))
    den = (4*ap**2*k2**2*th**2 + 8*ap**2*k2*th + 4*ap**2 + k2**2*th**4 + 4*k2*th**3 + 4*th**2)

    k1soln = jnp.where(invalid, k1, num / den)
    return k1soln


def optimize_sensor_plane(loss_and_grad_fn, lens, eps=0.01):
    if not isinstance(lens, jnp.ndarray):
        lens = jnp.array(lens)

    def change_dist_loss(t, key):
        lens_sens = lens.at[-1, 1].set(t)
        (val, aux), grad = loss_and_grad_fn(lens_sens, 0.0, key)
        grad = grad[-1, 1]
        return val, grad

    sens_dist = optimize(change_dist_loss, lens[-1, 1], eps)

    return lens.at[-1, 1].set(sens_dist)


def optimize(loss_and_grad_fn, t_init, eps, niters=200, key_init=jax.random.PRNGKey(0)):
    tdelt = 0
    key = key_init
    for i in range(niters):
        key, subkey = jax.random.split(key)
        loss, grad = loss_and_grad_fn(t_init + tdelt, subkey)
        tdelt = tdelt - eps * grad / jnp.sqrt(i + 1)

    best_dist = t_init + tdelt
    return best_dist


def average_element_size(lens):
    '''
    Assumes that the last element has 'air' behind it
    '''
    elements = []
    element_len = 0
    for l in lens:
        if l[2] <= lens[-1, 2] + constants.FLOAT_EPSILON:
            elements.append(element_len)
            element_len = 0
        else:
            element_len += l[1]

    return jnp.array(elements)


def glass_appx_volumes(lens : jnp.ndarray):
    '''calculate the volume of glass that is being used in the entire lens'''
    lens = clip_semidiameter_to_curvature(lens)

    nair = lens[-1, 2]
    glass_mask = lens[:-1, 2] > nair

    k1 = lens[:-1, 0]
    h1 = lens[:-1, 3]
    k2 = lens[1:, 0]
    h2 = lens[1:, 3]
    th = lens[:-1, 1]

    volumes = jax.vmap(singlet_appx_volume)(k1, h1, k2, h2, th)
    return jnp.abs(jnp.where(glass_mask, volumes, 0.0))


def singlet_appx_volume(k1, h1, k2, h2, th):
    # calculate the volume of the cylinder between the two surfaces
    sag1 = (k1 * h1**2)
    sag2 = (k2 * h2**2)

    frustum_thickness = th - sag1 + sag2

    spherical_cap1 = (1/6) * jnp.pi * sag1 * (3 * sag1**2 + h1**2)
    spherical_cap2 = (1/6) * jnp.pi * sag2 * (3 * sag2**2 + h2**2)
    cylinder = (1/3) * jnp.pi * frustum_thickness * (h1**2 + h1*h2 + h2**2)

    return spherical_cap1 - spherical_cap2 + cylinder

