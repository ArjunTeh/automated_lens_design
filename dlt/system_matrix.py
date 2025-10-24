import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from typing import Tuple, Union

import functools

from . import optimize
from . import zemax
from . import constants


def ray_transfer_matrix(state) -> Float[Array, "2 2"]:
    '''system matrix used by Meyer-Arendt
    The matrix includes the translation of the final element to the sensor
    therefore a proper lens should have a structure of
        [[ m, P ],
         [ d, 0 ]]
    where P is the power of the lens and m is the magnification of the lens
    '''
    return ray_transfer_matrix_list(state)[0]


def chief_ray_from_transfer_matrix(state, angle) -> Tuple[Float[Array, "2"], Float[Array, "2"], Float]:
    (ymax, pre_mat, aper_idx) = ray_transfer_matrix_list(state)[2]
    chief_x, chief_v = chief_ray_from_pre_mat(angle, pre_mat)
    return chief_x, chief_v, ymax


def chief_ray_from_pre_mat(angle, pre_mat):
    chief_height = -pre_mat[1, 0] * angle / pre_mat[1, 1]
    chief_v = jnp.array([jnp.cos(angle), -jnp.sin(angle), 0.0])

    chief_x = jnp.array([0.0, chief_height, 0.0])
    chief_x = chief_x - chief_v
    return chief_x, chief_v


def aperture_from_transfer_matrix(state):
    '''Compute the aperture of the lens from the transfer matrix
    Returns the surface index that is the stop and the position as well as the aperture radius from the object side'''
    return ray_transfer_matrix_list(state)[2]


def ray_transfer_matrix_mask(state, mask) -> Float[Array, "2 2"]:
    '''system matrix used by Meyer-Arendt
    The matrix includes the translation of the final element to the sensor
    therefore a proper lens should have a structure of
        [[ m, P ],
         [ d, 0 ]]
    where P is the power of the lens and m is the magnification of the lens
    
    In this case, the mask is a boolean array that specifies which elements of the lens to count in the matrix
    '''
    opt_mats = ray_transfer_matrix_mask_list(state, mask)
    sys_mat, _ = jax.lax.scan(lambda acc, x: (jnp.matmul(x, acc), None), xs=opt_mats, init=jnp.eye(2))
    return sys_mat


def ray_transfer_matrix_mask_list(state, mask) -> Float[Array, "N 2 2"]:
    diff_ior = -jnp.diff(state[:, 2], prepend=constants.DEFAULT_EXT_IOR)
    surface_power = state[:, 0] * diff_ior
    scaled_dist = state[:, 1] / state[:, 2]

    r_mats = jnp.tile(jnp.eye(2), (len(state), 1, 1))
    r_mats = r_mats.at[:, 0, 1].set(jnp.where(mask, -diff_ior * state[:, 0], 0))
    t_mats = jnp.tile(jnp.eye(2), (len(state), 1, 1))
    t_mats = t_mats.at[:, 1, 0].set(jnp.where(mask, -scaled_dist, 0))

    opt_mats = jnp.matmul(t_mats, r_mats)
    return opt_mats


@jax.jit
def invert_ray_transfer_matrix(mats) -> Float[Array, "N 2 2"]:
    '''This function assumes (and does not check) that the matrix is determinant 1'''
    invM = jnp.zeros_like(mats)
    invM = invM.at[..., 0, 0].set(mats[..., 1, 1])
    invM = invM.at[..., 1, 1].set(mats[..., 0, 0])
    invM = invM.at[..., 0, 1].set(-mats[..., 0, 1])
    invM = invM.at[..., 1, 0].set(-mats[..., 1, 0])
    return invM


def ray_transfer_matrix_list(state : Array) -> Tuple[Array, list[Array], Tuple[Float, Float, Int]]:
    opt_mat = jnp.eye(2)
    mat_list = []
    prev_ior = constants.DEFAULT_EXT_IOR
    pre_mat = opt_mat
    ymax = jnp.inf
    src_ang = 0.0
    aper_idx = -1
    for i, lens in enumerate(state):
        surface_power = lens[0] * (lens[2] - prev_ior)
        scaled_dist = lens[1] / lens[2]
        t_mat = jnp.eye(2)
        t_mat = t_mat.at[1, 0].set(-scaled_dist)
        r_mat = jnp.eye(2)
        r_mat = r_mat.at[0, 1].set(surface_power)

        # update the aperture max
        yi_max = (lens[3] - opt_mat[1, 0] * src_ang) / opt_mat[1, 1]
        update = yi_max < ymax
        ymax = jnp.where(update, yi_max, ymax)
        aper_idx = jnp.where(update, i, aper_idx)
        pre_mat = jnp.where(update, opt_mat, pre_mat)

        # update the aggregate matrix
        prop_mat = jnp.matmul(t_mat, r_mat)
        opt_mat = jnp.matmul(prop_mat, opt_mat)

        prev_ior = lens[2]

        mat_list.append(r_mat)
        mat_list.append(t_mat)

    return opt_mat, mat_list, (ymax, pre_mat, aper_idx)


def system_matrix_constraint_normal(state: jnp.ndarray):
    ''' Compute the normal vector of the constraint manifold at the given state'''
    def get_mat(state):
        opt_mat = ray_transfer_matrix(state)
        return opt_mat[1, 1]

    normal = jax.grad(get_mat)(state)
    normal = normal / jnp.linalg.norm(normal)
    return normal


def project_to_tangent_plane(state : jnp.ndarray, tangent : jnp.ndarray):
    ''' Project the state onto the tangent plane at the first surface'''
    normal = system_matrix_constraint_normal(state)
    return tangent - jnp.sum(normal*tangent) * normal


def retract_state_onto_focus(state : jnp.ndarray, surf_id : int, curvature=True):
    ''' Set the specified surface curvature(or thickness) so that the lens is in focus'''
    # TODO: There seems to be a bug with this simplification, look at the symbolic calculations to see if there is an issue
    opt_mat, mat_list = ray_transfer_matrix_list(state)

    # R = functools.reduce(lambda x, y: jnp.matmul(y, x), mat_list[:surf_id], jnp.eye(2))
    # L = functools.reduce(lambda x, y: jnp.matmul(y, x), mat_list[surf_id+1:], jnp.eye(2))
    R = jnp.eye(2)
    for i in range(surf_id):
        R = jnp.matmul(mat_list[i], R)

    L = jnp.eye(2)
    for i in range(surf_id+1, len(mat_list)):
        L = jnp.matmul(mat_list[i], L)

    n0 = state[surf_id - 1, 2] if surf_id > 0 else constants.DEFAULT_EXT_IOR
    n1 = state[surf_id, 2]

    proj_state = state.copy()
    if curvature:
        d = state[surf_id, 1]
        k_soln = (d*L[1, 1]*R[0, 1] - n1*L[1, 0]*R[0, 1] - n1*L[1, 1]*R[1, 1])/((d*n0*L[1, 1] - d*n1*L[1, 1] - n0*n1*L[1, 0] + n1**2*L[1, 0])*R[1, 1])
        proj_state = state.at[surf_id, 0].set(k_soln)
        dist = jnp.abs(k_soln - state[surf_id, 0])
    else:
        k = state[surf_id, 0]
        d_soln = n1*(-k*n0*L[1, 0]*R[1, 1] + k*n1*L[1, 0]*R[1, 1] + L[1, 0]*R[0, 1] + L[1, 1]*R[1, 1])/((-k*n0*R[1, 1] + k*n1*R[1, 1] + R[0, 1])*L[1, 1])
        proj_state = state.at[surf_id, 1].set(d_soln)
        dist = jnp.abs(d_soln - state[surf_id, 1])
    
    assert jnp.all(jnp.isfinite(proj_state))
    return proj_state, dist


def retract_state_by_gradient(state : jnp.ndarray):
    '''Use gradient descent to retract the state to the nearest point on the manifold'''
    def get_focus_plane(state, rng_key=None):
        opt_mat = ray_transfer_matrix(state)
        return opt_mat[1, 1]**2, (0,)

    def box_aperture_constraints(state):
        astate = state.copy()
        k = state[:, 0]
        ap = state[:, 3]
        out_mask = (k * ap) > 1
        new_k = jnp.where(jnp.signbit(k), -1 / ap, 1 / ap)
        new_k = jnp.where(out_mask, new_k, k)
        astate = astate.at[:, 0].set(new_k)

        min_bounds = jnp.tile(jnp.array([-0.5,  0, 1.0,   0.0]), (state.shape[0], 1))
        max_bounds = jnp.tile(jnp.array([ 0.5, 40, 2.0, 100.0]), (state.shape[0], 1))
        return jnp.clip(astate, a_min=min_bounds, a_max=max_bounds)

    def grad_project(state, tangent):
        newtan = tangent.at[:, 2].set(0.0) # no change to ior
        # newtan = newtan.at[:, 3].set(0.0) # no change to surface aperture
        return newtan


    loss_fn = jax.jit(jax.value_and_grad(get_focus_plane, has_aux=True))
    output = optimize.run_adam_descent(loss_fn, state, niters=1000, projector=grad_project, constraint=box_aperture_constraints)

    if output[0][-1] > 1e-4:
        print("WARNING: failed to retract state to focus plane")
        print("resulting value", output[0][-1])
        print("improvement", output[0][0] - output[0][-1])

    return output[1][-1]


def retract_state_by_normal(state : jnp.ndarray):
    normal = system_matrix_constraint_normal(state)
    normal = normal.at[:, 2].set(0.0) # no change to ior
    return retract_state_by_rootsearch(state, normal)


def retract_state_by_rootsearch(state : Array, direction : Array):
    '''Use line search to retract the state to the nearest point on the manifold'''
    def get_focus_plane(state):
        opt_mat = ray_transfer_matrix(state)
        return opt_mat[1, 1]

    fn = jax.jit(jax.value_and_grad(get_focus_plane))

    def body_fun(carry):
        iter_num, val, cur_step  = carry
        new_state = state + cur_step * direction
        val, state_grad = fn(new_state)
        val_grad = jnp.sum(state_grad * direction) 
        next_step = cur_step - val / val_grad
        return iter_num + 1, val, next_step

    def cond_fun(carry):
        iter_num, val, cur_step = carry
        return jnp.logical_and((iter_num < 100), (jnp.abs(val) > 1e-4))

    iters, val, step = jax.lax.while_loop(cond_fun, body_fun, (0, 1.0, 0.0))

    return state + step * direction


def minimally_rotate(a : jnp.ndarray, u : jnp.ndarray, v : jnp.ndarray):
    '''Find the minimal rotation that maps u to v and apply it to a'''

    # calculate the orthonormal basis
    un = u / jnp.linalg.norm(u)
    vn = v - jnp.sum(un * v) * un
    vn_mag = jnp.linalg.norm(vn)
    vn = vn / jnp.where(vn_mag > 1e-6, vn_mag, 1.0)

    a_tan = jnp.array([jnp.sum(un * a), jnp.sum(vn * a)])
    a_perp = a - a_tan[0] * un - a_tan[1] * vn

    cos_ang = jnp.sum(un * v) / jnp.linalg.norm(v)
    sin_ang = jnp.sqrt(1 - jnp.clip(cos_ang**2, a_max=1.0))

    a_tan_rot = jnp.array([cos_ang * a_tan[0] - sin_ang * a_tan[1], 
                           sin_ang * a_tan[0] + cos_ang * a_tan[1]])

    return a_tan_rot[0] * un + a_tan_rot[1] * vn + a_perp


def retract_and_transport(position, tangent_vec, direction):
    '''Exponential map approximation and transport for the system matrix manifold'''
    normal = system_matrix_constraint_normal(position)
    new_pos = retract_state_by_rootsearch(position + direction, normal)
    new_normal = system_matrix_constraint_normal(new_pos)

    new_tan = minimally_rotate(tangent_vec, normal, new_normal)
    return new_pos, new_tan