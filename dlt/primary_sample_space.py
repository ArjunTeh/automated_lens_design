import jax
import jax.numpy as jnp
import jax.tree as jt
import jax.random as jr

import equinox as eqx

from . import system_matrix
from . import constants

from jaxtyping import Int, Float, Array, Bool
from typing import NamedTuple, TypeVar, Tuple
from collections import namedtuple


class Lens(NamedTuple):
    '''NamedTuple for storing lens data'''
    data: jnp.ndarray
    nsurfaces: Int

    def toarray(self):
        return self.data[:self.nsurfaces]


class NormalizedLens(NamedTuple):
    '''NamedTuple for storing normalized lens data'''
    data: jnp.ndarray
    nsurfaces: Int
    characteristic_curvature: Float
    characteristic_thickness: Float

    def toarray(self):
        return self.data[:self.nsurfaces]


class PSSLens(NamedTuple):
    '''NamedTuple for storing lens data in primary sample space'''
    nelements: Int
    eltype: jnp.ndarray
    data: jnp.ndarray

SURFACE_DATA_SIZE = 4
ELEMENT_DATA_SIZE = (3, SURFACE_DATA_SIZE)

LensType = TypeVar('LensType', Lens, NormalizedLens)

def lens2normalized(lens : Lens, max_curv, max_thick):
    '''Convert a lens to normalized lens data'''
    if isinstance(lens, NormalizedLens):
        return lens
    data = lens.data
    data = data.at[:, 0].divide(max_curv)
    data = data.at[:, 1].divide(max_thick)
    return NormalizedLens(data, lens.nsurfaces, max_curv, max_thick)


def normalized2lens(normlens : NormalizedLens):
    dat = normlens.data
    dat = dat.at[:, 0].multiply(normlens.characteristic_curvature)
    dat = dat.at[:, 1].multiply(normlens.characteristic_thickness)
    return Lens(dat, normlens.nsurfaces)


def pss2lens(u : PSSLens, max_elements):
    max_surfaces = max_elements * 3

    lens_data = jnp.zeros((max_surfaces, SURFACE_DATA_SIZE), dtype=u.data.dtype)

    nelements = u.nelements
    eltype = u.eltype
    data = u.data

    sur_idx = 0
    for i in range(nelements):
        if eltype[i] == 1:
            lens_data.at[sur_idx:sur_idx+2].set(data[i][:2])
            sur_idx += 2
        elif eltype[i] == 2:
            lens_data.at[sur_idx:sur_idx+3].set(data[i][:3])
            sur_idx += 3

    nsurfaces = sur_idx
    return lens_data, nsurfaces


def lens2pss(lens, max_elements):
    lens_idx = get_lens_idx(lens)

    nelements = lens_idx.shape[0]
    eltype = jnp.zeros((max_elements,), dtype=jnp.int32)
    data = jnp.zeros((max_elements, *ELEMENT_DATA_SIZE), dtype=lens.dtype)
    data = data.at[:, :, 2].set(1.0)

    for i, li in enumerate(lens_idx):
        if li[1] - li[0] == 2:
            eltype = eltype.at[i].set(1)
            data = data.at[i, :2].set(lens[li[0]:li[1]])
        elif li[1] - li[0] == 3:
            eltype = eltype.at[i].set(2)
            data = data.at[i, :3].set(lens[li[0]:li[1]])
    
    return PSSLens(nelements, eltype, data)


@jax.jit
def topological_equals(lens1 : LensType, lens2 : LensType) -> Bool:
    # 1 if glass, 0 if air, -1 if greater the nsurfaces
    mask = jnp.arange(lens1.data.shape[0]) < lens1.nsurfaces
    glass_air_1 = jnp.where(lens1.data[:, 2] > constants.DEFAULT_EXT_IOR, 1, 0)
    glass_air_1 = jnp.where(mask, glass_air_1, -1)

    mask = jnp.arange(lens2.data.shape[0]) < lens2.nsurfaces
    glass_air_2 = jnp.where(lens2.data[:, 2] > constants.DEFAULT_EXT_IOR, 1, 0)
    glass_air_2 = jnp.where(mask, glass_air_2, -1)

    # check if the two arrays are equal
    mat_equals = jnp.all(glass_air_2 == glass_air_1)

    # check lens type -1 for negative, 0 for close to 0, 1 for positive
    zero1 = jnp.isclose(lens1.data[:, 0], 0.0, atol=1e-4)
    zero2 = jnp.isclose(lens2.data[:, 0], 0.0, atol=1e-4)

    sign1 = jnp.sign(lens1.data[:, 0])
    sign1 = jnp.where(zero1, 0, sign1)
    sign2 = jnp.sign(lens2.data[:, 0])
    sign2 = jnp.where(zero2, 0, sign2)
    # sign_equals = jnp.all(sign1 == sign2)
    sign_equals = True
    return jnp.logical_and(mat_equals, sign_equals)


@jax.jit
def topological_class(lens : LensType) -> Tuple[Bool, Int]:
    # True is glass, False if air
    # 1 if glass, 0 if air, -1 if greater the nsurfaces
    mask = jnp.arange(lens.data.shape[0]) < lens.nsurfaces
    glass = jnp.where(lens.data[:, 2] > constants.DEFAULT_EXT_IOR, True, False)
    glass = jnp.where(mask, glass, False)

    sign = jnp.sign(lens.data[:, 0])
    sign = jnp.where(jnp.isclose(lens.data[:, 0], 0.0, atol=1e-4), 0, sign)
    sign = jnp.where(mask, sign, 0)

    return glass, sign


@eqx.filter_jit
def topological_distance(lens1 : LensType, lens2 : LensType, pnorm : Int) -> Float:
    glass1, sign1 = topological_class(lens1)
    glass2, sign2 = topological_class(lens2)

    glass_diff = jnp.logical_xor(glass1, glass2).astype(int)
    sign_diff = jnp.abs(sign1 - sign2)

    glass_dist = jnp.linalg.norm(glass_diff, ord=pnorm)
    sign_dist = jnp.linalg.norm(sign_diff, ord=pnorm)
    return glass_dist + sign_dist


def homogenize_lens(lens_data : Float, max_surfaces):
    nsurfaces = lens_data.shape[0]
    lens_pad = jnp.resize(lens_data, (max_surfaces, lens_data.shape[1]))
    lens_pad = lens_pad.at[nsurfaces:, 0].set(0.0)
    lens_pad = lens_pad.at[nsurfaces:, 1].set(0.1)
    lens_pad = lens_pad.at[nsurfaces:, 2].set(1.0)
    lens_pad = lens_pad.at[nsurfaces:, 3].set(1000)
    return lens_pad


def expand_lens(lens : LensType, max_surfaces):
    lens_data = lens.data
    nsurfaces = lens.nsurfaces
    lens_pad = homogenize_lens(lens_data, max_surfaces)
    return lens._replace(data=lens_pad)


def lens_mask(u : PSSLens):
    max_elements = u.eltype.shape[0]
    cemented_mask = jnp.ones((max_elements, 3), dtype=jnp.bool_)
    singlet_mask = cemented_mask.at[:, 2].set(False)
    masks = jnp.where((u.eltype == 1)[:, None], singlet_mask, cemented_mask)
    return masks

def ray_transfer_matrix_from_lens(lens : Lens) -> Float[Array, "2 2"]:
    '''Compute the system matrix from a lens object'''
    mask = jnp.arange(lens.data.shape[0]) < lens.nsurfaces
    return system_matrix.ray_transfer_matrix_mask(lens.data, mask)


def lens_power(u : PSSLens):
    max_elements = u.eltype.shape[0]

    vmap_mat_fn = jax.vmap(system_matrix.ray_transfer_matrix_mask)

    masks = lens_mask(u)

    powers = vmap_mat_fn(u.data, masks)[:, 0, 1]

    element_mask = jnp.arange(max_elements) < u.nelements
    powers = jnp.where(element_mask, powers, 0.0)
    return powers


def get_lens_idx(lens : Array):
    nair = lens[-1, 2] + 1e-4 + constants.FLOAT_EPSILON
    iors = jnp.concat([nair[None], lens[:, 2]])
    glass = iors > nair

    rise = ~glass[:-1] & glass[1:]
    fall = glass[:-1] & ~glass[1:]

    rise_idx = jnp.argwhere(rise)[:, 0]
    fall_idx = jnp.argwhere(fall)[:, 0]

    # we assume that there rise and fall are the same size
    assert rise_idx.shape == fall_idx.shape

    lens_idx = jnp.stack([rise_idx, fall_idx + 1], axis=1)
    return lens_idx


def lens_statistics(lens : NormalizedLens):
    lens_array = normalized2lens(lens).toarray()
    lens_idx = get_lens_idx(lens_array)

    # calculate lens powers
    powers = jnp.array([system_matrix.ray_transfer_matrix(lens_array[li[0]:li[1]])[0, 1] for li in lens_idx])
    thickness = jnp.array([lens_array[li[0]:(li[1]-1), 1].sum() for li in lens_idx]) # we ignore the distance from the last element
    iors = jnp.array([lens_array[li[0], 2] for li in lens_idx])
    diameters = jnp.array([lens_array[li[0], 3] for li in lens_idx])
    curvatures = lens_array[:, 0]
    return dict(
        powers=powers,
        thickness=thickness,
        iors=iors,
        diameters=diameters,
        curvatures=curvatures,
    )


