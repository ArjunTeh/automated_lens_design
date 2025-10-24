import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import optimistix as om
import lineax as lx
import equinox as eqx
from jaxtyping import Array, ArrayLike, Bool, Float
from typing import TYPE_CHECKING, Tuple
from jax import jacfwd, device_put

from functools import partial
import numpy as np

from dlt import constants
from dlt import optimize
from dlt import system_matrix
from dlt import primary_sample_space as pss


# def full_lens_mutation_old(key, lens_state : pss.Lens, params)-> Tuple[pss.Lens, float, dict]:
#     '''
#     Pick a random mutation to apply to the lens
#     (currently only support adding and removing singlets from the design)

#     Assumes that lens is already homogenized
#     '''
#     add_prob = params['split_prob']
#     lens = lens_state.data
#     nsurfaces = lens_state.nsurfaces

#     lens = pss.homogenize_lens(lens, 3 * params['max_elements'])
#     u = pss.lens2pss(lens, params['max_elements'])
#     if u.nelements == params['min_elements']:
#         add_prob_corrected = 1.0

#     elif u.nelements >= params['max_elements']:
#         add_prob_corrected = 0.0

#     else:
#         add_prob_corrected = add_prob

#     key, jump_key, coin_key = jr.split(key, 3)
#     if jr.uniform(coin_key) < add_prob_corrected:
#         lens_idx, el_idx = pss.pick_add_lens(lens, key)

#         # sample a singlet to add
#         jump_key, coin_key = jr.split(jump_key)

#         if jr.uniform(coin_key) < params['cement_split_prob']:
#             add_elem = sample_random_cement(jump_key, params)
#             sample_prob = pdf_random_cement(add_elem, params)
#         else:
#             add_elem = sample_random_singlet(jump_key, params)
#             sample_prob = pdf_random_singlet(add_elem, params)

#         start_idx = lens_idx[1]
#         end_idx = start_idx + add_elem.shape[0]
#         prev_t = add_elem[-1, 1]
#         add_elem = add_elem.at[-1, 1].set(lens[start_idx-1, 1])
#         clens = lens.at[start_idx, 1].set(prev_t)
#         add_lens = jnp.concatenate([clens[:start_idx], add_elem, clens[start_idx:nsurfaces]], axis=0)
#         pre_mat = jnp.eye(2)

#         start_mask = jnp.ones_like(add_lens[:, 0], dtype=bool).at[start_idx:end_idx].set(False)
#         end_mask = jnp.ones_like(add_lens[:, 0], dtype=bool)
#         solve_mask = jnp.zeros_like(add_lens, dtype=bool)
#         solve_mask = solve_mask.at[:, 0].set(True).at[start_idx:end_idx, 0].set(False)
#         solve_mask = solve_mask.at[:, 1].set(True).at[start_idx:end_idx, 1].set(False)

#         masks = {
#             'original': start_mask,
#             'result': end_mask,
#             'solve': solve_mask,
#         }

#         opt_success, jump_lens, jac_det = rj_project(add_lens, masks, pre_mat)
#         sample_ratio = 1 / sample_prob

#         # if jump_lens[-1, 1] < 10.0:
#         #     raise ValueError('Invalid thickness')

#         prop_lens = pss.homogenize_lens(jump_lens, lens.shape[0])

#         forward_jump_prob = pss.pdf_add_lens(lens, el_idx)
#         backward_jump_prob = pss.pdf_remove_lens(prop_lens, el_idx + 1)
#         mutate_type = 'add'
#         mutation_type_ratio = (1-add_prob) / add_prob_corrected
#         nsurfaces += (end_idx - start_idx)
#     else:
#         lens_idx, el_idx = pss.pick_remove_lens(lens, key)
#         # if lens_idx[1] - lens_idx[0] > 2:
#         #     return lens_state, 0.0, dict(mutate_success=False)

#         start_idx = lens_idx[0]
#         end_idx = lens_idx[1]

#         sample_prob = pdf_random_singlet(lens[lens_idx[0]:lens_idx[1]], params)

#         merge_lens = lens[:nsurfaces]
#         if lens_idx[0] > 0:
#             merge_lens = merge_lens.at[lens_idx[0]-1, 1].set(lens[lens_idx[1]-1, 1])
#         pre_mat = jnp.eye(2)

#         start_mask = jnp.ones_like(merge_lens[:, 0], dtype=bool)
#         end_mask = jnp.ones_like(merge_lens[:, 0], dtype=bool).at[lens_idx[0]:lens_idx[1]].set(False)
#         solve_mask = jnp.zeros_like(merge_lens, dtype=bool)
#         solve_mask = solve_mask.at[:, 0].set(True).at[start_idx:start_idx+2, 0].set(False)
#         solve_mask = solve_mask.at[:, 1].set(True).at[start_idx:start_idx+2, 1].set(False)

#         masks = {
#             'original': start_mask,
#             'result': end_mask,
#             'solve': solve_mask,
#         }

#         opt_success, jump_lens, jac_det = rj_project(merge_lens, masks, pre_mat)
#         sample_ratio = sample_prob

#         prop_lens = pss.homogenize_lens(jump_lens[end_mask], lens.shape[0])

#         forward_jump_prob = pss.pdf_remove_lens(lens, el_idx)
#         backward_jump_prob = pss.pdf_add_lens(prop_lens, el_idx - 1)
#         mutate_type = 'remove'
#         mutation_type_ratio = add_prob / (1-add_prob_corrected)
#         nsurfaces -= lens_idx[1] - lens_idx[0]

#     aux = {
#         'mutate_success': opt_success,
#         'mutate_type': mutate_type,
#         'forward_jump_prob': forward_jump_prob,
#         'backward_jump_prob': backward_jump_prob,
#         'prop_jac_det': jac_det,
#         'prop_sample_ratio': sample_ratio,
#         'mutation_type_ratio': mutation_type_ratio,
#     }

#     sample_ratio = 1.0 # ignore sample_ratio for now
#     transition_ratio = mutation_type_ratio * (forward_jump_prob / backward_jump_prob) * jac_det * sample_ratio
#     prop_lens_state = pss.Lens(nsurfaces=nsurfaces, data=prop_lens)
#     return prop_lens_state, transition_ratio, aux


def full_lens_mutation(key, lens_state : pss.Lens, params)-> Tuple[pss.Lens, dict]:
    # pick a random lens and then decide what mutation to do to it
    lens_idx = pss.get_lens_idx(lens_state.data)
    lens_u = pss.lens2pss(lens_state.data, params['max_elements'])
    nlens = lens_idx.shape[0]

    key, subkey = jr.split(key)

    mutation_types_list = params['mutation_types']

    # choose a random index in lens_idx
    idx = jr.choice(subkey, lens_idx.shape[0])
    sublens = lens_state.data[lens_idx[idx][0]:lens_idx[idx][1]]

    if params['split_prob'] < 0.0:
        # use more complicated heuristic to determine where to add or remove elements
        lens_powers = pss.lens_power(lens_u)
        tot_power = jnp.sum(jnp.abs(lens_powers))
        power = jnp.abs(lens_powers[idx])
        increase_prob = jnp.sqrt(power / tot_power)
    else:
        # uniform heuristic
        increase_prob = params['split_prob']

    if lens_idx.shape[0] >= params['max_elements']:
        increase_prob = 0.0
    elif lens_idx.shape[0] <= params['min_elements']:
        increase_prob = 1.0

    key, coin_key, jump_key = jr.split(key, 3)
    glue_prob = 0.0

    transition_ratio = 1.0
    if sublens.shape[0] == 2:
        # we picked a singlet, so choose to either remove the singlet or add one after it
        if jr.uniform(coin_key) > increase_prob:
            # increase the number of surfaces
            is_last_surface = lens_idx[idx, 1] == lens_state.nsurfaces
            next_idx = jnp.minimum(idx+1, lens_idx.shape[0]-1)
            next_surface_is_cemented = (lens_idx[next_idx, 1] - lens_idx[next_idx, 0]) == 3
            glue_prob = jnp.where(is_last_surface, 0.0, jnp.exp(-sublens[-1, 1] / 10.0))
            glue_prob = jnp.where(next_surface_is_cemented, 0.0, glue_prob)
            if 'glue' not in mutation_types_list:
                glue_prob = 0.0

            coin_key, split_key = jr.split(coin_key)
            if jr.uniform(coin_key) < glue_prob:
                transition_ratio = (nlens / (nlens - 1)) * increase_prob / ((1 - increase_prob) * (glue_prob))
                mutation_type = 'glue singlet'
                mutation_func = glue_doublet
            else:
                transition_ratio = (nlens / (nlens - 1)) * increase_prob / ((1 - increase_prob) * (1 - glue_prob))
                mutation_type = 'remove singlet'
                mutation_func = remove_singlet

        else:
            # decide if the singlet should be glued to the previous one, or add a singlet
            mutation_type = 'add singlet'
            transition_ratio = ((nlens) / (nlens + 1)) * (1 - increase_prob) / (increase_prob)
            if 'aei' in mutation_types_list:
                mutation_func = aei_singlet
            else:
                mutation_func = add_singlet

    elif sublens.shape[0] == 3:
        # we picked a doublet, so choose to either split the doublet or add a singlet after it
        if jr.uniform(coin_key) < increase_prob:
            mutation_type = 'split doublet'
            if 'glue' in mutation_types_list:
                mutation_func = split_doublet
            elif 'aei' in mutation_types_list:
                mutation_func = aei_singlet
            else:
                mutation_func = add_singlet
            transition_ratio = (nlens / (nlens + 1)) * (1 - increase_prob) / (increase_prob)
        else:
            mutation_type = 'remove cement layer'
            mutation_func = doublet_to_singlet
            transition_ratio = (nlens / (nlens - 1)) * increase_prob / (1 - increase_prob)
    else:
        print(sublens)
        raise ValueError('Found element that is not a singlet or doublet')

    opt_success, jump_lens, jacdet = mutation_func(jump_key, lens_state, lens_idx[idx], params)

    aux = {
        'mutate_success': opt_success,
        'mutate_type': mutation_type,
        'increase_prob': increase_prob,
        'transition_ratio': transition_ratio,
        'glue_prob': glue_prob,
        'lens_idx': lens_idx[idx],
        'jacdet': jacdet,
    }

    jump_lens_data = pss.homogenize_lens(jump_lens.data, lens_state.data.shape[0])
    return pss.Lens(data=jump_lens_data, nsurfaces=jump_lens.nsurfaces), aux


def _maybe_project_lens(new_lens_state, solve_mask, lens_state, old_mask, params):
    if params['use_projection']:
        opt_success, jump_lens, jacdet = project_new_lens_to_old(new_lens_state, solve_mask, lens_state, old_mask)
    else:
        opt_success = True
        jump_lens = new_lens_state
        jacdet = 1.0

    return opt_success, jump_lens, jacdet


def add_singlet(key, lens_state : pss.Lens, lens_idx, params):
    # generate a random singlet and add it after lens_idx
    lens = lens_state.data
    nsurfaces = lens_state.nsurfaces

    add_elem = sample_random_singlet(key, params)
    sample_prob = pdf_random_singlet(add_elem, params)

    start_idx = lens_idx[1]
    prev_t = add_elem[-1, 1] * 0.01
    add_elem = add_elem.at[-1, 1].set(lens[start_idx-1, 1])

    clens = lens.copy()
    clens = clens.at[start_idx:start_idx+2].set(add_elem)
    clens = clens.at[start_idx-1, 1].set(prev_t)
    clens = clens.at[start_idx+2:].set(lens[start_idx:-2])
    add_lens_state = pss.Lens(data=clens, nsurfaces=nsurfaces+2)

    solve_mask = jnp.ones_like(clens, dtype=bool)
    solve_mask = solve_mask.at[start_idx-1, 1].set(False)
    solve_mask = solve_mask.at[start_idx:start_idx+2].set(False)
    solve_mask = solve_mask.at[start_idx+1, 1].set(True)
    solve_mask = solve_mask.at[:, 2:].set(False)
    solve_mask = solve_mask.at[nsurfaces+2:, :].set(False)

    old_mask = jnp.ones_like(lens, dtype=bool)
    old_mask = old_mask.at[:, 2:].set(False)
    old_mask = old_mask.at[nsurfaces:, :].set(False)

    return _maybe_project_lens(add_lens_state, solve_mask, lens_state, old_mask, params)


def remove_singlet(key, lens_state : pss.Lens, lens_idx, params):
    start_idx = lens_idx[0]
    end_idx = lens_idx[1]
    prev_idx = jnp.maximum(0, start_idx-1)

    lens = lens_state.data

    lens_size = end_idx - start_idx

    prev_t = jnp.where(start_idx-1 > 0, lens[start_idx:end_idx, 1].sum(), 0.0)
    clens = lens.copy()
    clens = clens.at[start_idx:-lens_size].set(lens[end_idx:])
    clens = clens.at[prev_idx, 1].add(prev_t)
    remove_lens_state = pss.Lens(data=clens, nsurfaces=lens_state.nsurfaces-lens_size)
    
    solve_mask = jnp.ones_like(clens, dtype=bool)
    solve_mask = solve_mask.at[:, 2:].set(False)
    solve_mask = solve_mask.at[remove_lens_state.nsurfaces:, :].set(False)

    old_mask = jnp.ones_like(lens, dtype=bool)
    old_mask = old_mask.at[:, 2:].set(False)
    old_mask = old_mask.at[lens_state.nsurfaces:, :].set(False)
    old_mask = old_mask.at[start_idx:end_idx, :].set(False)

    return _maybe_project_lens(remove_lens_state, solve_mask, lens_state, old_mask, params)


def aed_singlet(key, lens_state : pss.Lens, lens_idx, params):
    start_idx = lens_idx[0]
    end_idx = lens_idx[1]
    prev_idx = jnp.maximum(0, start_idx-1)

    lens = lens_state.data

    lens_size = end_idx - start_idx

    prev_t = lens[end_idx-1, 1]
    clens = lens.copy()
    clens = clens.at[start_idx:-lens_size].set(lens[end_idx:])
    clens = clens.at[prev_idx, 1].set(prev_t)
    remove_lens = pss.Lens(data=clens, nsurfaces=lens_state.nsurfaces-2)
    
    opt_success = True
    return opt_success, remove_lens



def aei_singlet(key, lens_state : pss.Lens, lens_idx, params):
    '''add another surface to the singlet specified by the element, lens_idx
    
        The reverse is a regular remove singlet
    '''
    nair = params['ior_air']
    ng = params['ior_glass']

    lens = lens_state.data
    nsurfaces = lens_state.nsurfaces

    start_idx = lens_idx[0]
    end_idx = lens_idx[1]

    add_elem_t = params['min_distance']
    add_elem_k = lens[end_idx-1, 0]
    next_t = lens[end_idx-1, 1]
    add_elem = jnp.array([
            [add_elem_k, add_elem_t, ng, params['max_semidiam']],
            [add_elem_k, next_t, nair, params['max_semidiam']],
        ])

    clens = lens.copy()
    clens = clens.at[end_idx-1, 1].set(params['min_distance'])
    clens = clens.at[end_idx:end_idx+2].set(add_elem)
    clens = clens.at[end_idx+2:].set(lens[end_idx:-2])
    add_lens_state = pss.Lens(data=clens, nsurfaces=nsurfaces+2)

    solve_mask = jnp.ones_like(clens, dtype=bool)
    solve_mask = solve_mask.at[start_idx-1, 1].set(False)
    solve_mask = solve_mask.at[start_idx:start_idx+2].set(False)
    solve_mask = solve_mask.at[start_idx+1, 1].set(True)
    solve_mask = solve_mask.at[:, 2:].set(False)

    old_mask = jnp.ones_like(lens, dtype=bool)
    old_mask = old_mask.at[:, 2:].set(False)
    old_mask = old_mask.at[lens_state.nsurfaces:, :].set(False)
    old_mask = old_mask.at[start_idx:end_idx, :].set(False)

    return _maybe_project_lens(add_lens_state, solve_mask, lens_state, old_mask, params)


def glue_doublet(key, lens_state : pss.Lens, lens_idx, params): 
    '''Glue two surfaces together specified by the element, lens_idx, and the one after it'''
    lens = lens_state.data
    nsurfaces = lens_state.nsurfaces

    nair = params['ior_air']
    ng = params['ior_glass']
    ng2 = params['ior_glass_cement']

    if lens_idx[1] - lens_idx[0] > 2:
        raise ValueError('Cannot glue a cemented doublet again')

    # check if the next lens is a singlet
    start_idx = lens_idx[0]
    end_idx = lens_idx[1]
    rem_idx = start_idx + 1

    # middle surfaces to be glued
    m1 = lens[rem_idx]
    m2 = lens[rem_idx+1]

    mid = jnp.array([
            (m1[0] + m2[0]) / 2,
            m2[1],
            ng2,
            (m1[3] + m2[3]) / 2,
        ])

    # remove the airspace between the singlet and the doublet
    # change the second element ior to be the second
    # solve for the rest of the lens, no samples taken?
    new_lens = lens.copy()
    new_lens = new_lens.at[rem_idx:-1].set(lens[end_idx:])
    new_lens = new_lens.at[rem_idx].set(mid)
    new_lens_state = pss.Lens(data=new_lens, nsurfaces=nsurfaces-1)

    # project the lens to the constraints
    solve_mask = jnp.ones_like(new_lens, dtype=bool)
    solve_mask = solve_mask.at[:, 2:].set(False)
    solve_mask = solve_mask.at[nsurfaces-1:, :].set(False)

    old_mask = jnp.ones_like(lens, dtype=bool)
    old_mask = old_mask.at[:, 2:].set(False)
    old_mask = old_mask.at[lens_state.nsurfaces:, :].set(False)
    old_mask = old_mask.at[rem_idx, :].set(False)

    return _maybe_project_lens(new_lens_state, solve_mask, lens_state, old_mask, params)


def split_doublet(key, lens_state : pss.Lens, lens_idx, params):
    '''Split doublet into two surfaces together specified by the element, lens_idx'''
    nair = params['ior_air']
    ng = params['ior_glass']

    lens = lens_state.data
    nsurfaces = lens_state.nsurfaces

    if lens_idx[1] - lens_idx[0] <= 2:
        raise ValueError('attempting to split singlet')

    # insert air between the two surfaces and change index of refraction to be the original glass
    # duplicate the second element curvature and add random amount of thickness then solve for everything else
    start_idx = lens_idx[0]
    mid_idx = lens_idx[0]+1
    end_idx = lens_idx[1]

    dmin, dmax = params['singlet_dist_distribution']
    d = jr.uniform(key, minval=dmin, maxval=dmax)
    m1 = lens[mid_idx].at[1].set(d)
    m1 = m1.at[2].set(nair)

    m2 = lens[mid_idx].at[2].set(ng)

    clens = lens.copy()
    clens = clens.at[mid_idx].set(m1)
    clens = clens.at[mid_idx+1].set(m2)
    clens = clens.at[mid_idx+2:].set(lens[mid_idx+1:-1])
    
    proj_lens = pss.Lens(nsurfaces=nsurfaces+1, data=clens)

    # sample a random singlet to add or remove?
    solve_mask = jnp.ones_like(clens, dtype=bool).at[mid_idx:mid_idx+1].set(False)
    solve_mask = solve_mask.at[:, 2:].set(False)
    solve_mask = solve_mask.at[nsurfaces+1:, :].set(False)

    old_mask = jnp.ones_like(lens, dtype=bool)
    old_mask = old_mask.at[:, 2:].set(False)
    old_mask = old_mask.at[lens_state.nsurfaces:, :].set(False)

    return _maybe_project_lens(proj_lens, solve_mask, lens_state, old_mask, params)


def doublet_to_singlet(key, lens_state : pss.Lens, lens_idx, params):
    '''add another surface to the singlet specified by the element, lens_idx'''
    nair = params['ior_air']
    ng = params['ior_glass']
    ng2 = params['ior_glass_cement']

    lens = lens_state.data
    nsurfaces = lens_state.nsurfaces

    if lens_idx[1] - lens_idx[0] <= 2:
        raise ValueError('attempting to remove layer from a singlet')

    # insert air between the two surfaces and change index of refraction to be the original glass
    # duplicate the second element curvature and add random amount of thickness then solve for everything else
    start_idx = lens_idx[0]
    end_idx = lens_idx[1]

    #TODO(ateh): there is probably a smarter way to pick which side to remove
    key, coin_key = jr.split(key)
    side = jr.choice(coin_key, 2)
    idx = start_idx + side

    clens = lens.copy()
    clens = clens.at[idx:-1].set(lens[idx+1:])
    
    proj_lens = pss.Lens(nsurfaces=nsurfaces-1, data=clens)

    # sample a random singlet to add or remove?
    solve_mask = jnp.ones_like(clens, dtype=bool)
    solve_mask = solve_mask.at[:, 2:].set(False)
    solve_mask = solve_mask.at[nsurfaces-1:, :].set(False)

    old_mask = jnp.ones_like(lens, dtype=bool)
    old_mask = old_mask.at[:, 2:].set(False)
    old_mask = old_mask.at[lens_state.nsurfaces:, :].set(False)
    old_mask = old_mask.at[idx, :].set(False)
    return _maybe_project_lens(proj_lens, solve_mask, lens_state, old_mask, params)


def sample_set_of_singlets(key, nelements, params) -> pss.Lens:
    '''Samples a set of singlets with the given number of elements'''
    key, subkey = jr.split(key)
    singlets = []
    for i in range(nelements):
        singlet = sample_random_singlet(subkey, params)
        singlets.append(singlet)

    lens_data = jnp.concatenate(singlets)
    lens_data = lens_data.at[-1, 1].set(params['focal_length'])
    lens_data = pss.homogenize_lens(lens_data, params['max_surfaces'])
    return pss.Lens(data=lens_data, nsurfaces=nelements*2)


def sample_random_singlet(key, params):
    '''Samples 2 curvatures and 1 thicknesses for a singlet'''
    key, singlet_key = jr.split(key, 2)
    na = params['ior_air']
    ng = params['ior_glass']
    max_semidiam = params['max_semidiam']

    kmu, kstd = params['singlet_curv_distribution']
    dmin, dmax = params['singlet_dist_distribution']

    curv_key, dist_key = jr.split(singlet_key, 2)
    curv_samples = jr.normal(curv_key, (2,))
    dist_samples = jr.uniform(dist_key, (2,), minval=dmin, maxval=dmax)

    ks = kmu + kstd * curv_samples
    ds = dist_samples

    lens = jnp.array([
        [ks[0], ds[0], ng, max_semidiam],
        [ks[1], ds[1], na, max_semidiam],
    ])
    return lens

def pdf_random_singlet(singlet, params):
    kmu, kstd = params['singlet_curv_distribution']
    dmin, dmax = params['singlet_dist_distribution']

    curv = singlet[:, 0]
    dist = singlet[:, 1]
    pcurv = jsp.stats.norm.pdf(curv, loc=kmu, scale=kstd).prod()

    # ignore the last thickness since it is not a part of the element itself
    pdist = jsp.stats.uniform.pdf(dist[:1], loc=dmin, scale=dmax).prod()
    return pcurv * pdist


def sample_random_cement(key, params):
    '''Samples 1 curvatures and 1 thickness for a cemented doublet'''
    key, cement_key, singlet_key = jr.split(key, 3)
    ng2 = params['ior_glass_cement']
    max_semidiam = params['max_semidiam']

    singlet = sample_random_singlet(singlet_key, params)

    kmu, kstd = params['cemented_curv_distribution']
    dmu, dstd = params['cemented_dist_distribution']

    samples = jr.normal(cement_key, (2,))
    ks = kmu + kstd * samples[0]
    ds = dmu + dstd * samples[1]
    cement = jnp.array([ks, ds, ng2, max_semidiam])

    coin = jr.uniform(key)
    s1 = jnp.where(coin < 0.5, singlet[0], cement)
    s2 = jnp.where(coin < 0.5, cement, singlet[0])
    s3 = singlet[1]

    lens = jnp.stack([s1, s2, s3])
    return lens


def pdf_random_cement(cement, params):
    cem_mask = cement[:, 2] == params['ior_glass_cement']
    sin_mask = jnp.logical_not(cem_mask)

    curv = cement[:, 0]
    dist = cement[:, 1]

    kmu, kstd = params['cemented_curv_distribution']
    dmu, dstd = params['cemented_dist_distribution']

    pcurv = jsp.stats.norm.pdf(curv, loc=kmu, scale=kstd).prod(where=cem_mask)
    pdist = jsp.stats.norm.pdf(dist, loc=dmu, scale=dstd).prod(where=cem_mask)

    pscurv = jsp.stats.norm.pdf(curv, 
                                loc=params['singlet_curv_distribution'][0], 
                                scale=params['singlet_curv_distribution'][1]).prod(where=sin_mask)
    psdist = jsp.stats.norm.pdf(dist, 
                                loc=params['singlet_dist_distribution'][0], 
                                scale=params['singlet_dist_distribution'][1]).prod(where=sin_mask)

    return 0.5 * (pcurv * pdist * pscurv * psdist)


# Define the objectives and constraints for the lens system
@jax.jit
def ray_transfer_constraint_res(x, x_mask, x0, x0_mask, pre_mat):
    xmat = system_matrix.ray_transfer_matrix_mask(x, x_mask)
    x0mat = system_matrix.ray_transfer_matrix_mask(x0, x0_mask)

    v = pre_mat[:, 1]
    return (xmat @ v) - (x0mat @ v)


@jax.jit
def mutate_objective(l, l0):
    obj_val = 0.5 * jnp.sum(jnp.square(l - l0))
    return jnp.sum(jnp.square(obj_val))


def vec2lens(l, etas, max_semidiam):
    return jnp.concatenate([
        jnp.reshape(l, (-1, 2)),
        etas[:, None],
        jnp.ones_like(etas[:, None]) * max_semidiam,
    ], axis=1)


def lens2vec(lens):
    return lens[:, (0, 1)].ravel(), lens[:, 2], lens[:, 3].max()


# @eqx.filter_jit
def project_new_lens_to_old(lens : pss.Lens, new_mask : jnp.ndarray, old_lens : pss.Lens, old_mask : jnp.ndarray) -> Tuple[Bool, pss.Lens, Float]:
    new_mat_mask = jnp.arange(lens.data.shape[0]) < lens.nsurfaces
    old_mat_mask = jnp.arange(old_lens.data.shape[0]) < old_lens.nsurfaces

    g_init = ray_transfer_constraint_res(lens.data, new_mat_mask, old_lens.data, old_mat_mask, jnp.eye(2))
    lambda0 = jnp.ones_like(g_init)

    @eqx.filter_jit
    def cons_fun(xdat, x0):
        xnew = lens.data.at[new_mask].set(xdat)
        xold = old_lens.data.at[old_mask].set(x0)
        return ray_transfer_constraint_res(xnew, new_mat_mask, xold, old_mat_mask, jnp.eye(2))

    @eqx.filter_jit
    def loss_fun(xdat, x0):
        return mutate_objective(xdat, x0)

    x0 = lens.data[new_mask].ravel()
    opt_params = x0
    success, (xstar, lam), solinfo = optimize.solve_constrained_optimization(loss_fun, cons_fun, x0, opt_params, lambda0=lambda0)
    xstar_jac = optimize.constrained_optimization_jacobian(loss_fun, cons_fun, x0, xstar, lam)
    xstar_jacdet = jnp.linalg.det(xstar_jac)

    solved_lens = lens.data.at[new_mask].set(xstar)
    new_lens = lens._replace(data=solved_lens)

    return success, new_lens, xstar_jacdet


def pick_random_indices(key, mask, npicks):
    nchoices = jnp.count_nonzero(mask)
    p_mask = (mask.astype(int) / nchoices).flatten()

    idx_range = jnp.arange(mask.size)
    indices = jr.choice(key, idx_range, shape=(npicks,), replace=False, p=p_mask)

    idx = jnp.unravel_index(indices, mask.shape)
    choice_mask = jnp.zeros_like(mask, dtype=bool).at[idx].set(True)
    return choice_mask