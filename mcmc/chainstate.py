import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from jaxtyping import Float
from typing import NamedTuple, Optional, Tuple, Union, Dict

from .langevin_dynamics import adam_state, initialize_adam_state
from dlt.primary_sample_space import Lens, NormalizedLens, lens2normalized, normalized2lens

class ChainState(NamedTuple):
    '''Generic class for storing information about a state for an MCMC chain'''
    accept: bool
    cur_state: NormalizedLens
    ray_log_den: Float
    ray_log_grad: NormalizedLens
    opt_state : Union[optax.OptState, adam_state]
    loss_aux: dict = {}
    mutate_aux: Optional[dict] = None
    temperature: Float = 1.0


def make_chain_state_from_lens(lens : NormalizedLens, loss, grad, aux, opt_state : Union[optax.OptState, adam_state, None] = None) -> ChainState:

    if opt_state is None:
        opt_state = initialize_adam_state(lens.data, lens.nsurfaces)
    new_state = ChainState(accept=True,
                           cur_state=lens,
                           ray_log_den=-loss,
                           ray_log_grad=grad,
                           opt_state=opt_state,
                           loss_aux=aux)
    return new_state


def save_chain_state(fname : str, state : ChainState):
    lens = state.cur_state
    lens_data = lens.data[:lens.nsurfaces]
    lens_char_curv = lens.characteristic_curvature
    lens_char_thick = lens.characteristic_thickness
    grad_data = state.ray_log_grad.data[:lens.nsurfaces]

    loss_aux = state.loss_aux
    if loss_aux is not None:
        # make new dictionary with 'loss_aux_' appended to the keys
        loss_aux_format = {f'loss_aux_{k}': v for k, v in loss_aux.items()}
    else:
        loss_aux_format = {}

    mutate_aux = state.mutate_aux
    if mutate_aux is not None:
        mutate_aux_format = {f'mutate_aux_{k}': v for k, v in mutate_aux.items()}
    else:
        mutate_aux_format = {}

    jnp.savez(fname, 
              lens=lens_data, 
              lens_characteristic_curvature=lens_char_curv,
              lens_characteristic_thickness=lens_char_thick,
              grad=grad_data,
              nsurfaces=lens.nsurfaces, 
              ray_log_den=state.ray_log_den, 
              **loss_aux_format,
              **mutate_aux_format,
    )


def load_chain_state(fname : str):
    data: Dict[str, jnp.ndarray] = jnp.load(fname) # type: ignore
    lensdata = jnp.array(data['lens'])
    ray_log_den = data['ray_log_den']
    char_curv = data.get('lens_characteristic_curvature', 1.0)
    char_thick = data.get('lens_characteristic_thickness', 1.0)
    grad_data = jnp.array(data.get('grad', lensdata))
    normlens = NormalizedLens(data=lensdata, nsurfaces=data['nsurfaces'], characteristic_curvature=char_curv, characteristic_thickness=char_thick)
    grad_normlens = normlens._replace(data=grad_data)
    opt_state = initialize_adam_state(data['lens'], data['nsurfaces'])

    # read in all keys that start with 'loss_aux_'
    loss_aux = {k[9:]: v for k, v in data.items() if k.startswith('loss_aux_')}
    mutate_aux = {k[11:]: v for k, v in data.items() if k.startswith('mutate_aux_')}

    return ChainState(cur_state=normlens, 
                      ray_log_den=ray_log_den,
                      ray_log_grad=grad_normlens, 
                      opt_state=opt_state, 
                      accept=True, 
                      loss_aux=loss_aux, 
                      mutate_aux=mutate_aux, 
                      temperature=1.0)
