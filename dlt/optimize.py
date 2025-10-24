import jax.numpy as jnp
import jax.scipy.optimize as jop
import jax
import jax.experimental.checkify as checkify
from tqdm import tqdm
import jaxopt

import equinox as eqx
import optimistix as optx
import lineax as lx

from . import tracing
from . import constants


def optimize_zemax_lens(loss_func, state):
    output = jop.minimize(loss_func, state, method='BFGS')
    return output
    

def assert_all_fn(valid):
    err, _ = checkify.checkify(lambda x : checkify.check(jnp.all(x), 'not all valid'))(valid)
    err.throw()


def finite_difference(loss_fn, state, eps=1e-6):
    orig_shape = state.shape
    stateravel = jnp.asarray(state.ravel())
    cur_loss = loss_fn(state)

    grad = jnp.zeros_like(stateravel)
    for i in range(stateravel.shape[0]):
        stateraveldx = stateravel.at[i].add(eps)
        dx_loss = loss_fn(stateraveldx.reshape(orig_shape))
        grad = grad.at[i].set((dx_loss - cur_loss) / eps)

    return grad.reshape(orig_shape)


def run_lbfgs(loss_fn, state : jnp.ndarray , niters : int, box_bounds, show_progress=False):

    box_bounds = (jnp.asarray(box_bounds[0]), jnp.asarray(box_bounds[1]))

    init_params = jnp.asarray(state).ravel()
    lm = jaxopt.LBFGSB(loss_fn, value_and_grad=True, 
                       maxiter=niters, 
                       has_aux=True, 
                       stepsize=0.,
                       maxls=40,
                       verbose=True, 
                       implicit_diff=False, 
                       jit='auto', 
                       unroll='auto')
    opt_state = lm.init_state(init_params, bounds=box_bounds)

    loss_hist = [opt_state.value]
    state_hist = [init_params.reshape(state.shape)]
    other_info_hist = [opt_state.aux]

    update_fn = jax.jit(lm.update)
    for i in tqdm(range(niters), disable=(not show_progress)):
        result = update_fn(state_hist[-1].ravel(), opt_state, bounds=box_bounds)
        state_hist.append(result.params.reshape(state.shape))
        loss_hist.append(result.state.value)
        other_info_hist.append(result.state.aux)
        opt_state = result.state

        if result.state.failed_linesearch:
            print("failed linesearch at iter", i)
            break

    return loss_hist, state_hist, other_info_hist


def run_LM_least_squares(loss_fn, state : jnp.ndarray , niters : int, show_progress=False):
    def loss_mat_fn(p):
        (loss, aux), grad = loss_fn(p)
        return jnp.expand_dims(loss, axis=0), aux

    init_params = state.copy().ravel()
    lm = jaxopt.LevenbergMarquardt(loss_mat_fn, maxiter=niters, has_aux=True, verbose=True, implicit_diff=False, jit=False, unroll=False)
    opt_state = lm.init_state(init_params)

    loss_hist = [opt_state.loss]
    state_hist = [init_params.reshape(state.shape)]
    other_info_hist = [opt_state.aux]
    for i in tqdm(range(niters), disable=(not show_progress)):
        result = lm.update(state_hist[-1].ravel(), opt_state)
        state_hist.append(result.params.copy().reshape(state.shape))
        loss_hist.append(result.state.loss)
        other_info_hist.append(result.state.aux)
        opt_state = result.state

    return loss_hist, state_hist, other_info_hist


def run_adam_descent(loss_fn, state, niters, termination_eps=None, constraint=None, projector=None, eps=1e-3, rng_key=None, beta1=0.9, beta2=0.999, mo1=None, mo2=None, show_progress=False, save_cadence=1, logger=None):
    '''returns loss trajectory, state trajectory, other info from the loss function, *adam info'''
    radii = state.copy()

    loss_hist = []
    grad_hist = []
    state_hist = []
    other_info_hist = []
    mo1_hist = []
    mo2_hist = []
    update_direction = []
    rng_key_hist = []

    rng_key = jax.random.PRNGKey(0) if rng_key is None else rng_key

    mo1 = jnp.zeros_like(state) if mo1 is None else mo1
    mo2 = jnp.zeros_like(state) if mo2 is None else mo2
    for i in tqdm(range(niters), disable=(not show_progress)):
        if not jnp.all(jnp.isfinite(radii)):
            print("failed at iter", i)
            break

        rng_key, subkey = jax.random.split(rng_key)
        (loss_val, other_info), loss_grad = loss_fn(radii, subkey)

        if i == 0:
            prev_loss_val = loss_val

        if not jnp.all(jnp.isfinite(loss_grad)):
            print("calculated nan grad at iter", i)
            break
    
        if termination_eps is not None and (i > 0): 
            # Add epsilon termination condition
            # rel_diff = jnp.abs(loss_val - prev_loss_val) / jnp.abs(prev_loss_val)
            rel_diff = jnp.linalg.norm(loss_grad) / jnp.abs(prev_loss_val)
            if rel_diff < termination_eps:
                print("loss val", loss_val)
                print("grad norm", jnp.linalg.norm(loss_grad))
                print("prev loss val", prev_loss_val)
                print("relative difference", rel_diff)
                print("termination condition met at iter", i)
                break

        if i % save_cadence == 0:
            if logger is not None:
                logger(i, radii, loss_val, loss_grad, other_info)
            loss_hist.append(loss_val)
            grad_hist.append(loss_grad)
            state_hist.append(radii)
            other_info_hist.append(other_info)
            mo1_hist.append(mo1)
            mo2_hist.append(mo2)
            rng_key_hist.append(subkey)

        if projector is not None:
            loss_grad = projector(radii, loss_grad)

        # if jnp.any(loss_grad > 1.0):
        #     raise ValueError("loss grad is too large")

        mo1 = beta1*mo1 + loss_grad * (1 - beta1)
        mo2 = beta2*mo2 + (loss_grad**2) * (1 - beta2)

        mo1hat = mo1 / (1 - beta1**(i+1))
        mo2hat = mo2 / (1 - beta2**(i+1))

        update_dir = -eps * mo1hat / (jnp.sqrt(mo2hat) + constants.FLOAT_EPSILON)

        radii = radii + update_dir
        update_direction.append(update_dir)

        prev_loss_val = loss_val

        if constraint is not None:
            radii = constraint(radii)

    loss_hist.append(loss_val)
    grad_hist.append(loss_grad)
    state_hist.append(radii)
    other_info_hist.append(other_info)
    mo1_hist.append(mo1)
    mo2_hist.append(mo2)
    update_direction.append(update_dir)
    rng_key_hist.append(subkey)

    return loss_hist, state_hist, other_info_hist, grad_hist, mo1_hist, mo2_hist, update_direction, rng_key_hist


def manifold_adam(loss_fn, state, niters, constraint=None, manifold_fns=None, eps=1e-3, beta1=0.9, beta2=0.999, mo1=None, mo2=None, show_progress=False):
    '''
    Uses a manifold version of adam, which relinquishes the squared term
    returns loss trajectory, state trajectory, other info from the loss function, *adam info
    '''
    radii = jnp.asarray(state)

    if manifold_fns is not None:
        exp_and_transport, projector = manifold_fns
    else:
        exp_and_transport = lambda x, y, z : (x + z, y)
        projector = lambda x, y : y

    mo1 = jnp.zeros_like(state) if mo1 is None else mo1
    mo2 = 0.0 if mo2 is None else mo2

    loss_hist = []
    grad_hist = []
    state_hist = []
    other_info_hist = []
    mo1_hist = []
    mo2_hist = []

    rng_key = jax.random.PRNGKey(0)

    for i in tqdm(range(niters), disable=(not show_progress)):
        if not jnp.all(jnp.isfinite(radii)):
            print("failed at iter", i)
            break

        rng_key, subkey = jax.random.split(rng_key)
        (loss_val, other_info), loss_grad = loss_fn(radii, subkey)

        loss_hist.append(loss_val)
        grad_hist.append(loss_grad)
        state_hist.append(radii)
        other_info_hist.append(other_info)
        mo1_hist.append(mo1)
        mo2_hist.append(mo2)

        loss_grad = projector(radii, loss_grad)

        mo1 = beta1*mo1 - loss_grad * (1 - beta1)
        mo2 = beta2*mo2 + jnp.sum(loss_grad*loss_grad) * (1 - beta2)

        mo1hat = mo1 / (1 - beta1**(i+1))
        mo2hat = mo2 / (1 - beta2**(i+1))

        delta_tangent = eps * mo1hat / (jnp.sqrt(mo2hat) + constants.FLOAT_EPSILON)

        radii, mo1 = exp_and_transport(radii, mo1, delta_tangent)

        if constraint is not None:
            radii = constraint(radii)
    
    loss_hist.append(loss_val)
    grad_hist.append(loss_grad)
    state_hist.append(radii)
    other_info_hist.append(other_info)
    mo1_hist.append(mo1)
    mo2_hist.append(mo2)

    return loss_hist, state_hist, other_info_hist, grad_hist, mo1_hist, mo2_hist


def gradient_descent(loss_fn, state, niters, constraint=None, manifold_fns=None, eps=1e-3, show_progress=False, save_cadence=1):
    '''returns loss trajectory, state trajectory, other info from the loss function'''
    radii = state.copy()

    if manifold_fns is not None:
        project_fn, retract_fn = manifold_fns
        radii = retract_fn(radii, radii)
    else:
        project_fn = lambda x, y : y
        retract_fn = lambda x, y : y

    loss_hist = []
    grad_hist = []
    state_hist = []
    orig_state_hist = []
    other_info_hist = []

    rng_key = jax.random.PRNGKey(0)

    for i in tqdm(range(niters), disable=(not show_progress)):
        assert jnp.all(jnp.isfinite(radii))

        rng_key, subkey = jax.random.split(rng_key)
        (loss_val, other_info), loss_grad = loss_fn(radii, subkey)

        assert jnp.all(jnp.isfinite(loss_grad))

        loss_grad = project_fn(radii, loss_grad)

        if i % save_cadence == 0:
            loss_hist.append(loss_val)
            grad_hist.append(loss_grad)
            state_hist.append(radii)
            other_info_hist.append(other_info)

        radii_step = radii - eps * loss_grad / jnp.sqrt(i+1)
        orig_state_hist.append(radii_step)
        radii = retract_fn(radii, radii_step)

        if constraint is not None:
            radii = constraint(radii)
    
    loss_hist.append(loss_val)
    grad_hist.append(loss_grad)
    state_hist.append(radii)
    other_info_hist.append(other_info)

    return loss_hist, state_hist, other_info_hist, grad_hist, orig_state_hist


def RMSProp_descent(loss_fn, state, niters, constraint=None, projector=None, eps=1e-3, rng_key=None, beta=0.9, mo=None, show_progress=False):
    radii = state.copy()

    loss_hist = []
    grad_hist = []
    state_hist = []
    other_info_hist = []

    rng_key = jax.random.PRNGKey(0) if rng_key is None else rng_key

    mo = jnp.zeros_like(state) if mo is None else mo
    for i in tqdm(range(niters), disable=(not show_progress)):
        if not jnp.all(jnp.isfinite(radii)):
            print("failed at iter", i)
            break

        rng_key, subkey = jax.random.split(rng_key)
        (loss_val, other_info), loss_grad = loss_fn(radii, subkey)

        loss_hist.append(loss_val)
        grad_hist.append(loss_grad)
        state_hist.append(radii)
        other_info_hist.append(other_info)

        if projector is not None:
            loss_grad = projector(radii, loss_grad)

        mo = beta*mo + (loss_grad**2) * (1 - beta)

        radii = radii - eps * loss_grad / (jnp.sqrt(mo) + constants.FLOAT_EPSILON)

        if constraint is not None:
            radii = constraint(radii)
    
    loss_hist.append(loss_val)
    grad_hist.append(loss_grad)
    state_hist.append(radii)
    other_info_hist.append(other_info)

    return loss_hist, state_hist, other_info_hist, grad_hist, mo


@eqx.filter_jit
def solve_constrained_optimization(f, constraints, x0, params, lambda0=None, rtol=1e-5, atol=1e-7, max_steps=200):
    """
    Solve a constrained optimization problem using Lagrange multipliers and Optimistix.
    
    Parameters:
    -----------
    f : callable
        The objective function to minimize, takes a vector x and returns a scalar.
    constraints : callable
        Function that computes the constraint values, takes a vector x and returns a vector c,
        where c(x) = 0 represents the constraints.
    x0 : array
        Initial guess for the solution.
    params : array
        Additional parameters for the objective function and constraints.
    lambda0 : array, optional
        Initial guess for the Lagrange multipliers. If None, initialize with ones.
    rtol, atol : float
        Relative and absolute tolerance for optimization.
    max_steps : int
        Maximum number of iterations.
    
    Returns:
    --------
    solution : tuple
        The solution (x, lambda) where x is the optimized parameter and lambda are the Lagrange multipliers.
    """
    # Set up the Lagrangian
    def lagrangian(x_lambda, params, aux):
        x, lam = x_lambda
        return f(x, params) + jnp.dot(lam, constraints(x, params))

    y0 = (x0, lambda0)
    
    # Define the gradient of the Lagrangian
    grad_lagrangian = jax.grad(lambda xl, aux: lagrangian(xl, params, aux), has_aux=False)
    
    solver = optx.Newton(rtol, atol, linear_solver=lx.AutoLinearSolver(well_posed=False))
    # solver = optx.Dogleg(rtol, atol, linear_solver=lx.AutoLinearSolver(well_posed=False))
    
    # Use a root-finding approach to find where the gradient is zero
    solution = optx.root_find(
        fn=grad_lagrangian,
        y0=y0,
        solver=solver,
        max_steps=max_steps,
        throw=False,
    )
    
    xstar = solution.value
    success = solution.result == optx.RESULTS.successful
    return success, xstar, solution


@eqx.filter_jit
def constrained_optimization_jacobian(f, constraints, x0, xstar, lambdastar, rtol=1e-5, atol=1e-7, max_steps=200):
    """
    Calculate the Jacobian of xstar w.r.t. x0 using Lagrange multipliers.
    
    Parameters:
    -----------
    f : callable
        The objective function to minimize. takes two arguments: x0 and xstar
    constraints : callable
        Function that computes the constraint values. take two arguments: x0 and xstar
    x0 : array
        Initial guess for the solution.
    xstar : array
        The optimal solution found by the solver.
    lambda0 : array, optional
        Initial guess for the Lagrange multipliers. If None, initialize with ones.
    rtol, atol : float
        Relative and absolute tolerance for optimization.
    max_steps : int
        Maximum number of iterations.
    
    Returns:
    --------
    xstar : array
        The optimized parameter after applying constraints.
    """

    # x0 and xstar should be the same shape
    if x0.shape != xstar.shape:
        raise ValueError("x0 and xstar must have the same shape")

    def lagrangian(x, params, lam):
        return f(x, params) + jnp.dot(lam, constraints(x, params))

    # Define the gradient of the Lagrangian
    second_orders = jax.jacrev(jax.jacfwd(lagrangian, argnums=(0, 1, 2)), argnums=(0, 1, 2))(xstar, x0, lambdastar)

    a11 = second_orders[0][0]  # d^2f/dx^2
    a12 = second_orders[0][2]  # d^2f/dxdlambda
    a21 = second_orders[2][0]  # d^2c/dx^2
    a22 = second_orders[2][2]  # d^2c/dlambda^2


    b1 = -second_orders[1][0]  # df/dx
    b2 = -second_orders[2][1]  # df/dlambda

    A = jnp.concat([
        jnp.concat([a11, a12], axis=1),
        jnp.concat([a21, a22], axis=1)
    ], axis=0)
    b = jnp.concat([b1, b2], axis=0)

    jac = jnp.linalg.solve(A, b)  # Solve for the Jacobian    

    # Implement gradient descent with constraints here if needed
    return jac[:a11.shape[0], :a11.shape[1]]  # Placeholder for actual implementation