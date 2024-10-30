# PACKAGES
import numpy as np
from scipy.integrate import solve_ivp
from multiprocessing import Pool
from scipy.interpolate import CubicSpline
from loguru import logger

# LOCAL FUNCTIONS
from modules.dynamics.system_eqs import odeGalerkin_einsum, odeGalerkin_matrix

def BFI(i, ni, Xi_f, Xi_b, t_sub, GP, flag_integration):
    """
    Performs backward-forward weighted integration within single integration time frame, bounded by initial and final condition.

    :param i: integration frame index
    :param ni: number of integration frames
    :param Xi_f: initial condition of states for forward integration, in time instants x states
    :param Xi_b: initial condition on states for backward integration, in tim instants x stated
    :param t_sub: time-resolved time vector from initial to final instant of integration frame
    :param GP: dictionary containing matrix of coefficients Chi, linear and quadratic coeffs L,Q, among others
    :param flag_integration: flag indicating if integration is carried out in matrix form ('matrix') or einsum ('einsum')

    :return X_bf, t_bf: integrated states within t_bf
    """

    # Parameters
    nx = len(Xi_f)
    C, L, Q = GP['C'], GP['L'], GP['Q']
    Chi = GP['Chi']

    # Time vectors for the integration
    t_f = np.copy(t_sub)
    t_b = t_f[::-1]

    t_span_f = [t_f[0], t_f[-1]]
    t_span_b = [t_b[0], t_b[-1]]

    # Separate forward and backward integrations

    if flag_integration == 'matrix':
        sol_f = solve_ivp(lambda t,y: odeGalerkin_matrix(t,y,Chi), t_span_f, Xi_f, method='RK45', t_eval=t_f)
        sol_b = solve_ivp(lambda t,y: odeGalerkin_matrix(t,y,Chi), t_span_b, Xi_b, method='RK45', t_eval=t_b)

    elif flag_integration == 'einsum':
        sol_f = solve_ivp(lambda t,y: odeGalerkin_einsum(t,y,C,L,Q), t_span_f, Xi_f, method='RK45', t_eval=t_f)
        sol_b = solve_ivp(lambda t,y: odeGalerkin_einsum(t,y,C,L,Q), t_span_b, Xi_b, method='RK45', t_eval=t_b)

    # Retrieve solutions
    t_f_sol = sol_f.t
    X_f = sol_f.y.T
    t_b_sol = sol_b.t
    X_b = sol_b.y.T

    # If integration diverges and stops before, set states as null values
    if len(t_f_sol) < len(t_f):
        X_f = np.append(X_f, np.zeros((len(t_f) - len(t_f_sol), nx)), axis=0)
        t_f_sol = t_f
    if len(t_b_sol) < len(t_b):
        X_b = np.append(X_b, np.zeros((len(t_b) - len(t_b_sol), nx)), axis=0)
        t_b_sol = t_b

    # Weighting parameter
    nt = len(t_f)
    lim = 10
    weight = np.linspace(-lim, lim, nt - 2)
    sig = (1 / (1 + np.e ** (-weight))).reshape(-1, 1)

    # Weight solutions
    X_b = np.flip(X_b, axis=0)

    X_bf = np.zeros((nt, nx))
    X_bf[0, :] = X_f[0, :]
    X_bf[-1, :] = X_b[-1, :]

    X_weighted = np.multiply(X_f[1:-1, :], (1 - sig)) + np.multiply(X_b[1:-1, :], sig)
    X_bf[1:-1, :] = X_weighted

    logger.debug("Completed integration of " + str(i+1) + "/" + str(ni-1) + " time frames")

    return X_bf, t_f

def integrator(Xi, ti, Dt, GP, Phi, flag_integration, N_process):
    """
    Performs a physically-informed integration in between available snapshots. Leverages the dynamical system retrieved
    from Galerkin Proejections

    :param Xi: initial conditions of states, corresponding to POD temporal coefficients projected from available NTR snapshots
    :param ti: time instants corresponding to initial conditions
    :param Dt: time separation objective (time-resolved)
    :param Phi: spatial truncated POD modes
    :param GP: dictionary containing matrix of coefficients Chi, linear and quadratic coeffs L,Q, among others
    :param flag_integration: flag indicating if integration is carried out in matrix form ('matrix') or einsum ('einsum')
    :param N_process: number of cores used for parallel integration

    :return test_GP: dictionary containing states of the system with temporal resolution, as well as snapshot matrix and time vector
    """

    # Parameters
    tol = 1e-5
    ni = len(ti)
    nx = np.shape(Xi)[1]

    # Create time-resolved time vector
    t = np.array([ti[0]])
    for i in range(ni-1):
        t_sub = np.arange(ti[i], ti[i+1] + Dt*tol, Dt)
        t = np.concatenate((t, t_sub[1:]))
    nt = len(t)

    # Initialize time-resolved coordinates
    X = np.zeros((nt, nx))
    X[0, :] = Xi[0, :]

    # Non-parallelized integration
    if N_process == 1:
        logger.info("Non-parallel integration")
        for i in range(ni-1):

            t_bf = np.arange(ti[i], ti[i + 1] + Dt * tol, Dt)
            X_bf = BFI(i, ni, Xi[i,:], Xi[i+1,:], t_bf, GP, flag_integration)[0]

            it0 = np.where(np.abs(ti[i] - t) < tol)[0][0]
            itf = np.where(np.abs(ti[i+1] - t) < tol)[0][0]
            X[it0+1:itf+1, :] = X_bf[1:, :]


    # Parallelized integration with multiple processes
    else:
        logger.info("Parallel integration")
        with Pool(processes=N_process) as pool:
            # issue multiple tasks each with multiple arguments
            results = [pool.apply_async(BFI, args=(i, ni, Xi[i,:], Xi[i+1,:],
                       np.arange(ti[i], ti[i + 1] + Dt * tol, Dt), GP,
                       flag_integration)).get() for i in range(ni-1)]

        for result in results:
            X_bf, t_bf = result

            it0 = np.where(np.abs(t_bf[0] - t) < tol)[0][0]
            itf = np.where(np.abs(t_bf[-1] - t) < tol)[0][0]
            X[it0 + 1:itf+1, :] = X_bf[1:, :]

    Ddt = np.dot(Phi, X.T)
    test_GP = {'t': t, 'X': X, 'Ddt': Ddt}

    return test_GP

def interpolator(Xi, ti, Dt, Phi):
    """
    Interpolates states of the system in between available ICs using a cubic spline function

    :param Xi: initial conditions of states, corresponding to POD temporal coefficients projected from available NTR snapshots
    :param ti: time instants corresponding to initial conditions
    :param Dt: time separation objective (time-resolved)
    :param Phi: spatial truncated POD modes

    :return test_interp: dictionary containing states of the system with temporal resolution, as well as snapshot matrix and time vector
    """

    # Parameters
    tol = 1e-3
    ni = len(ti)
    nx = np.shape(Xi)[1]

    # Create time-resolved time vector
    t = np.array([ti[0]])
    for i in range(ni-1):
        t_sub = np.arange(ti[i], ti[i+1] + Dt*tol, Dt)
        t = np.concatenate((t, t_sub[1:]))
    nt = len(t)

    # Initialize time-resolved coordinates
    X = np.zeros((nt, nx))

    # Spline interpolation process
    for i in range(nx):
        X[:,i] = CubicSpline(ti, Xi[:,i])(t)

    Ddt = np.dot(Phi, X.T)
    test_interp = {'t': t, 'X': X, 'Ddt': Ddt}

    return test_interp
