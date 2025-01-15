# PACKAGES
import numpy as np

# LOCAL PACKAGES
import modules.dynamics.system_eqs as system_eqs

def get_fom_modes(Xtrue, X, flag_FOM):
    """
    Obtains Figure of Merit for modes

    :param Xtrue: ground truth modes, in time instants x states
    :param X: reconstructed modes, in time instants x states
    :param flag_FOM: flag indicating the FOM type. 'R2d' for R2 determination coefficient,
    'R2c' for correlation coefficient, 'err_rel' for relative error, 'Sc' for cosine similarity,
    'de' for root mean square error

    :return: FOM, in states
    """

    if flag_FOM == 'R2d': # Determination coefficient
        Xmean = np.mean(Xtrue, axis=0)[None, :]
        FOM = 1 - np.sum((X - Xtrue) ** 2, axis=0) / np.sum((Xtrue - Xmean) ** 2, axis=0)
    elif flag_FOM == 'R2c': # Correlation coefficient
        FOM = np.sum(Xtrue * X, axis=0) ** 2 / (np.sum(Xtrue ** 2, axis=0) * np.sum(X ** 2, axis=0))
    elif flag_FOM == 'err_rel': # Mean relative error
        FOM = np.mean(np.abs((Xtrue - X) / Xtrue), axis=0)
    elif flag_FOM == 'Sc': # Cosine similarity
        FOM = np.sum(np.multiply(Xtrue, X), axis=0) / np.linalg.norm(Xtrue, axis=0) ** 2
    elif flag_FOM == 'de': # Root mean square error
        std_a = np.std(Xtrue, axis=0)
        FOM = 1 / std_a * np.sqrt(np.mean((Xtrue - X) ** 2, axis=0))
    else:
        raise Exception("FOM type is not included here. Please select ('R2d', 'R2c', 'err_rel', 'Sc' or 'de')")

    return FOM

def get_fit_galerkin_acc(a, da, Chi):
    """
    Computes R2 determination coefficient for temporal mode derivatives

    :param a: set of non-normalized temporal POD temporal modes of set fluctuations, time instants x modes
    :param da: set of non-normalized temporal POD temporal mode derivatives of set fluctuations, time instants x modes
    :param Chi: matrix of coefficients, function x modes

    :return R2d: R2 determination coefficient for temporal mode derivatives, in states
    """

    if type(da) != type(np.array([])):
        raise Exception("No derivative of temporal modes available")

    # Library of functions up to 2nd order polynomials
    Theta = system_eqs.pool_polynomials(a)

    # Derive temporal modes derivatives from Galerkin system
    da_GP = Theta @ Chi

    # Retrieve R2 determination coefficient for derivatives
    R2d = get_fom_modes(da, da_GP, 'R2d')

    return R2d

def get_active_terms(Chi):
    """
    Retrieves number of active terms in Galerkin system

    :param Chi: matrix of coefficients, function x modes

    :return: number of constant, linear and quadratic
    """

    nr = np.shape(Chi)[1]

    n_c = np.sum((Chi[0:1] != 0), axis=0)
    n_l = np.sum((Chi[1:nr + 1] != 0), axis=0)
    n_q = np.sum((Chi[nr + 1:] != 0), axis=0)

    return n_c, n_l, n_q