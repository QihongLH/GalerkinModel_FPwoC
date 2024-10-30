import numpy as np
from scipy import integrate
from loguru import logger

def get_POD(D, dDdt=[]):
    """
    Retrieves POD complete basis

    :param D: snapshot matrix, spatial points x time instants
    :param dDdt: temporal derivative of snapshot matrix, spatial points x time instants
    :param DP: pressure gradient of snapshot matrix, spatial points x time instants

    :return POD: complete dictionary with spatial, temporal modes and singular values
    """

    # POD
    Phi, Sigma, Psi = np.linalg.svd(D, full_matrices=False)

    POD = {'Phi': Phi, 'Sigma': Sigma, 'Psi': Psi, 'dPsi':[]}

    if type(dDdt) == type(np.array([])):
        dPsi = np.dot(np.dot(np.linalg.inv(np.diag(Sigma)), Phi.T), dDdt)
        POD['dPsi'] = dPsi

    return POD

def truncate_POD(POD, r_method, r_threshold):
    """
    Retrieves POD  truncated basis

    :param POD: complete dictionary with spatial, temporal modes and singular values
    :param r_method: flag to select truncation method
    :param r_threshold: corresponding threshold value for given truncation method

    :return PODr: truncated dictionary with spatial, temporal modes and singular values
    """

    # PARAMETERS
    Phi, Sigma, Psi = POD['Phi'], POD['Sigma'], POD['Psi']
    dPsi = POD['dPsi']
    nt = Psi.shape[1]

    # TRUNCATION METHOD
    if r_method == 'energy':
        nr = energy_truncation(Sigma, r_threshold)
    elif r_method == 'elbow':
        E = np.cumsum(Sigma**2) / np.sum(Sigma**2)
        nr = elbow_fit(np.arange(1, nt+1), E)
    elif r_method == 'optimal':
        nr = optht(nt/Phi.shape[0], Sigma, sigma=None)
    elif r_method == 'manual':
        nr = r_threshold
    else:
        nr = nt

    # Truncate basis (note that nr is number of modes)
    Phir = Phi[:, 0:nr]
    Sigmar = Sigma[0:nr]
    Psir = Psi[0:nr, :]

    PODr = {'Phi': Phir, 'Sigma': Sigmar, 'Psi': Psir, 'dPsi': []}

    if type(dPsi) == type(np.array([])):
        dPsir = dPsi[0:nr, :]
        PODr['dPsi'] = dPsir

    Er = np.sum(Sigma[:nr] ** 2) / np.sum(Sigma ** 2)
    logger.debug("Truncated POD basis to " + str(nr) + " modes and " + "{:.2f}".format(Er*100) + "% of energy")

    return PODr

def project_POD(test, POD):
    """
    Updates flow dictionary to include temporal POD coefficients from projection onto spatial POD modes from training set

    :param test: dictionary containing among others, snapshot matrix Ddt, in spatial points x time instants
    :param POD: dictionary containing among others, spatial POD modes, in spatial points x modes

    :return test: updated dictionary with temporal modes a, in time instants x modes
    """

    Phi = POD['Phi']
    Ddt = test['Ddt']
    dDdt = test['dDdt']

    test['a'] = np.dot(Phi.T, Ddt).T
    test['da'] = []
    if type(dDdt) == type(np.array([])):
        test['da'] = np.dot(Phi.T, dDdt).T

    return test

def energy_truncation(S,E):
    """
    Finds the threshold vector index for reaching a certain level of energy

    :param S: singular value matrix diagonal
    :param E: energy level

    :return: truncation position in number of modes (index + 1)
    """

    energy = np.cumsum(S**2)/np.sum(S**2) # Cumulative energy up to each mode
    r = np.where(energy >= E)[0][0] # Find the index of the first mode that satisfies the energy level

    return r + 1

def elbow_fit(x,y):

    """
    Finds the elbow of the (x,y) curve in the x-position (i.e. the index value).
    Follows procedure of the Brindise&Vlachos (2017) in "Proper orthogonal decomposition truncation method
    for data denoising and order reduction"

    :param x: array of x values
    :param y: array of y values

    :return: elbow position in x (index + 1)
    """

    ny = len(y)
    R2 = np.zeros(ny)
    err = np.zeros(ny)


    for i in range(ny):

        if i == 0:
            y1 = y[0:(i+1)]
        else:
            c1 = np.polyfit(x[0:(i+1)], y[0:(i+1)], 1)
            y1 = np.polyval(c1, x[0:(i + 1)])
        if i == ny-1:
            y2 = y[(i+1):]
        else:
            c2 = np.polyfit(x[(i+1):], y[(i+1):], 1)
            y2 = np.polyval(c2, x[(i+1):])

        yt = np.concatenate((y1, y2)).reshape(ny, 1)

        R2[i] = 1 - np.sum( ( yt - y.reshape(ny, 1) )**2 ) / np.sum( ( y.reshape(ny, 1) - np.mean(y.reshape(ny, 1)) )**2  )
        err[i] = np.sqrt( np.sum( ( yt - y.reshape(ny, 1) )**2 )/ny )

    # Find the elbow
    elbow = np.argmax(R2/err) + 1

    return elbow

def optht(beta, sv, sigma=None):
    """Compute optimal hard threshold for singular values.

    Off-the-shelf method for determining the optimal singular value truncation
    (hard threshold) for matrix denoising.

    The method gives the optimal location both in the case of the known or
    unknown noise level.

    Parameters
    ---------
    beta : scalar or array_like
        Scalar determining the aspect ratio of a matrix, i.e., ``beta = m/n``,
        where ``m >= n``.  Instead the input matrix can be provided and the
        aspect ratio is determined automatically.

    sv : array_like
        The singular values for the given input matrix.

    sigma : real, optional
        Noise level if known.

    Returns
    -------
    k : int
        Optimal target rank.

    Notes
    -----
    Code is adapted from Matan Gavish and David Donoho, see [1]_.

    References
    ----------
     [1] Gavish, Matan, and David L. Donoho.
       "The optimal hard threshold for singular values is 4/sqrt(3)"
        IEEE Transactions on Information Theory 60.8 (2014): 5040-5053.
        http://arxiv.org/abs/1305.5870
    """
    # Compute aspect ratio of the input matrix
    if isinstance(beta, np.ndarray):
        m = min(beta.shape)
        n = max(beta.shape)
        beta = m / n

    # Check ``beta``
    if beta < 0 or beta > 1:
        raise ValueError('Parameter `beta` must be in (0,1].')

    if sigma is None:
        # Approximate ``w(beta)``
        coef_approx = _optimal_SVHT_coef_sigma_unknown(beta)
        # Compute the optimal ``w(beta)``
        coef = (_optimal_SVHT_coef_sigma_known(beta)
                / np.sqrt(_median_marcenko_pastur(beta)))
        # Compute cutoff
        cutoff = coef * np.median(sv)
    else:
        # Compute optimal ``w(beta)``
        coef = _optimal_SVHT_coef_sigma_known(beta)
        # Compute cutoff
        cutoff = coef * np.sqrt(len(sv)) * sigma
    # Compute and return rank
    greater_than_cutoff = np.where(sv > cutoff)
    if greater_than_cutoff[0].size > 0:
        k = np.max(greater_than_cutoff) + 1
    else:
        k = 0
    return k

def _optimal_SVHT_coef_sigma_known(beta):
    """Implement Equation (11)."""
    return np.sqrt(2 * (beta + 1) + (8 * beta)
                   / (beta + 1 + np.sqrt(beta**2 + 14 * beta + 1)))
def _optimal_SVHT_coef_sigma_unknown(beta):
    """Implement Equation (5)."""
    return 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
def _mar_pas(x, topSpec, botSpec, beta):
    """Implement Marcenko-Pastur distribution."""
    if (topSpec - x) * (x - botSpec) > 0:
        return np.sqrt((topSpec - x) *
                       (x - botSpec)) / (beta * x) / (2 * np.pi)
    else:
        return 0
def _median_marcenko_pastur(beta):
    """Compute median of Marcenko-Pastur distribution."""
    botSpec = lobnd = (1 - np.sqrt(beta))**2
    topSpec = hibnd = (1 + np.sqrt(beta))**2
    change = 1

    while change & ((hibnd - lobnd) > .001):
        change = 0
        x = np.linspace(lobnd, hibnd, 10)
        y = np.zeros_like(x)
        for i in range(len(x)):
            yi, err = integrate.quad(
                _mar_pas,
                a=x[i],
                b=topSpec,
                args=(topSpec, botSpec, beta),
            )
            y[i] = 1.0 - yi

        if np.any(y < 0.5):
            lobnd = np.max(x[y < 0.5])
            change = 1

        if np.any(y > 0.5):
            hibnd = np.min(x[y > 0.5])
            change = 1

    return (hibnd + lobnd) / 2.