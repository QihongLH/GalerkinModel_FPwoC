# PACKAGES
import numpy as np

def get_reyn_stresses_2D(grid, Ddt):
    """
    Retrieves Reynold stresses for 2D flow

    :param grid: dictionary containing X, Y grids
    :param Ddt: snapshot matrix of fluctuations of velocity

    :return REYN: dictionary containing TKE and Reynolds stresses
    """

    X = grid['X']

    ny, nx = np.shape(X)

    U = Ddt[0: nx * ny, :]
    V = Ddt[nx * ny:nx * ny * 2, :]

    REYN = {}
    REYN['uu'] = np.mean(U ** 2, axis=1)
    REYN['uv'] = np.mean(U * V, axis=1)
    REYN['vv'] = np.mean(V ** 2, axis=1)
    REYN['TKE'] = 1 / 2 * (REYN['uu'] + REYN['vv'])

    return REYN

def get_MSE(Dtrue, D, B, var_true, flag_type):
    """
    Estimates Mean Square Error (normalized with variance of ground truth flow) for snapshot matrix

    :param Dtrue: ground truth of snapshot matrix, spatial points x time instants
    :param D: reconstructed snapshot matrix, spatial points x time instants
    :param B: mask grid (1 if body, 0 otherwise)
    :param flag_type: 'W' if whole error, 'S' if spatial, 'T' if temporal
    :param var_true: variance of ground truth flow (typically std_D ** 2 or (std_u ** 2 + std_v ** 2))

    :return: MSE
    """

    # Parameters
    ny, nx = np.shape(B)

    # Get data outside mask
    B = np.reshape(B, (ny * nx), order='F')
    if np.shape(D)[0] == (ny * nx):
        i_nonmask = np.where(np.isnan(B))
    elif np.shape(D)[0] == 2 * (ny * nx):
        i_nonmask = np.where(np.isnan(np.concatenate((B, B))))
    else:
        i_nonmask = np.where(np.isnan(np.concatenate((B,B,B))))

    if flag_type != 'S':
        Xtrue = Dtrue[i_nonmask, :][0,:,:]
        X = D[i_nonmask, :][0,:,:]
    else:
        Xtrue, X = np.copy(Dtrue), np.copy(D)

    # Compute temporal (T), spatial (S) or whole (W) error
    if flag_type == 'T':
        MSE = np.mean((Xtrue - X)**2, axis=0) / var_true
    elif flag_type == 'S':
        MSE = np.mean((Xtrue - X) ** 2, axis=1)
        MSE = (MSE[:nx * ny] + MSE[nx * ny:]) / var_true
    elif flag_type == 'W':
        MSE = np.mean((Xtrue - X) ** 2) / var_true

    return MSE

def errort2subsampling(error_t, ts, Dt):
    """
    Transforms error evolution in time to mean error per instant within integration window

    :param error_t: temporal error or accuracy metric
    :param ts: time separation of ICs prior to integration
    :param Dt: time resolution of integration

    :return error_fom: subsampling mean error
    """

    # Number of snapshots in integration window
    N = int(ts // Dt) + 1

    error_fom = np.zeros((N))
    for i in range(N):
        error_fom[i] = np.mean(error_t[i::N-1])

    return error_fom

def get_cos_similarity(Dtrue, D, B):
    """
    Estimates temporal cosine similarity Sc for snapshot matrix

    :param Dtrue: ground truth of snapshot matrix, spatial points x time instants
    :param D: reconstructed snapshot matrix, spatial points x time instants
    :param B: mask grid (1 if body, 0 otherwise)

    :return Sc: cosine similarity
    """

    # Parameters
    ny, nx = np.shape(B)
    nt = np.shape(Dtrue)[1]

    # Get data outside mask
    B = np.reshape(B, (ny * nx), order='F')
    if np.shape(D)[0] == (ny * nx):
        i_nonmask = np.where(np.isnan(B))
    elif np.shape(D)[0] == 2 * (ny * nx):
        i_nonmask = np.where(np.isnan(np.concatenate((B, B))))
    else:
        i_nonmask = np.where(np.isnan(np.concatenate((B,B,B))))

    Xtrue = Dtrue[i_nonmask, :][0,:,:]
    X = D[i_nonmask, :][0,:,:]

    # Cosine similarity
    Sc = np.sum(np.multiply(Xtrue, X), axis=0) / np.linalg.norm(Xtrue, axis=0) ** 2

    return Sc




