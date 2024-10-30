# PACKAGES
import numpy as np

def diff_1st_2D(grid,D):
    """
    1st order differentiation in space

    :param grid: dictionary containing X, Y grids
    :param D: 2D snapshot matrix (N_v, N_t)
    :return: 1st order gradient of D (Dx, Dy)
    """

    # Parameters and initialization
    X = grid['X']
    Y = grid['Y']

    N_y, N_x = np.shape(X)
    N_t = np.shape(D)[1]

    dx = np.abs(X[0, 1] - X[0, 0])
    dy = np.abs(Y[0, 0] - Y[1, 0])

    U = np.reshape(D[0:N_x * N_y,:], (N_y, N_x, N_t), order='F')
    V = np.reshape(D[N_x * N_y:2 * N_x * N_y, :], (N_y, N_x, N_t), order='F')

    Ux, Uy, Vx, Vy = np.zeros((4, N_y, N_x, N_t))

    # Gradients
    Ux[:, 1:-1, :] = (U[:, 2:, :] - U[:, 0:-2, :]) / (2 * dx)
    Ux[:, 0, :] = (-3 * U[:, 0, :] + 4 * U[:, 1, :] - U[:, 2, :]) / (2 * dx)
    Ux[:, -1, :] = (3 * U[:, -1, :] - 4 * U[:, -2, :] + U[:, -3, :]) / (2 * dx)

    Uy[1:-1, :, :] = (U[2:, :, :] - U[0:-2, :, :]) / (2 * dy)
    Uy[0, :, :] = (-3 * U[0, :, :] + 4 * U[1, :, :] - U[2, :, :]) / (2 * dy)
    Uy[-1, :, :] = (3 * U[-1, :, :] - 4 * U[-2, :, :] + U[-3, :, :]) / (2 * dy)

    Vx[:, 1:-1, :] = (V[:, 2:, :] - V[:, 0:-2, :]) / (2 * dx)
    Vx[:, 0, :] = (-3 * V[:, 0, :] + 4 * V[:, 1, :] - V[:, 2, :]) / (2 * dx)
    Vx[:, -1, :] = (3 * V[:, -1, :] - 4 * V[:, -2, :] + V[:, -3, :]) / (2 * dx)

    Vy[1:-1, :, :] = (V[2:, :, :] - V[0:-2, :, :]) / (2 * dy)
    Vy[0, :, :] = (-3 * V[0, :, :] + 4 * V[1, :, :] - V[2, :, :]) / (2 * dy)
    Vy[-1, :, :] = (3 * V[-1, :, :] - 4 * V[-2, :, :] + V[-3, :, :]) / (2 * dy)

    # Set to null boundary regions
    imask = np.where(grid['B'] == 1)
    Ux[imask[0], imask[1], :] = 0
    Uy[imask[0], imask[1], :] = 0
    Vx[imask[0], imask[1], :] = 0
    Vy[imask[0], imask[1], :] = 0

    # Reshape
    Dx = np.concatenate((np.reshape(Ux, (N_x * N_y, N_t), order='F'), np.reshape(Vx, (N_x * N_y, N_t), order='F')), axis=0)
    Dy = np.concatenate((np.reshape(Uy, (N_x * N_y, N_t), order='F'), np.reshape(Vy, (N_x * N_y, N_t), order='F')), axis=0)

    return Dx, Dy

def diff_time(D,t):
    """
    Differentiation in time

    :param D: snapshot matrix (N_v, N_t)
    :param t: time array
    :return: time derivative of D
    """

    # Parameters and initialization
    dt = t[1] - t[0]
    N_t = np.shape(D)[1]
    dDdt = np.zeros(np.shape(D))

    # Central difference
    dDdt[:, 1:-1] = (D[:, 2:] - D[:, 0:-2]) / (2 * dt)

    # Forward difference
    dDdt[:, 0] = ( - 3*D[:, 0] + 4*D[:, 1] - D[:, 2] ) / (2 * dt)

    # Backward difference
    dDdt[:, -1] = -(- 3 * D[:, -1] + 4 * D[:, -2] - D[:, -3]) / (2 * dt)

    return dDdt

def get_2Dvorticity(grid,D):
    """
    Planar vorticity

    :param grid: dictionary containing X, Y grids
    :param D: snapshot matrix (N_v, N_t)
    :return: vorticity out-of-plane component
    """

    # Parameters
    X = grid['X']
    Y = grid['Y']

    N_y, N_x = np.shape(X)

    # Gradients
    Dx, Dy = diff_1st_2D(grid, D)

    Vx = Dx[N_x * N_y:2 * N_x * N_y, :]
    Uy = Dy[0:N_x * N_y, :]

    # Planar vorticity
    w = Vx - Uy

    return w

def get_laplacian_2D(grid,D):
    """
    Laplacian of 2D flow

    :param grid: dictionary containing X, Y grids
    :param D: 2D snapshot matrix (N_v, N_t)
    :return: Dxx + Dyy
    """

    # Parameters
    X = grid['X']
    Y = grid['Y']

    N_y, N_x = np.shape(X)
    N_t = np.shape(D)[1]

    dx = np.abs(X[0, 1] - X[0, 0])
    dy = np.abs(Y[0, 0] - Y[1, 0])

    # Initialization
    Uxx = np.zeros((N_y, N_x, N_t))
    Uyy = np.zeros((N_y, N_x, N_t))
    Vxx = np.zeros((N_y, N_x, N_t))
    Vyy = np.zeros((N_y, N_x, N_t))

    U = np.reshape(D[0:N_x * N_y,:], (N_y, N_x, N_t), order='F')
    V = np.reshape(D[N_x * N_y:2 * N_x * N_y, :], (N_y, N_x, N_t), order='F')

    # Gradients
    Uxx[:, 1:-1, :] = (U[:, 2:, :] - 2 * U[:, 1:-1, :] + U[:, 0:-2, :]) / dx ** 2
    Uxx[:, 0, :] = (2 * U[:, 0, :] - 5 * U[:, 1, :] + 4 * U[:, 2, :] - U[:, 3, :]) / dx ** 3
    Uxx[:, -1, :] = (2 * U[:, -1, :] - 5 * U[:, -2, :] + 4 * U[:, -3, :] - U[:, -4, :]) / dx ** 3

    Uyy[1:-1, :, :] = (U[2:, :, :] - 2 * U[1:-1, :, :] + U[0:-2, :, :]) / dy ** 2
    Uyy[0, :, :] = (2 * U[0, :, :] - 5 * U[1, :, :] + 4 * U[2, :, :] - U[3, :, :]) / dy ** 3
    Uyy[-1, :, :] = (2 * U[-1, :, :] - 5 * U[-2, :, :] + 4 * U[-3, :, :] - U[-4, :, :]) / dy ** 3

    Vxx[:, 1:-1, :] = (V[:, 2:, :] - 2 * V[:, 1:-1, :] + V[:, 0:-2, :]) / dx ** 2
    Vxx[:, 0, :] = (2 * V[:, 0, :] - 5 * V[:, 1, :] + 4 * V[:, 2, :] - V[:, 3, :]) / dx ** 3
    Vxx[:, -1, :] = (2 * V[:, -1, :] - 5 * V[:, -2, :] + 4 * V[:, -3, :] - V[:, -4, :]) / dx ** 3

    Vyy[1:-1, :, :] = (V[2:, :, :] - 2 * V[1:-1, :, :] + V[0:-2, :, :]) / dy ** 2
    Vyy[0, :, :] = (2 * V[0, :, :] - 5 * V[1, :, :] + 4 * V[2, :, :] - V[3, :, :]) / dy ** 3
    Vyy[-1, :, :] = (2 * V[-1, :, :] - 5 * V[-2, :, :] + 4 * V[-3, :, :] - V[-4, :, :]) / dy ** 3

    # Set to null boundary regions
    imask = np.where(grid['B'] == 1)
    Uxx[imask[0], imask[1], :] = 0
    Uyy[imask[0], imask[1], :] = 0
    Vxx[imask[0], imask[1], :] = 0
    Vyy[imask[0], imask[1], :] = 0

    # Laplacian
    Dxx = np.concatenate((np.reshape(Uxx, (N_x * N_y, N_t), order='F'), np.reshape(Vxx, (N_x * N_y, N_t), order='F')), axis=0)
    Dyy = np.concatenate((np.reshape(Uyy, (N_x * N_y, N_t), order='F'), np.reshape(Vyy, (N_x * N_y, N_t), order='F')), axis=0)

    return Dxx + Dyy

def get_divergence_2D(grid,D):
    """
    Divergence of 2D flow

    :param grid: dictionary containing X, Y grids
    :param D: 2D snapshot matrix (N_v, N_t)
    :return: Ux + Vy
    """

    # Parameters
    X = grid['X']
    Y = grid['Y']

    N_y, N_x = np.shape(X)
    N_t = np.shape(D)[1]

    dx = np.abs(X[0, 0] - X[0, 1])
    dy = np.abs(Y[0, 0] - Y[1, 0])

    # Initialization
    Ux = np.zeros((N_y, N_x, N_t))
    Vy = np.zeros((N_y, N_x, N_t))

    U = np.reshape(D[0:N_x * N_y,:], (N_y, N_x, N_t), order='F')
    V = np.reshape(D[N_x * N_y:2 * N_x * N_y, :], (N_y, N_x, N_t), order='F')

    # Gradients
    Ux[:, 1:-1, :] = (U[:, 2:, :] - U[:, 0:-2, :]) / (2 * dx)
    Ux[:, 0, :] = (-3 * U[:, 0, :] + 4 * U[:, 1, :] - U[:, 2, :]) / (2 * dx)
    Ux[:, -1, :] = (3 * U[:, -1, :] - 4 * U[:, -2, :] + U[:, -3, :]) / (2 * dx)

    Vy[1:-1, :, :] = (V[2:, :, :] - V[0:-2, :, :]) / (2 * dy)
    Vy[0, :, :] = (-3 * V[0, :, :] + 4 * V[1, :, :] - V[2, :, :]) / (2 * dy)
    Vy[-1, :, :] = (3 * V[-1, :, :] - 4 * V[-2, :, :] + V[-3, :, :]) / (2 * dy)

    # Set to null boundary regions
    imask = np.where(grid['B'] == 1)
    Ux[imask[0], imask[1], :] = 0
    Vy[imask[0], imask[1], :] = 0

    # Divergence
    Div = np.reshape(Ux + Vy, (N_x * N_y, N_t), order='F')

    return Div