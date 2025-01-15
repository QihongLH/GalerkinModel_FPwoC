# PACKAGES
import numpy as np

# LOCAL FUNCTIONS
import modules.dynamics.differentiation as diff

def get_pgrad(grid, D, dDdt, Re):
    """
    Computes the pressure gradient term in the Navier-Stokes equations
    :param grid: Grid dictionary, with X and Y domains
    :param D: Snapshot matrix
    :param dDdt: Time derivative of the snapshot matrix
    :param Re: Reynolds number
    :return: pressure gradient term
    """
    X = grid['X']
    Y = grid['Y']

    m = X.shape[0]
    n = X.shape[1]

    U = D[0:m*n, :]
    V = D[m*n:2*m*n, :]

    # Get gradient
    Dx, Dy = diff.diff_1st_2D(grid, D)
    Ux = Dx[0:m*n, :]
    Uy = Dy[0:m*n, :]
    Vx = Dx[m*n:2*m*n, :]
    Vy = Dy[m*n:2*m*n, :]

    # Get Laplacian of the flow
    D2D = diff.get_laplacian_2D(grid, D)

    # Enforce divergence = DivD = 0 (incompressible flow)
    #DivD = get_divergence(grid, D)
    #DivD = np.concatenate((DivD, DivD), axis=0)
    # Get divergence of the dyadic product (convective term)
    #DXD = np.multiply(DivD, D) + np.concatenate((np.multiply(U,Ux) + np.multiply(V,Uy), np.multiply(U,Vx) + np.multiply(V,Vy)), axis=0)
    DXD = np.concatenate((np.multiply(U,Ux) + np.multiply(V,Uy), np.multiply(U,Vx) + np.multiply(V,Vy)), axis=0)

    DP = 1/Re*D2D - dDdt - DXD

    return DP