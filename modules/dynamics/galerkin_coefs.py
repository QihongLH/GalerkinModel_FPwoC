# PACKAGES
import numpy as np
import time
from loguru import logger

# LOCAL FUNCTIONS
import modules.dynamics.differentiation as diff
import modules.dynamics.system_eqs as system_eqs

def galerkin_coefs(grid, Phif, Phi0, Sigmaf, Psif, Re, flag_pressure, flag_integration, DP=[]):
    """
    Retrieves Galerkin-POD coefficients

    :param grid: dictionary containing X, Y grids and body mask B
    :param Phif: set of POD spatial modes of training set fluctuations, spatial points x modes
    :param Phi0: POD spatial mode of training set mean component (a0 = 1), spatial points x modes
    :param Sigmaf: set of POD singular values of training set fluctuations, modes x modes
    :param Psif: set of POD temporal modes of training set fluctuations, modes x time instants
    :param DP: snapshot matrix of pressure gradient, spatial points x time instants
    :param Re: Reynolds number of flow
    :param flag_pressure: flag indicating if no pressure coefficients are to be obtained ('none'), or else following the
    quadratic method ('analytical') or linear ('empirical')
    :param flag_integration: flag indicating if system of eqs is created through matmul ('matrix') or einsum ('einsum')

    :return GPcoef: dictionary containing matrix of coefficients or linear and quadratic coefficients
    """

    # Add 2nd dimension to mean spatial POD mode
    Phi0 = Phi0.reshape(-1, 1)

    # Retrieve linear-viscous and quadratic-convective coefficients
    Lv = linear_viscous(grid, Phif, Phi0)
    logger.debug("Retrieved linear viscous coefficients...")

    Qc = quadratic_convective(grid, Phif, Phi0)
    logger.debug("Retrieved quadratic convective coefficients...")

    # Retrieve pressure coefficients depending on flags
    if flag_pressure == 'analytical':
        Lp = np.zeros_like(Lv)
        Qp = quadratic_pressure(Phif, Phi0, Sigmaf, Psif, DP)
        logger.debug("Retrieved quadratic analytical pressure coefficients...")
    elif flag_pressure == 'empirical':
        Lp = linear_pressure(Phif, Phi0, Sigmaf, Psif, DP)
        Qp = np.zeros_like(Qc)
        logger.debug("Retrieved linear empirical pressure coefficients...")
    elif flag_pressure == 'none':
        Lp = np.zeros_like(Lv)
        Qp = np.zeros_like(Qc)

    C = Lp[1:,0] + 1 / Re * Lv[1:,0] + Qp[1:,0,0] + Qc[1:,0,0]
    L = Lp[1:,1:] + 1 / Re * Lv[1:,1:] + Qp[1:,0,1:] + Qc[1:,0,1:] + Qp[1:,1:,0] + Qc[1:,1:,0]
    Q = Qp[1:,1:,1:] + Qc[1:,1:,1:]
    GPcoef = {'C':[], 'L':[], 'Q':[], 'Chi':[]}
    GPcoef['Lv'] = Lv
    GPcoef['Lp'] = Lp
    GPcoef['Qc'] = Qc
    GPcoef['Qp'] = Qp

    if flag_integration == 'einsum':
        GPcoef['C'] = C
        GPcoef['L'] = L
        GPcoef['Q'] = Q

    elif flag_integration == 'matrix':
        # Get non-normalized temporal modes and library of functions up to 2nd order polynomials
        af = np.dot(np.diag(Sigmaf), Psif)
        nr = np.shape(Phif)[1]
        Theta = system_eqs.pool_polynomials(af.T)

        # Parameters and initialization of matrix coefficients
        nf = np.shape(Theta)[1]
        Chi = np.zeros((nf, nr))

        # Dynamics of fluctuating modes corresponding to constant functions
        Chi[0, :] = 1 / Re * Lv[1:, 0] + Qc[1:, 0, 0] + Qp[1:, 0, 0] + Lp[1:, 0]

        # Dynamics of fluctuating modes corresponding to linear functions
        for j in range(1, nr + 1):
            Chi[j, :] = 1 / Re * Lv[1:, j] + Qc[1:, 0, j] + Qp[1:, 0, j] + Qc[1:, j, 0] + Qp[1:, j, 0] + Lp[1:, j]

        # Dynamics of fluctuating modes corresponding to quadratic functions
        count = 0
        for i in range(1, nr + 1):
            for j in range(i, nr + 1):
                if j == i:
                    Chi[count + nr + 1, :] = Qc[1:, i, j] + Qp[1:, i, j]
                else:
                    Chi[count + nr + 1, :] = Qc[1:, i, j] + Qp[1:, i, j] + Qc[1:, j, i] + Qp[1:, j, i]
                count += 1

        GPcoef['Chi'] = Chi
        logger.debug("Re-organized matrix of coefficients...")

    return GPcoef

def sparsify_coeffs(Sigmaf, Psif, dPsif, Chi, tol):
    """
    Sets to null relative values of matrix coefficients below a certain tolerance

    :param Sigmaf: set of POD singular values of training set fluctuations, modes x modes
    :param Psif: set of POD temporal modes of training set fluctuations, modes x time instants
    :param Psif: set of POD temporal mode derivatives of training set fluctuations, modes x time instants
    :param Chi: matrix of coefficients, function x modes
    :param tol: tolerance thresholding value

    :return Chi: sparsified matrix of coefficients, function x modes
    """

    if type(dPsif) != type(np.array([])):
        raise Exception("No derivative of temporal modes available")

    # Get non-normalized temporal modes and derivatives
    af = np.dot(np.diag(Sigmaf), Psif)
    daf = np.dot(np.diag(Sigmaf), dPsif)

    # Library of functions up to 2nd order polynomials
    Theta = system_eqs.pool_polynomials(af.T)

    # Norm of library functions & derivatives
    normTheta = np.linalg.norm(Theta, axis=0)
    normdX = np.linalg.norm(daf.T, axis=0)

    # Normalize matrix of coefficients and threshold
    Chi_n = (Chi.T * normTheta).T / normdX
    Chi_n[np.where(np.abs(Chi_n) < tol)] = 0
    Chi[np.where(Chi_n == 0)] = 0

    return Chi

def linear_viscous(grid, Phif, Phi0):
    """
    Retrieves linear-viscous coefficients

    :param grid: dictionary containing X, Y grids
    :param Phif: set of POD spatial modes of training set fluctuations, spatial points x modes
    :param Phi0: POD spatial mode of training set mean component (a0 = 1), spatial points x modes

    :return Lv: linear viscous coefficients, modes + 1 x modes + 1
    """

    # Complete set of spatial modes
    Phi = np.concatenate((Phi0, Phif), axis=1)

    # 2D Laplacian of POD modes
    D2Phi = diff.get_laplacian_2D(grid, Phi)

    # Linear-viscous coefficients
    Lv = np.dot(Phi.T, D2Phi)
    return Lv

def quadratic_convective(grid, Phif, Phi0):
    """
    Retrieves quadratic-convective coefficients

    :param grid: dictionary containing X, Y grids
    :param Phif: set of POD spatial modes of training set fluctuations, spatial points x modes
    :param Phi0: POD spatial mode of training set mean component (a0 = 1), spatial points x modes

    :return Qc: quadratic convective coefficients, modes + 1 x modes + 1 x modes + 1
    """

    # Complete set of spatial modes
    Phi = np.concatenate((Phi0, Phif), axis=1)

    # Parameters
    X = grid['X']
    Y = grid['Y']

    m, n = X.shape
    nr = Phif.shape[1]

    # Horizontal and vertical components of spatial modes
    Phiu = Phi[0:m*n, :]
    Phiv = Phi[m*n:2*m*n, :]

    # Get gradient of spatial modes
    Phix, Phiy = diff.diff_1st_2D(grid, Phi)
    Phiux = Phix[0:m*n, :]
    Phiuy = Phiy[0:m*n, :]
    Phivx = Phix[m*n:2*m*n, :]
    Phivy = Phiy[m*n:2*m*n, :]

    # Enforce divergence = DivD = 0 (incompressible flow)
    #DivPhi = get_divergence(grid, Phi)
    #DivPhi = np.concatenate((DivPhi, DivPhi), axis=0)
    # Get divergence of the dyadic product (convective term) of spatial modes and coefficients
    Qc = np.zeros((nr+1,nr+1,nr+1))

    for j in range(nr+1):
        #DXD = np.multiply(DivPhi[:,j], Phi) + np.concatenate((np.multiply(Phiu[:,j],Phiux) + np.multiply(Phiv[:,j],Phiuy), np.multiply(Phiu[:,j],Phivx) + np.multiply(Phiv[:,j],Phivy)), axis=0)
        DXD = np.concatenate((np.multiply(Phiu[:,j].reshape(-1,1),Phiux) + np.multiply(Phiv[:,j].reshape(-1,1),Phiuy), np.multiply(Phiu[:,j].reshape(-1,1),Phivx) + np.multiply(Phiv[:,j].reshape(-1,1),Phivy)), axis=0)

        Qc[:,j,:] = np.dot(-Phi.T, DXD)

    return Qc

def quadratic_pressure(Phif, Phi0, Sigmaf, Psif, DP):
    """
    Retrieves quadratic-pressure coefficients

    :param Phif: set of POD spatial modes of training set fluctuations, spatial points x modes
    :param Phi0: POD spatial mode of training set mean component (a0 = 1), spatial points x modes
    :param Sigmaf: set of POD singular values of training set fluctuations, modes x modes
    :param Psif: set of POD temporal modes of training set fluctuations, modes x time instants
    :param DP: snapshot matrix of pressure gradient, spatial points x time instants

    :return Qp: quadratic pressure coefficients, modes + 1 x modes + 1 x modes + 1
    """

    # Complete set of spatial modes
    Phi = np.concatenate((Phi0, Phif), axis=1)

    # Complete set of temporal modes and singular values
    a0 = np.ones((1,np.shape(Psif)[1]))
    Psi = np.concatenate((a0/np.linalg.norm(a0), Psif), axis=0)
    Sigma = np.concatenate(([np.linalg.norm(a0)], Sigmaf))

    # Parameters
    nv, nr = np.shape(Phif)

    # Spatial modes associated to pressure gradient
    DPi = np.dot(DP, np.dot(Psi.T, np.linalg.inv(np.diag(Sigma))))
    DPj_plus = DPi[:,0].reshape((nv,1)) + DPi
    DPj_minus = DPi[:,0].reshape((nv,1)) - DPi

    # Recursive coefficient scheme
    Qi = np.dot(Phi.T, -DPi[:,0])

    Qij_plus = np.dot(Phi.T, -DPj_plus)
    Qij_minus = np.dot(Phi.T, -DPj_minus)

    Qijk_0 = np.zeros((nr+1, nr+1, nr+1))
    for j in range(nr+1):
        for k in range(j, nr+1):
            Qijk_0[:,j,k] = np.dot(Phi.T, -(DPi[:,0] + DPi[:,j] + DPi[:,k]))

    # Recursive scheme
    Qijk_pi = np.zeros((nr+1, nr+1, nr+1))

    Qijk_pi[:,0,0] = Qi

    Qijk_pi[:,0,1:] = Qij_plus[:,1:]/2 - Qij_minus[:,1:]/2

    for j in range(1, nr+1):
            Qijk_pi[:,j,j] = Qij_plus[:,j]/2 + Qij_minus[:,j]/2 - Qi[:]

    for j in range(1, nr+1):
        for k in range(j+1, nr+1):
            Qijk_pi[:,j,k] = Qijk_0[:,j,k] - Qijk_pi[:,0,0] - Qijk_pi[:,0,j] - Qijk_pi[:,0,k] - Qijk_pi[:,j,j] - Qijk_pi[:,k,k]

    return Qijk_pi

def linear_pressure(Phif, Phi0, Sigmaf, Psif, DP):
    """
    Retrieves linear-pressure (empirical) coefficients

    :param Phif: set of POD spatial modes of training set fluctuations, spatial points x modes
    :param Phi0: POD spatial mode of training set mean component (a0 = 1), spatial points x modes
    :param Sigmaf: set of POD singular values of training set fluctuations, modes x modes
    :param Psif: set of POD temporal modes of training set fluctuations, modes x time instants
    :param DP: snapshot matrix of pressure gradient, spatial points x time instants

    :return Lv: linear pressure coefficients, modes + 1 x modes + 1
    """

    # Complete set of spatial modes
    Phi = np.concatenate((Phi0, Phif), axis=1)

    # Complete set of temporal modes and singular values
    a0 = np.ones((1, np.shape(Psif)[1]))
    Psi = np.concatenate((a0 / np.linalg.norm(a0), Psif), axis=0)
    Sigma = np.concatenate(([np.linalg.norm(a0)], Sigmaf))
    A = np.dot(np.diag(Sigma), Psi).T

    # Solve linear system of equations
    LHS = np.dot(-Phi.T, DP).T
    Lp = np.dot(np.linalg.pinv(A), LHS).T

    return Lp
