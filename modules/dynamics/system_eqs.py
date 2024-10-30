# PACKAGES
import numpy as np

def odeGalerkin_einsum(t, X, C, L, Q):
    '''
    Function handle for integration of system states through np.einsum

    :param Chi: coefficient matrix for every system state
    :param X: matrix containing states of the system, time instants x states
    :param C: constant coefficients, modes
    :param L: linear coefficients, modes x modes
    :param Q: quadratic coefficients, modes x modes x modes

    :return: time derivative of the state
    '''

    return C + (L @ X) + np.einsum('jki,j,k->i', Q, X, X)

def odeGalerkin_matrix(t, X, Chi):
    '''
    Function handle for integration of system states through matrix multiplication and library of functions

    :param t: time parameter (required for integration)
    :param Chi: coefficient matrix for every system state
    :param X: matrix containing states of the system, time instants x states

    :return: time derivative of the state
    '''

    # Create library of functions
    Theta = pool_polynomials(X)

    dX = np.dot(Theta, Chi)

    return dX

def pool_polynomials(X,polyorder=np.array([0, 1, 2])):

    '''Obtains the library of functions Theta from the regression fit dyin/dt = Theta(yin)*Xi,
    where yin denotes the states of the system. The library Theta will be composed of a set of polynomals of order polyorder

    :param X: state of the system of dimension nt x nx
    :param polyorder: order of the polynomials

    :return: library of functions of dimension nt x nf

    Copyright 2015, All rights reserved
    Code by Steven L. Brunton
    For Paper, "Discovering Governing Equations from Data:  Sparse Identification of Nonlinear Dynamical Systems"
    by S. L. Brunton, J. L. Proctor, and J. N. Kutz'''

    if X.ndim == 1:
        X = X.reshape(1,-1)

    nt, nx = np.shape(X)
    Theta = np.empty((nt,1))
    if (np.where(polyorder == 0)[0]).size != 0:
        Theta = np.ones((nt,1))

    if (np.where(polyorder == 1)[0]).size != 0:
        Theta = np.append(Theta, X, axis=1)

    if (np.where(polyorder == 2)[0]).size != 0:
        for i in range(nx):
            Theta = np.append(Theta, np.multiply(X[:, i].reshape(-1,1), X[:, i:]), axis=1)

    return Theta