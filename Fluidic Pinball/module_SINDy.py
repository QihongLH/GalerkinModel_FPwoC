# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:11:09 2023

@author: QihongLi
"""

# Import packages
import numpy as np

def  poolpolyData(X,r,polyorder):
    """
    %%                          poolpolyData.m
    %--------------------------------------------------------------------------
    %
    % This function obtains the library of functions Theta from the regression 
    % fit dyin/dt = Theta(yin)*Xi, where yin denotes the states of the system. The
    % library Theta will be composed of a set of polynomials of order polyorder
    %
    % INPUTS
    %
    %   X         : matrix containing states of the system for a certain
    %               timespan of dimension Nt x r
    %   r        : number of states of the system
    %   polyorder : array containing polynomial orders span by the library
    %
    % OUTPUT
    %
    %   Theta     : library of functions of dimension Nt x Nf
    %
    %--------------------------------------------------------------------------
    %
    % Copyright 2015, All Rights Reserved
    % Code by Steven L. Brunton
    % For Paper, "Discovering Governing Equations from Data: 
    %        Sparse Identification of Nonlinear Dynamical Systems"
    % by S. L. Brunton, J. L. Proctor, and J. N. Kutz
    """


    Nt = X.shape[0]
    
    # Initialize Theta as a zeros matrix
    
    num = 1
    den = 1
    for i in range(0,polyorder):
        num = num*(r+i-1)
        den = den*(i+1)
        # ns = r*(10+1)*(10+2)*(10+3)/(2*3*4)
    ns = int(num/den)
    Theta = np.zeros([Nt, ns])
    
    ind = 0
    
    if polyorder == 0:
        # Theta =  np.zeros(Nt)
        Theta[:,ind] = np.ones(Nt)
        ind += 1
    
    if polyorder == 1:
        Theta =  np.zeros([Nt,r-1])
        for i in range(0,r-1):
            Theta[:,ind] = X[:,i]
            ind += 1
    
    if polyorder == 2:
        # Theta =  np.zeros([Nt,int(r*(r-1)/2)])
        for i in range(0,r-1):
            for j in range(i,r-1):
                Theta[:,ind] = X[:,i]*X[:,j]
                ind += 1

    if polyorder == 3:
        for i in range(0,r-1):
            for j in range(i,r-1):
                for k in range(j,r-1):
                    Theta[:,ind] = X[:,i]*X[:,j]*X[:,k]
                    ind += 1
                    
    if polyorder == 4:
        for i in range(0,r-1):
            for j in range(i,r-1):
                for k in range(j,r-1):
                    for l in range(k,r-1):
                       Theta[:,ind] = X[:,i]*X[:,j]*X[:,k]*X[:,l]
                       ind += 1
    
    if polyorder == 5:
        for i in range(0,r-1):
            for j in range(i,r-1):
                for k in range(j,r-1):
                    for l in range(k,r-1):
                        for m in range(l,r-1):
                            Theta[:,ind] = X[:,i]*X[:,j]*X[:,k]*X[:,l]*X[:,m]
                            ind += 1
    
    if polyorder == 6:
        for i in range(0,r-1):
            for j in range(i,r-1):
                for k in range(j,r-1):
                    for l in range(k,r-1):
                        for m in range(l,r-1):
                            for n in range(m,r-1):
                                Theta[:,ind] = X[:,i]*X[:,j]*X[:,k]*X[:,l]*X[:,m]*X[:,n]
                                ind += 1
                            
    if polyorder == 7:
        for i in range(0,r-1):
            for j in range(i,r-1):
                for k in range(j,r-1):
                    for l in range(k,r-1):
                        for m in range(l,r-1):
                            for n in range(m,r-1):
                                for o in range(n,r-1):
                                    Theta[:,ind] = X[:,i]*X[:,j]*X[:,k]*X[:,l]*X[:,m]*X[:,n]*X[:,o]
                                    ind += 1
                                
                                
    return Theta
                            
"""                       
    for polyorder in range(max_polyorder + 1):
        if polyorder == 0:
            Theta[:, ind] = np.ones(Nt)
            ind += 1
        else:
            for i in range(nVars-1):
                if polyorder == 1:
                    Theta[:, ind] = X[:, i]
                    ind += 1
                else:
                    for j in range(i, nVars-1):
                        if polyorder == 2:
                            Theta[:, ind] = X[:, i] * X[:, j]
                            ind += 1
                        else:
                            for k in range(j, nVars-1):
                                if polyorder == 3:
                                    Theta[:, ind] = X[:, i] * X[:, j] * X[:, k]
                                    ind += 1
                                else:
                                    for l in range(k, nVars-1):
                                        if polyorder == 4:
                                            Theta[:, ind] = X[:, i] * X[:, j] * X[:, k] * X[:, l]
                                            ind += 1
                                        else:
                                            for m in range(l, nVars-1):
                                                if polyorder == 5:
                                                    Theta[:, ind] = X[:, i] * X[:, j] * X[:, k] * X[:, l] * X[:, m]
                                                    ind += 1
                                                else:
                                                    for Nt in range(m, nVars-1):
                                                        if polyorder == 6:
                                                            Theta[:, ind] = X[:, i] * X[:, j] * X[:, k] * X[:, l] * X[:, m] * X[:, Nt]
                                                            ind += 1
                                                        else:
                                                            for o in range(Nt, nVars-1):
                                                                if polyorder == 7:
                                                                    Theta[:, ind] = X[:, i] * X[:, j] * X[:, k] * X[:, l] * X[:, m] * X[:, Nt] * X[:, o]
                                                                    ind += 1
"""




