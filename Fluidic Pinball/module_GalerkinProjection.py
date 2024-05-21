# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 18:00:15 2023

@author: QihongLi
"""

import numpy as np
import module_utility as mu
import module_SINDy as ms
from memory_profiler import profile



#%%
def galerkin_model(a, L, Q):

    """RHS of POD-Galerkin model, for time integration"""
    return (L @ a) + np.einsum('jki,j,k->i', Q, a, a)

@profile
def getGalerkinCoeffs(grid,PODr,NTRset,param):
    r = param['r']
    m = param['m']
    n = param['n']
    Phir = PODr['Phir']
    # Get Laplacian and gradients of each mode
    der = {}
    D2Ur = np.zeros([m*n*2,r])
    DUdx = np.zeros(Phir.shape)
    DUdy = np.zeros(Phir.shape)
    
    
    for i in range(0,r):
        # Add second dimension to arrays Phir and Psir 
        Phiu = Phir[0:m*n,i].reshape([m,n]).T # ,order='F'
        Phiv = Phir[m*n:,i].reshape([m,n]).T # ,order='F'
        
        
        # get Laplacian
        D2u = mu.Laplacian2D(Phiu,grid['dx'],grid['dy']).reshape([n*m,],order='F')
        D2v = mu.Laplacian2D(Phiv,grid['dx'],grid['dy']).reshape([n*m,],order='F')
        D2Ur[:,i] = np.concatenate([D2u,D2v],axis=0)
        
        # get Gradient
        dudx,dudy = mu.Gradient_2D(Phiu,grid['dx'],grid['dy'])
        dvdx,dvdy = mu.Gradient_2D(Phiv,grid['dx'],grid['dy'])
        DUdx[:,i] = np.concatenate([dudx.reshape([m*n,],order='F'),dvdx.reshape([m*n,],order='F')],axis=0)
        DUdy[:,i] = np.concatenate([dudy.reshape([m*n,],order='F'),dvdy.reshape([m*n,],order='F')],axis=0)
        
    der['DUdx'] = DUdx # Gradient with respect to x
    der['DUdy'] = DUdy # Gradient with respect to y
    der['D2Ur'] = D2Ur # Laplacian
        
    GPcoeff = {}
    GPcoeff['l'] = get_l(Phir,D2Ur,param['nu'],r)
    GPcoeff['qc'],GPcoeff['qc_u'] = get_qc(Phir,DUdx,DUdy,r,param)
    
    dudx,dudy = mu.Gradient_2D(Phiu,grid['dx'],grid['dy'])
    dpdx,dpdy = mu.GradientP_2D(NTRset,param,der,grid)
    
    Dp0 = np.concatenate([dpdx,dpdy],axis=0)
    
    GPcoeff['lp'] = get_qp_emp(NTRset['uf'],PODr,Dp0,r)
        
    return GPcoeff

# Get viscous coefficients
def get_l(Phir,D2Ur,nu,r):
    lc = np.zeros([r,r])
    for i in range(0,r):
        for j in range(0,r):
            lc[i,j] = Phir[:,i].T@D2Ur[:,j]*nu
    return lc
    

# Get convective coefficients
def get_qc(Phir,DUdx,DUdy,r,param):
    m = param['m']
    n = param['n']
    qc = np.zeros([r,r,r])
    nn = int((r+1)*(r)/2)
    qc_u = np.zeros([r,nn])

    for i in range(0,r): # 3rd dimension of modes
        for j in range(0,r):
            for k in range(0,r):
                A1 = Phir[0:m*n,j]*DUdx[0:m*n,k] + Phir[0:m*n,j]*DUdy[m*n:,k] 
                A2 = Phir[0:m*n,k]*DUdx[0:m*n,j] + Phir[m*n:,k]*DUdy[0:m*n,j] 
                B1 = Phir[m*n:,j]*DUdx[0:m*n,k] + Phir[m*n:,j]*DUdy[m*n:,k] 
                B2 = Phir[0:m*n,k]*DUdx[m*n:,j] + Phir[m*n:,k]*DUdy[m*n:,j] 

                qc[j,k,i] = -Phir[:,i].T@np.concatenate([A1+A2, B1+B2],axis=0)
                    
        qaux = qc[:,:,i] + qc[:,:,i].T 
        # Reshape matrix into a vector for each mode derivative
    
    for i in range(0,r):   
        Q = []
        for j in range(0,r):
                Q.extend(qc[j,j:,i])
        qc_u[i,:] = np.array(Q)

    return qc,qc_u


def get_qp(PODr,r,Dpr0): # recursive scheme, for now stand-by

    q00i = np.zeros([1,r])
    qji = np.zeros([r,r])
    qj0i = np.zeros([r,r])
    qjji = np.zeros([r,r])
    qjki = np.zeros([r,r,r])
    qjki0 = np.zeros([r,r,r])
    
    qij_p = np.zeros([r,r])
    qij_m = np.zeros([r,r])

    
    nn = int((r+1)*(r)/2)
    qp_u = np.zeros([r,nn])
   
    for i in range(0,r):
        q00i[:,i]= -PODr['Phir'][:,i]@Dpr0[:,0]
        
        for j in range(0,r):
           qji[j,i] = -PODr['Phir'][:,i]@(Dpr0[:,0] + Dpr0[:,j])
           
           qj0i[j,i] = -PODr['Phir'][:,i]@Dpr0[:,j]
           # qjji[j,i] = -PODr['Phir'][:,i]@Dpr0[:,0] 
           
           qij_p[j,i] = -PODr['Phir'][:,i]@(Dpr0[:,0] + Dpr0[:,j])
           qij_m[j,i] = -PODr['Phir'][:,i]@(Dpr0[:,0] - Dpr0[:,j])
           qjji[j,i] = (qij_p[j,i] + qij_m[j,i])/2 - q00i[0,i]
           
           for k in range(0,r):
               qjki0[j,k,i] = -PODr['Phir'][:,i]@(Dpr0[:,0] + Dpr0[:,j] + Dpr0[:,k])
               
    for i in range(0,r):
        for j in range(0,r):
            for k in range(j,r):
                qjki[j,k,i] = qjki0[j,k,i] - q00i[:,i] - qj0i[j,i] - qj0i[k,i] - qjji[i,j] - qjji[k,i]
           
                
               
    return qjki, qp_u


def get_qp_emp(uf,PODr,Dp0,r):
    # LHS = np.zeros([uf.shape[1],r])
    A = np.zeros([uf.shape[1],r])
    Lp = np.zeros([r,r])
    for i in range(0,r):
        LHS = -PODr['Phir'][:,i].T@Dp0
        for j in range(0,r):
            A[:,j] = uf.T@PODr['Phir'][:,j] 
        lp = np.linalg.pinv(A)@LHS
        Lp[i,:] = lp.T

    
    return Lp

def myODESystem(t, a, N, l_ij, lp_ij, q_ijk, FlagPressure):
    da_dt = np.zeros(N)
    
    if FlagPressure:
        for i in range(N):
            term1 = sum((l_ij[i][j] + lp_ij[i][j]) * a[j] for j in range(N))
            term2 = sum(sum(q_ijk[i][j][k] * a[j] * a[k] for k in range(N)) for j in range(N))
            da_dt[i] = term1 + term2
            if i == 0:
                da_dt[i]=0# I added this term, in our ODE we are "integrating also a0 which is a constant (=1)-> da0/dt = 0 seems to be more stable, check. I.T.
    else:
        for i in range(N):
            term1 = sum((l_ij[i][j]) * a[j] for j in range(N))
            # term2 = sum(sum(q_ijk[i][j][k] * a[j] * a[k] for k in range(N)) for j in range(N))
            
            term2 = sum(sum(q_ijk[j,k,i] * a[j] * a[k] for k in range(N)) for j in range(N))

            da_dt[i] = term1 + term2
            if i == 0:
                da_dt[i]=0
    
    return da_dt    


# def myODESystem(a, N, l_ij, q_ijk):
#     da_dt = np.zeros(N)

#     for i in range(N):
#         # term1 = sum((l_ij[i][j]) * a[j] for j in range(N))
#         term1 = sum((l_ij[i][j]) * a[j] for j in range(N))
#         term2 = sum(sum(q_ijk[i][j][k] * a[j] * a[k] for k in range(N)) for j in range(N))
#         da_dt[i] = term1 + term2
#         if i == 0:
#             da_dt[i]=0

#     return da_dt    

    # """RHS of POD-Galerkin model, for time integration"""
    # return (L @ a) + np.einsum('ijk,j,k->i', Q, a, a)

    
    
    
    
    
    
    
    
    
    
    
    
    
