# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 10:52:29 2023

@author: QihongLi
"""

import numpy as np
import scipy.io as sio
import numpy.matlib
import matplotlib.pyplot as plt

#%% Compute statistics

def ComputeStats(data,param):
    
    U = np.concatenate((dataset['u'],data['v']),axis=0)              # Concatenate velocity components in general Velocity vector 
    # data['um'] = np.mean(dataset['u'],1)                                          # Compute mean velocity 
    # data['vm'] = np.mean(data['v'],1)                                          # Compute mean velocity 
    data['um'] = np.mean(U,1)                                          # Compute mean velocity 
    data['uf'] = U - np.matlib.repmat(data['um'], U.shape[1],1).T # Compute velocity fluctuation
    data['u2m'] = np.mean(dataset['u']**2,1)
    data['v2m'] = np.mean(dataset['u']**2,1)
    data['uvm'] = np.mean(dataset['u']@dataset['v'],1)
    
    POD = {}
    # Velocity POD decomposition
    POD['Phi'],POD['Sigma'],POD['Psi']  = np.linalg.svd(NTR_Train['uf'] , full_matrices=False) 
    # Careful: This Psi is already transposed!!!!!
    POD['a'] = (np.diag(POD['Sigma'])@POD['Psi']).T                         # GP expansion coefficients
    a0 = np.ones([NTR_Train['uf'].shape[1],1])                              # a = 1 for mean velocity term

    # Create a figure and subplots
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # # Plot on the first subplot
    # ax1.plot(x, y1, label='sin(x)')
    # ax1.set_title('Plot 1')
    # ax1.legend()
    
    # # Plot on the second subplot
    # ax2.plot(x, y2, label='cos(x)', color='orange')
    # ax2.set_title('Plot 2')
    # ax2.legend()
    
    # plt.tight_layout()  # Adjust the layout for better spacing
    # plt.show()
    
    # nmodes = np.linspace(1, POD['Sigma'].shape[0],POD['Sigma'].shape[0])
    # plt.figure()
    # plt.plot(nmodes,POD['Sigma']**2/np.sum(POD['Sigma']**2))
    
    # energy = np.cumsum(POD['Sigma']**2)/np.sum(POD['Sigma']**2)
    # plt.figure()
    # plt.plot(nmodes,energy)

    
    # Truncated POD decomposition
    PODr = {}
    r = param['r']
    # Velocity decomposition
    PODr['ar'] = np.concatenate([a0,POD['a'][:,0:r-1]],1)
    PODr['Phir']  = np.concatenate([NTR_Train['um'].reshape([-1,1]),POD['Phi'][:,0:r-1]],1)
    Sigma0 = np.linalg.norm(a0)
    Psi0 = a0/np.linalg.norm(a0)
    PODr['Psir']  = np.concatenate([Psi0.T,POD['Psi'][0:r-1,:]],0)
    PODr['Sigmar']  = np.diag(np.concatenate([Sigma0.reshape([-1,1]),POD['Sigma'][0:r-1].reshape([-1,1])],0).reshape([r,]))
    # PODr['a_pr'] = np.concatenate([a0,a_p[:,0:r-1]],1)
    # PODr['Phi_pr']  = np.concatenate([NTR_Train['pm'].reshape([-1,1]),Phi_p[:,0:r-1]],1)
    del POD

    return PODr, data