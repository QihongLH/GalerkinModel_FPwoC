# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:42:07 2023

@author: QihongLi
"""
import numpy as np
import scipy.io as sio
import math
import matplotlib.pylab as plt
import matplotlib.animation as animation

# Import modules
import module_utility as mu

def createTRset(filename,var):

    data = sio.loadmat(filename)
    TRset = {}
    for i in var:
        if i in data:
            TRset[i] = data[i]
    
    return TRset
    
def createNTR(filename,TRset,ts,dt,var,flag):
    NtTR = TRset['t'].shape[1]                      # Number of snapshots in TR dataset
    NtNTR = int(np.floor((NtTR-1)/np.floor(ts/dt)) + 1)  # Number of snapshots in NTR dataset
    # Generate NtNTR random integers between 1 and NtTR
    
    
    """
    it = np.random.choice(range(1, NtTR + 1), NtNTR, replace=False)
    # Sort the generated integers in ascending order
    it = np.sort(it)
    """
    np.random.seed(5)
    it = np.random.permutation(NtTR)[:NtNTR]
    it = np.sort(it)
    # Select the first NtNTR elements from the permutation
    # it = np.linspace(0,NtNTR-1,NtNTR).astype(int)
    # it = it + 1
    NTRset = {}
    
   
    if flag == 'YES':  # Create NTR with irregular spacing
        for i in var:
            if i in TRset:
                NTRset[i] = TRset[i][:,it-1]
        NTRset['index'] = it
    else: # Create NTR with regular spacing
        for i in var:
            if i in TRset:
                end_value = TRset[i].shape[1]   
                indices = np.arange(0, end_value+1, np.ceil(ts / dt)).astype(int)
                NTRset[i] = TRset[i][:,indices]
        NTRset['index'] = indices
            # NTRset[i] = TRset[i][:,0:-1:math.ceil(ts/dt)-1]
    
    return NTRset

            
def Laplacian2D(f,dx,dy):
    fx = np.zeros_like(f)
    fy = np.zeros_like(f)
    """
    fx[:,1:-2] = (f[:,0:-3]+f[:,2:-1]-2*f[:,1:-2])/dx**2
    fx[:,0] = (2*f[:,0]-5*f[:,1]+4*f[:,2]-f[:,3])/dx**2
    fy[:,1:-2] = (f[:,0:-3]+f[:,2:-1]-2*f[:,1:-2])/dy**2
    fy[:,0] = (2*f[:,0]-5*f[:,1]+4*f[:,2]-f[:,3])/dy**2
    fy[:,0] = (2*f[:,-1]-5*f[:,-2]+4*f[:,-3]-f[:,-4])/dy**2
    """
    fx[:,1:-1] = (f[:,0:-2]+f[:,2:]-2*f[:,1:-1])/(dx**2)
    fx[:,0] = (2*f[:,0]-5*f[:,1]+4*f[:,2]-f[:,3])/(dx**2)
    fx[:,-1] = (2*f[:,-1]-5*f[:,-2]+4*f[:,-3]-f[:,-4])/(dx**2)

    fy[1:-1,:] = (f[0:-2,:]+f[2:,:]-2*f[1:-1,:])/(dy**2)
    fy[0,:] = (2*f[0,:]-5*f[1,:]+4*f[2,:]-f[3,:])/(dy**2)
    fy[-1,:] = (2*f[-1,:]-5*f[-2,:]+4*f[-3,:]-f[-4,:])/(dy**2)
    
    LAP = fx + fy
    return LAP

def Gradient_2D(u,dx,dy): # Compute gradient function 
    
    dudy,dudx = np.gradient(u)
    dudx = dudx/dx
    dudy = dudy/dy
    
    dudx[:,0] = [-3*u[:,0] + 4*u[:,1] - u[:,2]]/(2*dx)  # start region
    dudy[0,:] = [-3*u[0,:] + 4*u[1,:] - u[2,:]]/(2*dy)
    
    dudx[:,-1] = [3*u[:,-1] - 4*u[:,-2] + u[:,-3]]/(2*dx) # end region
    dudy[-1,:] = [3*u[-1,:] - 4*u[-2,:] + u[-3,:]]/(2*dy)
    
    return dudx, dudy


def GradientP_2D(NTRset,param,der,grid): # Compute pressure gradient (pressure obtained from NS)
    
    
    m = param['m']
    n = param['n']
    
    nu = param['nu']
    u = NTRset['u']
    v = NTRset['v']
    du = NTRset['du']
    dv = NTRset['dv']
    DUdx = der['DUdx'] 
    DUdy = der['DUdy'] 
    D2Ur = der['D2Ur'] 
    
    dudx = np.zeros([n*m,NTRset['t'].shape[0]])
    dudy = np.zeros_like(dudx)
    dvdx = np.zeros_like(dudx)
    dvdy = np.zeros_like(dudx)
    dudy = np.zeros_like(dudx)
    dpdx = np.zeros_like(dudx)
    dpdy = np.zeros_like(dudx)
    D2u = np.zeros_like(dudx)
    D2v = np.zeros_like(dudx)
    # Laplacian fields
    for i in range(0,NTRset['t'].shape[0]):
            ui = NTRset['u'][:,i].reshape([m,n]).T
            vi = NTRset['v'][:,i].reshape([m,n]).T
     #      
            gradient_result = mu.Gradient_2D(ui,grid['dx'],grid['dy'])
            dudx[:,i],dudy[:,i] = gradient_result[0].reshape([n*m,],order='F'), gradient_result[1].reshape([n*m,],order='F')
            gradient_result = mu.Gradient_2D(vi,grid['dx'],grid['dy'])
            dvdx[:,i],dvdy[:,i] = gradient_result[0].reshape([n*m,],order='F'), gradient_result[1].reshape([n*m,],order='F')
            # DUdx[:,i] = np.concatenate([dudx,dvdx],axis=0).reshape([n*m*2,])
            # DUdy[:,i] = np.concatenate([dudy,dvdy],axis=0).reshape([n*m*2,])
            D2u[:,i] = mu.Laplacian2D(ui,grid['dx'],grid['dy']).reshape([n*m,],order='F')
            D2v[:,i] = mu.Laplacian2D(vi,grid['dx'],grid['dy']).reshape([n*m,],order='F')
    
    dpdx = nu*D2u - du - u*dudx - v*dudy
    dpdy = nu*D2v - dv - u*dvdx - v*dvdy


    return dpdx, dpdy
    


def normalise(X):
    n = np.zeros([1,X.shape[1]])
    Xn = np.zeros_like(X)
    for j in range(0,X.shape[1]):
        n[:,j] = np.linalg.norm(X[:,j],ord = 0)
        Xn[:,j] = X[:,j]/int(n[:,j])
        
    
    return n

def makegif(time,grid,data,param,filename):
    
    fig  = plt.figure()
    ax   = plt.subplot(111)
    fond = ax.pcolormesh(grid['X'],grid['Y'],data['uf'][0:data['u'].shape[0],0].reshape([param['m'],param['n']]),cmap='jet',clim=[-0.4, 0.4])
    
    def animate(t):
        ax.cla()
        fond = ax.pcolormesh(grid['X'],grid['Y'],data['uf'][0:data['u'].shape[0],t].reshape([param['m'],param['n']]),cmap='jet',clim=[-0.4, 0.4])
        
        return  fond

    ani = animation.FuncAnimation(fig, animate, frames=time)
    plt.show()

    ani.save(filename)    
    

import matplotlib.pyplot as plt
from matplotlib import animation

# def makevideo(time, grid, data, param, filename, fps=10, codec='libx264'):
#     fig, ax = plt.subplots()
#     fond = ax.pcolormesh(grid['X'], grid['Y'], data['uf'][0:data['u'].shape[0], 0].reshape([param['m'], param['n']]), cmap='jet', clim=[-0.4, 0.4])

#     def animate(t):
#         ax.clear()
#         fond = ax.pcolormesh(grid['X'], grid['Y'], data['uf'][0:data['u'].shape[0], t].reshape([param['m'], param['n']]), cmap='jet', clim=[-0.4, 0.4])
#         return fond

#     ani = animation.FuncAnimation(fig, animate, frames=time)

    
#     writer = animation.FFMpegWriter(fps=fps, codec=codec)
#     ani.save(filename, writer=writer)

#     plt.show()

# def makevideo(time, grid, data, param, filename, fps=10, codec='libx264'):
#     fig, ax = plt.subplots()
#     fond = ax.pcolormesh(grid['X'], grid['Y'], data[:,0].reshape([param['m'], param['n']]), cmap='jet', clim=[-0.4, 0.4])

#     def animate(t):
#         ax.clear()
#         fond = ax.pcolormesh(grid['X'], grid['Y'], data[:,t].reshape([param['m'], param['n']]), cmap='jet', clim=[-0.4, 0.4])
#         return fond

#     ani = animation.FuncAnimation(fig, animate, frames=time)

    
#     writer = animation.FFMpegWriter(fps=fps, codec=codec)
#     ani.save(filename, writer=writer)

#     plt.show()

# def makevideo(time, grid, data1, param, filename, fps=100, codec='libx264'):
#     fig, ax = plt.subplots()
#     fond = ax.pcolormesh(grid['X'], grid['Y'], data1[:,0].reshape([param['m'], param['n']]), cmap='jet', clim=[-0.4, 0.4])
#     snapshot_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', fontsize=10)
    
#     def animate(t):
#         ax.clear()
#         fond = ax.pcolormesh(grid['X'], grid['Y'], data1[:,t].reshape([param['m'], param['n']]), cmap='jet', clim=[-0.4, 0.4])
        
#         # Display the snapshot number in the video frame
#         snapshot_text.set_text('Snapshot: {}'.format(t + 1))  # Display snapshot number (start from 1)
        
#         return fond, snapshot_text

#     ani = animation.FuncAnimation(fig, animate, frames=time)

#     writer = animation.FFMpegWriter(fps=fps, codec=codec)
#     ani.save(filename, writer=writer)

#     plt.show()
    
def compute_error(error,set1,set2,NTR_tind,ref,param,flag_sep):
#------------------------------------------------------------------------------
# INPUTS
# error: empty list
# data1/data2: 2D matrices with data -> dimensions [2*Np,Nt]
# NTR_tind: vector with time of available snapshot
# ref: Normalising parameter -> std(NTR_Train['uf'])
# param: list with 
    # data['Np']: number of points per snapshot
# flag_sep: 1 -> Compute rms as function of subsampling snapshot spacing
#           0 -> Flag off
#------------------------------------------------------------------------------
# OUTPUTS
# error['u_rmse']/error['u_rmse']: Root Mean Square Error (normalised)
#------------------------------------------------------------------------------

    u1 = set1[0:param['Np'],:]
    v1 = set1[param['Np']:,:]
    u2 = set2[0:param['Np'],:]
    v2 = set2[param['Np']:,:]

    # Root mean squared error: sqrt((mean(data1-data2)**2,1))
    # error['u_rmse'] = np.sqrt(np.mean((u1-u2)**2,1))/ref['u']
    # error['v_rmse'] = np.sqrt(np.mean((v1-v2)**2,1))/ref['v']
    error['u_rmse'] = np.sqrt(np.mean((u1-u2)**2,1))
    error['v_rmse'] = np.sqrt(np.mean((v1-v2)**2,1))
    Dt = int(NTR_tind[1] - NTR_tind[0])
    
    if flag_sep:
        # error['u_fom'] = np.zeros([NTR_tind.shape[0]-2,1])
        # error['v_fom'] = np.zeros([NTR_tind.shape[0]-2,1])
        
        # aux1 = np.sqrt(np.mean((u1-u2)**2,0))/ref['u']
        # aux2 = np.sqrt(np.mean((v1-v2)**2,0))/ref['v']
        
        aux1 = np.mean((u1-u2)**2,0)/ref['u']**2
        aux2 = np.mean((v1-v2)**2,0)/ref['v']**2
        auxV = (np.mean((u1-u2)**2,0)+ np.mean((v1-v2)**2,0))/(ref['u']**2+ref['v']**2)
        
        
        error['u_fom'] = np.zeros([Dt+1,1])
        error['v_fom'] = np.zeros([Dt+1,1])
        error['V_fom'] = np.zeros([Dt+1,1])
        
        for i in range(0,Dt+1):
            # error['u_fom'][i-1] = np.mean(np.sqrt(np.mean((u1[:,0:-Dt:Dt]-u1[:,i:-Dt+i:Dt])**2,1)),0)
            # error['v_fom'][i-1] = np.mean(np.sqrt(np.mean((v1[:,0:-Dt:Dt]-v1[:,i:-Dt+i:Dt])**2,1)),0)
            error['u_fom'][i] = np.mean(aux1[i:-Dt+i:Dt])
            error['v_fom'][i] = np.mean(aux2[i:-Dt+i:Dt])
            error['V_fom'][i] = np.mean(auxV[i:-Dt+i:Dt])
            
    return error
            
            
def computeAcc(data_set,param):
    data_set['du'] = np.gradient(data_set['u'], param['PIV_dt'], axis=1)
    data_set['dv'] = np.gradient(data_set['v'], param['PIV_dt'], axis=1)
    
    return data_set['du'], data_set['dv']














    
    

    