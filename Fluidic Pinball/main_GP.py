# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:21:02 2023

@author: QihongLi
"""
# Import packages
import numpy as np
import scipy.io as sio
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import CubicSpline

# from sklearn.metrics import mean_squared_error
from matplotlib import animation
import os
import time
import pickle
import h5py
from tqdm import tqdm  # Import tqdm for progress bar
from scipy.io import savemat

import warnings
import cProfile
from multiprocessing import Pool
# import line_profiler
from pyinstrument import Profiler
# from memory_profiler import profile
import memory_profiler
from scipy.integrate import solve_ivp
from scipy.integrate import solve_bvp

import subprocess
from sklearn.metrics import mean_squared_error
import copy
import pandas as pd
from scipy.optimize import fsolve
from scipy.optimize import root
from scipy.integrate import solve_bvp



# Import modules
import module_utility as mu
import module_GalerkinProjection as gp
import module_configuration as mc
import module_integration as mi
# import int2 as mi

# @profile
def main():
    warnings.filterwarnings("ignore")

    #%% CONFIGURATION
    FlagRunCode = 1                                                             # Run code
    FlagPlots = 1                                                               # Plot results
    FlagVideos = 1                                                              # Do videos
    FlagData = 0
            
    FlagParal = 0                                                               # Paralelise integration
    FlagPressure = 0                                                            # Include pressure term
    FlagTruncation = 1                                                          # Truncation method
        # 0 Manual number of modes
        # 1 Energy truncation
        # 2 Elbow method
        # FlagGPcoeff = 0
        
    FlagLoadData = 0                                                            # Load raw data files
    FlagPOD = 0                                                                 # Do POD
    FlagGPcoeff = 0                                                             # Compute Galerkin coefficients
    FlagInt = 1                                                                 # Integrate
    FlagIntegration = 1                                                         # Integration method
        # 1 Solve My system of equations
        # 2 ODE system built with Einsum
        
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 11
    
    #%% Parameter settings
    param = {}
    param['r'] = 10                                                         # rank     
    # Original: choose r = 11
    modes = [1,2,3]                                                             # Modes to be represented

    param['dsnaps'] = 10
    
    param['E'] = 0.8                                                        # Energy to cut number of modes

    param['dt'] = 1/10                                                      # Non-dimensional time spacing
    param['nu'] = 1/130                                                     # Non-dimensional kinematic viscosity, equivalent to 1/Re
    
    Res = 25 #pix/D

    grid = {}                                                               # Initialise empty dictionary
    
    data = sio.loadmat('data\FP_grid.mat')                                  # Load grid data
    grid['X'] = data['X'].T
    grid['Y'] = data['Y'].T
    grid['dx'] = np.abs(grid['X'][0,0] - grid['X'][1,0])
    grid['dy'] = np.abs(grid['Y'][0,0] - grid['Y'][0,1])
    del data
    
    param['n'],param['m'] = grid['X'].T.shape                               # Get dimensions of the problem 
    param['Np'] = param['m']*param['n']                                     # Total number of points
    # m number of x points
    # n number of y points
    
    DataName = ['FP_10k_13k','FP_14k_24k']
    
    param['filename'] = f"data\{DataName[FlagData]}.mat" # "data\FP_14k_24k.mat"     # Load Training and Testing data
    var_str = {'u','v','du','dv','t','p','w','dudx','dudy','dvdx','dvdy'}   # Fields available in the data sets
    # Integrator configuration - Initialize integrator keywords for solve_ivp to replicate the odeint defaults
    integrator_keywords = {}
    integrator_keywords['rtol'] = 1e-15
    integrator_keywords['method'] = 'LSODA' #'LSODA'
    integrator_keywords['atol'] = 1e-10
        
    #%% Folder paths (FP) definition
    # Define parameters in the data set and the target folder to save results
    var_str = {'u','v','du','dv','t','p','w','dudx','dudy','dvdx','dvdy'}   # Fields available in the data setsdv'}

    FolderName = f"results\{DataName[FlagData]}_{param['dsnaps']}s\Galerkin"
        
    if FlagTruncation == 0:
        FP_results = f"{FolderName}\{param['r']}r"
    elif FlagTruncation == 1:
        FP_results = f"{FolderName}\{int(param['E']*100)}E"
    else:
        FP_results = f"{FolderName}\Elb"

    if FlagPressure == 1:
        folder_path = f"{FP_results}_p"
    else:
        # folder_path = FP_results
        folder_path = f"{FP_results}"

    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    param['filenameData'] = 'data\python'
    if not os.path.exists(param['filenameData']):
        os.makedirs(param['filenameData'])
        
        
    #%% 0. Post-processing of results
    if FlagRunCode:
        if FlagLoadData:
            print(f"Loading data")
            
            FP_10k_13k = mu.createTRset("data\FP_10k_13k.mat",var_str)                  # Generate Time-Resolved Training dataset
            FP_14k_24k = mu.createTRset("data\FP_14k_24k.mat",var_str)                  # Generate Time-Resolved Training dataset

            param['filenameData'] = 'data\python'
            if not os.path.exists(param['filenameData']):
                os.makedirs(param['filenameData'])
                        
            file_path = os.path.join(param['filenameData'], 'FP_10k_13k.pkl')
            with open(file_path, 'wb') as fp:
                pickle.dump(FP_10k_13k, fp)
                print('FP_10k_13k Dictionary saved successfully to file:', file_path)
                
            file_path = os.path.join(param['filenameData'], 'FP_14k_24k.pkl')
            with open(file_path, 'wb') as fp:
                pickle.dump(FP_14k_24k, fp)
                print('FP_14k_24k Dictionary saved successfully to file:', file_path)
                
            if FlagData == 0:
                TR_Train = dict(FP_10k_13k)
                del FP_14k_24k
                
            elif FlagData == 1:
                TR_Train = dict(FP_14k_24k)
                del FP_10k_13k
            
        else:
            
            file_path = os.path.join(param['filenameData'], f"{DataName[FlagData]}.pkl")
            with open(file_path, 'rb') as fp:
                TR_Train = pickle.load(fp)
                print(f'{DataName[FlagData]} dictionary loaded')
                print()
            
        # Training set: NTR                                                                                    
        NTR_Train = {}
        NTR_Train['ts'] = param['dt']*param['dsnaps']                               # Average time separation between snapshots [s]
        NTR_Train['Flag_Irr'] = ''                                                  # Irregular spacing Flag: choose YES for Training set
        NTR_Train = mu.createNTR(param['filename'],TR_Train,NTR_Train['ts'],param['dt'],var_str,NTR_Train['Flag_Irr'])
        # del TR_Train
        
        TR_Train['t'] = TR_Train['t'][0,:]
        NTR_Train['t'] = NTR_Train['t'][0,:]
        
        # Testing set: TR for comparison
        TR_Test = dict(TR_Train)      
        # NTR for integration
        NTR_Test = dict(NTR_Train)

        np.save(f'{FolderName}/Tindex.npy',NTR_Test['index'])
        np.save(f'{FolderName}/Tvec.npy',TR_Test['t'])
        

        #%% II. Transformation into reduced-order model 
        U = np.concatenate((NTR_Train['u'],NTR_Train['v']),axis=0)              # Concatenate velocity components in general Velocity vector 
        NTR_Train['um'] = np.mean(U,1)                                          # Compute mean velocity 
        NTR_Train['uf'] = U - np.matlib.repmat(NTR_Train['um'], U.shape[1],1).T # Compute velocity fluctuation
        
        # Preparing datasets
        TR_Test['um'] = NTR_Train['um']
        
        # Time resolved Test set - For comparison
        Utest = np.concatenate((TR_Test['u'],TR_Test['v']),axis=0) 
        TR_Test['uf'] = Utest - np.matlib.repmat(TR_Test['um'], Utest.shape[1],1).T
        
        # Non time-resolved Test set - For integration
        Utest = np.concatenate((NTR_Test['u'],NTR_Test['v']),axis=0) 
        NTR_Test['uf'] = Utest - np.matlib.repmat(TR_Test['um'], Utest.shape[1],1).T
        
        del U, Utest
        
        #%%
        
        Res = 25 #pix/D
        xmin=-5
        ymin=-4+4/Res
        
        # Dimensionless coordinate
        theta   =   2*np.pi*np.arange(0,100+1)/100              
        xCyl1   =   - 1.299 + 0.5*np.cos(theta)
        yCyl1   =   0.5*np.sin(theta)
        xCyl    =   0.5*np.cos(theta)                
        yCyl2   =   0.75 + 0.5*np.sin(theta)                 
        yCyl3   =   -0.75 + 0.5*np.sin(theta)
        
        def make_animation_3(grid, NTR_Test, param):
            # Create a figure and subplots for the animation
            fig, ax0 = plt.subplots()
           
            # Define global variables for text annotations
            global snapshot_text0
            snapshot_text0 = None

           
            # Define the first plot (left)
            fond0 = ax0.pcolormesh(grid['X'], grid['Y'], NTR_Test['uf'][0:param['Np'], 0].reshape([param['m'], param['n']]), cmap='jet', clim=[-0.4, 0.4])
            snapshot_text0 = ax0.text(0.02, 0.95, '', transform=ax0.transAxes, color='white', fontsize=10)
            ax0.plot(xCyl1, yCyl1, 'k')
            ax0.plot(xCyl, yCyl2, 'k')
            ax0.plot(xCyl, yCyl3, 'k')
            ax0.fill(xCyl, yCyl2, color='white')
            ax0.fill(xCyl, yCyl3, color='white')
            ax0.fill(xCyl1, yCyl1, color='white')
            plt.axis('tight')
            plt.axis('equal')
            # cbar0 = fig.colorbar(fond0, ax=ax0)
            # cbar0.set_label('$u^{\prime}/U_{\infty}$', rotation=90, labelpad=20)
            # ax0.set_aspect('equal', 'box')
            ax0.axis('tight')
            ax0.axis('equal')
            # ax0.set_title('Cumulative Energy')        
            # plt.ylabel('$Y/D$')
            # plt.xlabel('$X/D$')
            ax0.axis('off')
            ax0.set_xticks([])
            ax0.set_yticks([])
            fig.tight_layout()

           
            # Function to update the plots for each frame of the animation
            def animate(t):
                global snapshot_text0
       
                # Update the first plot
                ax0.clear()
                fond0 = ax0.pcolormesh(grid['X'], grid['Y'], NTR_Test['uf'][0:param['Np'], t].reshape([param['m'], param['n']]), cmap='jet', clim=[-0.4, 0.4])
                ax0.plot(xCyl1, yCyl1, 'k')
                ax0.plot(xCyl, yCyl2, 'k')
                ax0.plot(xCyl, yCyl3, 'k')
                ax0.fill(xCyl, yCyl2, color='white')
                ax0.fill(xCyl, yCyl3, color='white')
                ax0.fill(xCyl1, yCyl1, color='white')
                # ax0.set_aspect('equal', 'box')
                plt.axis('tight')
                plt.axis('equal')
                ax0.axis('tight')
                ax0.axis('equal')
                ax0.axis('off')
                ax0.set_xticks([])
                ax0.set_yticks([])
                fig.tight_layout()
                # plt.ylabel('$Y/D$')
                # plt.xlabel('$X/D$')
       
       
                return fond0, snapshot_text0
   
            # Create the animation object with the custom interval
            # ani = animation.FuncAnimation(fig, animate, frames=400, interval=lambda frame: interval_function(frame))
            ani = animation.FuncAnimation(fig, animate, frames=NTR_Test['uf'].shape[1]-1, interval=20)  # 100 milliseconds interval

       
            # Save the animation as an MP4 video
            current_directory = os.getcwd()
            video_path = os.path.join(current_directory, f'{folder_path}/video_3.mp4')
            writer = animation.FFMpegWriter(fps=2, codec='h264')
            ani.save(video_path, writer=writer)
       
            # Save the animation as a GIF
            gif_path = os.path.join(current_directory, f'{folder_path}/video_3.gif')
            ani.save(gif_path, writer='imagemagick', fps=2)

           
        make_animation_3(grid, NTR_Test, param)
        
        
        print(f"Proper Orthogonal Decomposition")

        if FlagPOD:
            start_time = time.time()
            POD = {}
            # Velocity POD decomposition
            POD['Phi'],POD['Sigma'],POD['Psi']  = np.linalg.svd(NTR_Train['uf'] , full_matrices=False) 
                
            # Careful: This Psi is already transposed!!!!!
            POD['a'] = (np.diag(POD['Sigma'])@POD['Psi']).T                         # GP expansion coefficients
                
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken for POD of TR train data: {elapsed_time} seconds")  
            print()
            
            file_path = os.path.join(FolderName, 'POD.pkl')
            with open(file_path, 'wb') as fp:
                pickle.dump(POD, fp)
                print('POD Dictionary saved successfully to file:', file_path)
                print()
                
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken for POD of TR test data: {elapsed_time} seconds")  
            print()
            
            # Computing features for the reconstruction with GP
            GPrecon = {}
            # Velocity decomposition
            GPrecon['Phi'],GPrecon['Sigma'],GPrecon['Psi']  = np.linalg.svd(NTR_Test['uf'], full_matrices=False) 
                
            GPrecon['a'] = (np.diag(GPrecon['Sigma'])@GPrecon['Psi']).T             # Modes are in columns
 
            file_path = os.path.join(FolderName, 'GPrecon.pkl')
            with open(file_path, 'wb') as fp:
                pickle.dump(GPrecon, fp)
                print('GPrecon Dictionary saved successfully to file:', file_path)
                print()
                            
        else:
            # Read dictionary pkl file
            file_path = os.path.join(FolderName, 'POD.pkl')
            with open(file_path, 'rb') as fp:
                POD = pickle.load(fp)
                print('POD dictionary loaded')
                
            TR_a = np.load(f'{FolderName}/TR_a.npy')
            
            file_path = os.path.join(FolderName, f'GPrecon.pkl')
            with open(file_path, 'rb') as fp:
                GPrecon = pickle.load(fp)
                print('GPrecon dictionary loaded')
                
        energy = np.cumsum(POD['Sigma']**2)/np.sum(POD['Sigma']**2)

        # Create a figure and subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot on the first subplot
        ax1.semilogx(POD['Sigma']**2/np.sum(POD['Sigma']**2))
        ax1.set_title('Mode Energy')
        ax1.grid()
        
        # Plot on the second subplot
        ax2.semilogx(energy)
        ax2.set_title('Cumulative Energy')
        ax2.grid()
        
        plt.tight_layout()  # Adjust the layout for better spacing
        plt.show()
        plt.savefig(f"{FolderName}/Energy plot.png", dpi=300, bbox_inches='tight')
        
        if param['r'] > GPrecon['Sigma'].shape[0]:
            param['r'] = GPrecon['Sigma'].shape[0]-1
            
        if FlagTruncation == 0:
            print(f"Number of {param['r']} the energy {energy[param['r']]*100}%")
        else:
            param['r'] = np.abs(energy - param['E']).argmin()
            print(f"Number of modes for {param['E']*100}% of energy: r = {param['r']}")
            
        # Truncated POD decomposition
        PODr = {}
        r = param['r']
        # Velocity decomposition
        a0 = np.ones([POD['a'].shape[1],1])*numpy.linalg.norm(NTR_Train['um'])                              # a = 1 for mean velocity term
        # a0 = np.ones([POD['a'].shape[1],1])

        PODr['ar'] = np.concatenate([a0,POD['a'][:,0:r-1]],1)
        PODr['Phir']  = np.concatenate([NTR_Train['um'].reshape([-1,1])/numpy.linalg.norm(NTR_Train['um']),POD['Phi'][:,0:r-1]],1)
        # PODr['Phir']  = np.concatenate([NTR_Train['um'].reshape([-1,1]),POD['Phi'][:,0:r-1]],1)
        Sigma0 = np.linalg.norm(a0)
        Psi0 = a0/np.linalg.norm(a0)
        PODr['Psir']  = np.concatenate([Psi0.T,POD['Psi'][0:r-1,:]],0)
        PODr['Sigmar']  = np.diag(np.concatenate([Sigma0.reshape([-1,1]),POD['Sigma'][0:r-1].reshape([-1,1])],0).reshape([r,]))
        
        del POD
        
        # # Project accelerations in SVD basis
        # dU = np.concatenate((NTR_Train['du'],NTR_Train['dv']),axis=0) 
        # PODr['dar'] = np.zeros([r-1,dU.shape[1]]);
        # for i in range(0,dU.shape[1]):
        #     PODr['dar'][:,i] = PODr['Phir'][:,1:r].T@dU[:,i]
        
        del Sigma0, Psi0 #, dU
        
        #%% III. Galerkin projection
        if FlagGPcoeff:
            print(f"Get Galerkin Coefficients")
            start_time = time.time()
            GPcoeff = gp.getGalerkinCoeffs(grid,PODr,NTR_Train,param)
        
            print()
            # Ensure the target folder exists or create it if it doesn't
            os.makedirs(folder_path, exist_ok=True)
        
            # Save dictionary to person_data.pkl file in the target folder
            file_path = os.path.join(folder_path, 'GPcoeff.pkl')
            with open(file_path, 'wb') as fp:
                pickle.dump(GPcoeff, fp)
                print('GPcoeff Dictionary saved successfully to file:', file_path)
                
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken for GP coefficients: {elapsed_time} seconds")  
            print()
                
        else:
            print(f"Load Galerkin Coefficients")
            print()
            GPcoeff = {}      
                  
            # Read dictionary pkl file
            file_path = os.path.join(folder_path, 'GPcoeff.pkl')
            with open(file_path, 'rb') as fp:
                GPcoeff = pickle.load(fp)
                print('GPcoeff dictionary loaded')
 
       #%% IV. TEMPORAL MODE SET INTEGRATION AND VELOCITY AND PRESSURE RECONSTRUCTION
       
        a0 = np.ones([GPrecon['a'].shape[1],1])*numpy.linalg.norm(NTR_Train['um'])
        # a0 = np.ones([GPrecon['a'].shape[1],1])
        
        # Velocity decomposition
        GPrecon['ar'] = np.concatenate([a0,GPrecon['a'][:,0:r-1]],1)
        GPrecon['Phir']  = np.concatenate([TR_Test['um'].reshape([-1,1])/numpy.linalg.norm(NTR_Train['um']),GPrecon['Phi'][:,0:r-1]],1)
        # GPrecon['Phir']  = np.concatenate([TR_Test['um'].reshape([-1,1]),GPrecon['Phi'][:,0:r-1]],1)

        Sigma0 = np.linalg.norm(a0)
        Psi0 = a0/np.linalg.norm(a0)
        GPrecon['Psir']  = np.concatenate([Psi0.T,GPrecon['Psi'][0:r-1,:]],0)
        GPrecon['Sigmar']  = np.diag(np.concatenate([Sigma0.reshape([-1,1]),GPrecon['Sigma'][0:r-1].reshape([-1,1])],0).reshape([r,]))
        
        a_interp =  {}
        a_subs = CubicSpline(NTR_Test['t'], GPrecon['a'])
        a_interp['a'] = a_subs(TR_Test['t'])
        a_interp['Urec'] = GPrecon['Phi'][:,0:r-1]@a_interp['a'][:,0:r-1].T # Urecon is the fluctuation velocity reconstruction
                 

        print(f"Save results")  
        file_path = os.path.join(folder_path, f'a_interp.pkl')
        with open(file_path, 'wb') as fp:
            pickle.dump(a_interp, fp)
            print('a_interp Dictionary saved successfully to file:', file_path)  

        del a0, Sigma0, Psi0
                    
       
        if FlagInt:

            # Integration of ODEs for each states using an average between forward and backward integration
            if FlagIntegration == 1:   # Solve my system of equations
                print(f"Integration of temporal coefficients")   
                start_time = time.time()
                
                a = []       # Initialise list
                nt = int(NTR_Test['t'][1]/param['dt']) + 1
                r = param['r']

                for i in tqdm(range(0,NTR_Test['index'].shape[0]-1), desc="Integration Progress"):
                # for i in tqdm(range(0,int(300/param['dsnaps'])), desc="Integration Progress"):
                # for i in tqdm(range(0,5), desc="Integration Progress"):

                    # building initial condition for both the integrations
                    # a_IC_f = GPrecon['Phir'][:,:].T@NTR_Test['uf'][:,i]
                    # a_IC_f [0,] = 1
                    # a_IC_b = GPrecon['Phir'][:,:].T@NTR_Test['uf'][:,i+1]
                    # a_IC_b [0,] = 1
                    
                    a_IC_f = GPrecon['ar'][i,:]
                    a_IC_b = GPrecon['ar'][i+1,:]
                                        
                    # Time vector for the integration
                    tvec_f = np.linspace(TR_Test['t'][NTR_Test['index'][i]],TR_Test['t'][NTR_Test['index'][i+1]],nt)
                    tvec_b = tvec_f[::-1]
                    
                    
                    '''
                    tvec_f = tvec_f[0:int(nt/2)-1]
                    tvec_b = tvec_b[0:int(nt/2)-1]

                    # result_f = odeint(gp.myODESystem, a_IC_f, tvec_f, args=(r, GPcoeff['l'], GPcoeff['lp'], GPcoeff['qc']), method = 'lsoda')
                    # result_b = odeint(gp.myODESystem, tvec_b, a_IC_b, args=(r, GPcoeff['l'], GPcoeff['lp'], GPcoeff['qc'], FlagPressure))

                    t_span = (tvec_f[0],tvec_f[-1])
                    result_f = solve_ivp(gp.myODESystem, t_span , a_IC_f ,t_eval=tvec_f, args=(r, GPcoeff['l'], GPcoeff['lp'], GPcoeff['qc'], FlagPressure)).y.T
                    t_span = (tvec_b[0],tvec_b[-1])
                    result_b = solve_ivp(gp.myODESystem, t_span , a_IC_b ,t_eval=tvec_b, args=(r, GPcoeff['l'], GPcoeff['lp'], GPcoeff['qc'], FlagPressure)).y.T
    
                    
                    result_b = result_b[::-1]
                    

                    # Take the last three points of vec1 and the first three points of vec2
                    boundary_points_vec1 = result_f[-3:]
                    boundary_points_vec2 = result_b[:3]
                    
                    # Interpolate points between the boundary points using cubic spline
                    t = np.arange(tvec_f[-1], tvec_b[-1], param['dt'])  # Parameter values for interpolation
                    interp_fun = CubicSpline([tvec_f[-3], tvec_b[3]], [boundary_points_vec1[-1], boundary_points_vec2[0]])
                    transition_points = interp_fun(t)
                    # transition_points = CubicSpline([tvec_f[-1], tvec_b[-1]], [boundary_points_vec1[-1], boundary_points_vec2[0]])(t)[1:-1]  # Exclude the first and last points
                    
                    # Concatenate vec1, transition points, and vec2
                    result = np.concatenate((result_f, transition_points[1:3,:], result_b))
                    a.append(result)

                    var = np.var(GPrecon['a'],axis=0)
                    
                    def compute_variance(result_f, ref_var):
                        # var_f = np.zeros([result_f.shape[1],result_f.shape[0]])
                        
                        # for j in range(1, result_f.shape[0]):  # Adjusting the range to include the last column
                        #     var_f[:,j] = np.var(result_f[:j, :],axis=0)
                            
                        #     # if var_f >= ref_var:
                        #     #     return var_f, j  # Returning the variance and the index where it exceeds the reference variance
                        
                        var_f = np.zeros(result_f.shape[0])
                        
                        for j in range(1, result_f.shape[0]):  # Adjusting the range to include the last column
                            var_f[j] = np.var(result_f[:j, :])
                            
                            # if var_f >= ref_var:
                            #     return var_f, j  # Returning the variance and the index where it exceeds the reference variance
                            
                        
                        # If the loop completes without finding a variance exceeding the reference variance
                        return var_f
                                        
                    variance, index = compute_variance(result_f, var)
                    
                    '''

                    tvec_f = tvec_f[0:int(nt/2)+1]
                    tvec_b = tvec_b[0:int(nt/2)+1]
                    
                    t_span_f = (tvec_f[0],tvec_f[-1])
                    result_f = solve_ivp(gp.myODESystem, t_span_f , a_IC_f ,t_eval=tvec_f, args=(r, GPcoeff['l'], GPcoeff['lp'], GPcoeff['qc'], FlagPressure)).y.T
                    t_span_b = (tvec_b[0],tvec_b[-1])
                    result_b = solve_ivp(gp.myODESystem, t_span_b , a_IC_b ,t_eval=tvec_b, args=(r, GPcoeff['l'], GPcoeff['lp'], GPcoeff['qc'], FlagPressure)).y.T
    
                    result_b = result_b[::-1]
                                    
                    weight = np.linspace(1,0,int(nt/2)+1)**3*0.6
    
                    # Hafl 1
                    y0 = 0.5*result_f[-1,:] + 0.5*result_b[0,:]
                    
                    solution_1 = solve_ivp(gp.myODESystem, t_span_f[::-1], y0, t_eval=[tvec_f[-1]], args=(r, GPcoeff['l'], GPcoeff['lp'], GPcoeff['qc'], FlagPressure)).y.T
                    solution_2 = solve_ivp(gp.myODESystem, t_span_b[::-1], y0, t_eval=[tvec_b[-1]], args=(r, GPcoeff['l'], GPcoeff['lp'], GPcoeff['qc'], FlagPressure)).y.T

                    y0_1 = weight[0]*solution_1 + (1-weight[0])*result_f[-2,:]  # Initial condition at the current time step
                    y0_2 = weight[0]*solution_2 +(1-weight[0])*result_b[1,:]  # Initial condition at the current time step
                        
                    solution_f = np.zeros([result_f.shape[0]-1,result_f.shape[1]])
                    solution_b = np.zeros([result_b.shape[0]-1,result_b.shape[1]])
                    solution_f[0,:] = solution_1
                    solution_b[0,:] = solution_2
                    
                    for j in range(2,int(nt/2)+1):

                        
                        t_eval_1 = [tvec_f[-j]]  # Time point at which to evaluate the solution
                        t_eval_2 = [tvec_b[-j]]  # Time point at which to evaluate the solution

                        # Integrate from the current time step to the next time point
                        solution_1 = solve_ivp(gp.myODESystem, t_span_f[::-1], y0_1.flatten(), t_eval=t_eval_1, args=(r, GPcoeff['l'], GPcoeff['lp'], GPcoeff['qc'], FlagPressure)).y.T
                        solution_2 = solve_ivp(gp.myODESystem, t_span_b[::-1], y0_2.flatten(), t_eval=t_eval_2, args=(r, GPcoeff['l'], GPcoeff['lp'], GPcoeff['qc'], FlagPressure)).y.T

                        y0_1 = weight[j]*solution_1 + (1-weight[j])*result_f[-j]  # Initial condition at the current time step
                        y0_2 = weight[j]*solution_2 + (1-weight[j])*result_b[j]  # Initial condition at the current time step
                                                
                        solution_f[j-1,:] =  y0_1
                        solution_b[j-1,:] =  y0_2

                    solution_f = solution_f[::-1]
                    solution_f[0,:] = a_IC_f
                    solution_b[-1,:] = a_IC_b
                    
                    result = np.concatenate((solution_f, [y0], solution_b))
                    
                    a.append(result[0:-1,:])
  

                a.append(a_IC_b)    
                a_recon = np.vstack(a)
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                print(f"Time taken for integration: {elapsed_time} seconds")         
                print()
                
            else:               # Solve system of equations by Einum
                print(f"Integration of temporal coefficients with einsum")   
                start_time = time.time()
            
                a = []       # Initialise list
                nt = int(NTR_Test['t'][1]/param['dt']) + 1
                r = param['r']

                model = lambda t, a: gp.galerkin_model(a, GPcoeff['l'], GPcoeff['qc'])
                
                
                # def model(a, t, l, qc):                             # For the odeint
                #     # Here, a is the state vector, t is the time, l and qc are parameters
                #     return gp.galerkin_model(a, l, qc)
                
                # for i in tqdm(range(0,NTR_Test['index'].shape[0]), desc="Integration Progress"):
                # for i in tqdm(range(0,int(300/param['dsnaps'])), desc="Integration Progress"):
                for i in tqdm(range(0,10), desc="Integration Progress"):

                    # building initial condition for both the integrations
                    a_IC_f = GPrecon['Phir'][:,:].T@NTR_Test['uf'][:,i]
                    a_IC_f [0,] = 1
                    a_IC_b = GPrecon['Phir'][:,:].T@NTR_Test['uf'][:,i+1]
                    a_IC_b [0,] = 1
                    

                    # time vector for the integration
                    # tvec_f = np.arange(TR_Test['t'][0,NTR_Test['index'][i]],TR_Test['t'][0,NTR_Test['index'][i+1]]+param['dt'],param['dt'])
                    tvec_f = np.linspace(TR_Test['t'][NTR_Test['index'][i]],TR_Test['t'][NTR_Test['index'][i+1]],nt)
                    # tvec_b = tvec_f[::-1]
                    tvec_b = np.linspace(TR_Test['t'][NTR_Test['index'][i+1]],TR_Test['t'][NTR_Test['index'][i]],nt)
                    
                    t_span_f = (NTR_Test['t'][i], NTR_Test['t'][i+1])
                    t_span_b = (NTR_Test['t'][i+1], NTR_Test['t'][i])
            
                    result_f = solve_ivp(model, t_span_f, a_IC_f, t_eval=tvec_f,
                            **integrator_keywords).y.T
                    
                    result_b = solve_ivp(model, t_span_b, a_IC_b, t_eval=tvec_b,
                            **integrator_keywords).y.T
                    
                    # result_f = odeint(model, a_IC_f, tvec_f, args=(GPcoeff['l'], GPcoeff['qc']))        # For the odeint
                    # result_b = odeint(model, a_IC_b, tvec_b, args=(GPcoeff['l'], GPcoeff['qc']))


                    # #reallocating the initial conditions
                    # # result_f[0,:] = a_IC_f
                    # # result_b[0,:] = a_IC_b
                    
                    # result_b = result_b[::-1]
                    
                    # #  weights for the average of the integrations
                    # w_b = np.linspace(0,1,result_f.shape[0])#**3
                    # w_f = w_b[::-1]
            
                    # w_f = np.cos(np.linspace(0,np.pi/2,result_f.shape[0]))**2
                    # w_b = np.sin(np.linspace(0,np.pi/2,result_f.shape[0]))**2        
                    
                    # a_aux = np.zeros([result_f.shape[0]-1,result_f.shape[1]])          # Initialise list
                    # for j in range(0,result_f.shape[1]):
                    #     # weighting the results
                    #     a_aux[:,j] = result_f[0:-1,j]*w_f[0:-1] + result_b[0:-1,j]*w_b[0:-1] 
                        
                    # update the list of a
                    a.append(result_f[:,0:-1])
                                        
                a.append(a_IC_b)
                a_recon = np.vstack(a)


                a_recon[:,0] = 1
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                print(f"Time taken for integration: {elapsed_time} seconds")         
                print()

        
    
            print(f"GP Flow reconstruction between snapshots")    
            GPrecon['Urec'] = GPrecon['Phir'][:,1:]@a_recon[:,1:].T # Urecon is the fluctuation velocity reconstruction - Truncated modes
            
                
            
            np.save(f'{folder_path}/GP_Urec{FlagIntegration}.npy', GPrecon['Urec'])
            np.save(f'{FolderName}/TR_Urec.npy', TR_Test['uf'])
            np.save(f'{folder_path}/GP_a.npy', a_recon)            
            

        else:
                                      
            TR_Test['uf'] = np.load(f'{FolderName}/TR_Urec.npy')
            a_recon = np.load(f'{folder_path}/GP_a.npy')
            GPrecon['Urec'] = np.load(f'{folder_path}/GP_Urec{FlagIntegration}.npy')            


        # Low Order Reconstruction 
        print(f"LOR Flow reconstruction between snapshots")    
        start_time = time.time()
        
        Utest = np.concatenate((TR_Test['u'],TR_Test['v']),axis=0) 
        TR_Test['uf'] = Utest - np.matlib.repmat(TR_Test['um'], Utest.shape[1],1).T
        TR_Test['Psi'] = np.linalg.inv(np.diag(GPrecon['Sigma']))@GPrecon['Phi'].T@TR_Test['uf']
        TR_Test['a'] = np.diag(GPrecon['Sigma'][:])@TR_Test['Psi'][:,:]
        
        np.save(f'{FolderName}/TR_a.npy',TR_Test['a'])
        print('Ref Temporal modes saved successfully')

        LOrecon = {}
        LOrecon['Urec'] = GPrecon['Phi'][:,0:r-1]@np.diag(GPrecon['Sigma'][0:r-1])@TR_Test['Psi'][0:r-1,:]
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken for LOR reconstruction: {elapsed_time} seconds")  
        print(f"Save results") 
    
        np.save(f'{folder_path}/LO_Urec.npy', LOrecon['Urec'])

    
    #%% V. COMPUTE ERROR
        print(f"Compute errors")  
        
        # Compute standard deviation
        ref = {}
        # ref['u'] = np.sqrt(np.mean(NTR_Train['uf'][0:TR_Test['u'].shape[0],]*NTR_Train['uf'][0:TR_Test['u'].shape[0],]))
        # ref['v'] = np.sqrt(np.mean(NTR_Train['uf'][TR_Test['u'].shape[0]:,]*NTR_Train['uf'][TR_Test['u'].shape[0]:,]))
        ref['u'] = np.std(NTR_Train['uf'][0:TR_Test['u'].shape[0],])
        ref['v'] = np.std(NTR_Train['uf'][0:TR_Test['u'].shape[0],])
        
        ORI_GP = {}
        LOR_GP = {}
        ORI_LOR= {}
        LOR_interp = {}
        ORI_interp = {}
        
        
        nt = GPrecon['Urec'].shape[1]
        mu.compute_error(ORI_GP,TR_Test['uf'][:,0:nt],GPrecon['Urec'][:,0:nt],NTR_Test['index'],ref,param,1)
        mu.compute_error(LOR_GP,LOrecon['Urec'][:,0:nt],GPrecon['Urec'][:,0:nt],NTR_Test['index'],ref,param,1)
        mu.compute_error(ORI_LOR,TR_Test['uf'][:,0:nt],LOrecon['Urec'][:,0:nt],NTR_Test['index'],ref,param,1)
        mu.compute_error(LOR_interp,LOrecon['Urec'][:,0:nt],a_interp['Urec'][:,0:nt],NTR_Test['index'],ref,param,1)
        mu.compute_error(ORI_interp,TR_Test['uf'][:,0:nt],a_interp['Urec'][:,0:nt],NTR_Test['index'],ref,param,1)

        
        file_path = os.path.join(folder_path, 'ORI_GP.pkl')
        with open(file_path, 'wb') as fp:
            pickle.dump(ORI_GP, fp)
            print('ORI_GP Dictionary saved successfully to file:', file_path)
            print()
        file_path = os.path.join(folder_path, 'LOR_GP.pkl')
        with open(file_path, 'wb') as fp:
            pickle.dump(LOR_GP, fp)
            print('LOR_GP Dictionary saved successfully to file:', file_path)
            print()
        file_path = os.path.join(folder_path, 'ORI_LOR.pkl')
        with open(file_path, 'wb') as fp:
            pickle.dump(ORI_LOR, fp)
            print('ORI_LOR Dictionary saved successfully to file:', file_path)
            print()
            
        file_path = os.path.join(folder_path, 'LOR_interp.pkl')
        with open(file_path, 'wb') as fp:
            pickle.dump(LOR_interp, fp)
            print('LOR_interp Dictionary saved successfully to file:', file_path)
            print()
            
        file_path = os.path.join(folder_path, 'ORI_interp.pkl')
        with open(file_path, 'wb') as fp:
            pickle.dump(ORI_interp, fp)
            print('ORI_interp Dictionary saved successfully to file:', file_path)
            print()
                
    else:
        GPrecon = {}
        LOrecon = {}
        TR_Test = {}
        NTR_Test = {}
        # Load LO_Urec.npy
        LOrecon['Urec'] = np.load(f'{folder_path}/LO_Urec.npy')
        
        # Load GP_Urec.npy
        GPrecon['Urec'] = np.load(f'{folder_path}/GP_Urec{FlagIntegration}.npy')
        
        # Load TR_Urec.npy
        TR_Test['uf'] = np.load(f'{FolderName}/TR_Urec.npy')

        NTR_Test['index'] = np.load(f'{FolderName}/Tindex.npy')
        TR_Test['t'] = np.load(f'{FolderName}/Tvec.npy')

        NTR_Test['t'] = TR_Test['t'][NTR_Test['index']]

        a_recon = np.load(f'{folder_path}/GP_a.npy')

        # Read dictionary pkl file
        file_path = os.path.join(folder_path, 'ORI_GP.pkl')
        with open(file_path, 'rb') as fp:
            ORI_GP = pickle.load(fp)
            print('ORI_GP dictionary loaded')
        file_path = os.path.join(folder_path, 'LOR_GP.pkl')
        with open(file_path, 'rb') as fp:
            LOR_GP = pickle.load(fp)
            print('LOR_GP dictionary loaded')
        file_path = os.path.join(folder_path, 'ORI_LOR.pkl')
        with open(file_path, 'rb') as fp:
            ORI_LOR = pickle.load(fp)
            print('ORI_LOR dictionary loaded')       
        file_path = os.path.join(folder_path, 'LOR_interp.pkl')
        with open(file_path, 'rb') as fp:
            LOR_interp = pickle.load(fp)
            print('LOR_interp dictionary loaded')
        file_path = os.path.join(folder_path, 'ORI_interp.pkl')
        with open(file_path, 'rb') as fp:
            ORI_interp = pickle.load(fp)
            print('ORI_interp dictionary loaded')
    
    #%% Plots

    if FlagPlots == 1:        
        Res = 25 #pix/D
        xmin=-5
        ymin=-4+4/Res
        
        # Dimensionless coordinate
        theta   =   2*np.pi*np.arange(0,100+1)/100              
        xCyl1   =   - 1.299 + 0.5*np.cos(theta)
        yCyl1   =   0.5*np.sin(theta)
        xCyl    =   0.5*np.cos(theta)                
        yCyl2   =   0.75 + 0.5*np.sin(theta)                 
        yCyl3   =   -0.75 + 0.5*np.sin(theta)
        
        #%% PLOT TWO SNAPSHOTS

        if FlagIntegration == 2:
            sol_path = "figures_new"
        else:
            sol_path = "figures"
        
        if not os.path.exists(f"{folder_path}\\{sol_path}"):
                os.makedirs(f"{folder_path}\\{sol_path}")

        plt.rcParams['font.size'] = 11

        Res = (375-179)/30 #pix/
        Nim = 10

        f, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(3, 2, sharex='col', sharey='row',figsize = (6.7,4))
        im = ax1.pcolormesh(grid['X'],grid['Y'],TR_Test['uf'][0:param['Np'],Nim].reshape([param['m'],param['n']]),cmap='jet',clim=[-0.4, 0.4])
        ax1.plot(xCyl1, yCyl1, 'k')
        ax1.plot(xCyl, yCyl2, 'k')
        ax1.plot(xCyl, yCyl3, 'k')
        ax1.fill(xCyl, yCyl2, color='white')
        ax1.fill(xCyl, yCyl3, color='white')
        ax1.fill(xCyl1, yCyl1, color='white')
        plt.axis('tight')
        plt.axis('equal')
        ax1.set_aspect('equal', adjustable='box')
        # ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax1.title.set_text(f"Snapshot {Nim}")
        # ax1.text(0.05, 0.85, 'a) PIV', transform=ax1.transAxes, color='black', fontsize=11, fontweight='bold')
        ax1.text(0.05, 0.8, 'a) Ref', transform=ax1.transAxes, color='black', fontsize=11)

        ax2.pcolormesh(grid['X'],grid['Y'],LOrecon['Urec'][0:param['Np'],Nim].reshape([param['m'],param['n']]),cmap='jet',clim=[-0.4, 0.4])
        ax2.plot(xCyl1, yCyl1, 'k')
        ax2.plot(xCyl, yCyl2, 'k')
        ax2.plot(xCyl, yCyl3, 'k')
        ax2.fill(xCyl, yCyl2, color='white')
        ax2.fill(xCyl, yCyl3, color='white')
        ax2.fill(xCyl1, yCyl1, color='white')
        plt.axis('tight')
        plt.axis('equal')
        ax2.set_aspect('equal', adjustable='box')
        # ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
        # ax2.title.set_text('b) LOR')
        ax2.text(0.05, 0.8, 'b) LOR', transform=ax2.transAxes, color='black', fontsize=11)

        ax3.pcolormesh(grid['X'],grid['Y'],GPrecon['Urec'][0:param['Np'],Nim].reshape([param['m'],param['n']]),cmap='jet',clim=[-0.4, 0.4])
        ax3.plot(xCyl1, yCyl1, 'k')
        ax3.plot(xCyl, yCyl2, 'k')
        ax3.plot(xCyl, yCyl3, 'k')
        ax3.fill(xCyl, yCyl2, color='white')
        ax3.fill(xCyl, yCyl3, color='white')
        ax3.fill(xCyl1, yCyl1, color='white')
        plt.axis('tight')
        plt.axis('equal')
        ax3.set_aspect('equal', adjustable='box')
        # ax3.set_yticks([-1, -0.5, 0, 0.5, 1])
        # ax3.title.set_text('c) GP')
        ax3.text(0.05, 0.8, 'c) GP', transform=ax3.transAxes, color='black', fontsize=11)

        Nim = 25

        ax4.pcolormesh(grid['X'],grid['Y'],TR_Test['uf'][0:param['Np'],Nim].reshape([param['m'],param['n']]),cmap='jet',clim=[-0.4, 0.4])
        ax4.plot(xCyl1, yCyl1, 'k')
        ax4.plot(xCyl, yCyl2, 'k')
        ax4.plot(xCyl, yCyl3, 'k')
        ax4.fill(xCyl, yCyl2, color='white')
        ax4.fill(xCyl, yCyl3, color='white')
        ax4.fill(xCyl1, yCyl1, color='white')
        plt.axis('tight')
        plt.axis('equal')
        ax4.set_aspect('equal', adjustable='box')
        # ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax4.title.set_text(f"Snapshot {Nim}")
        ax4.text(0.05, 0.8, 'a) Ref', transform=ax4.transAxes, color='black', fontsize=11)

        ax5.pcolormesh(grid['X'],grid['Y'],LOrecon['Urec'][0:param['Np'],Nim].reshape([param['m'],param['n']]),cmap='jet',clim=[-0.4, 0.4])
        ax5.plot(xCyl1, yCyl1, 'k')
        ax5.plot(xCyl, yCyl2, 'k')
        ax5.plot(xCyl, yCyl3, 'k')
        ax5.fill(xCyl, yCyl2, color='white')
        ax5.fill(xCyl, yCyl3, color='white')
        ax5.fill(xCyl1, yCyl1, color='white')
        plt.axis('tight')
        plt.axis('equal')
        ax5.set_aspect('equal', adjustable='box')
        # ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
        # ax2.title.set_text('b) LOR')
        ax5.text(0.05, 0.8, 'b) LOR', transform=ax5.transAxes, color='black', fontsize=11)

        ax6.pcolormesh(grid['X'],grid['Y'],GPrecon['Urec'][0:param['Np'],Nim].reshape([param['m'],param['n']]),cmap='jet',clim=[-0.4, 0.4])
        ax6.plot(xCyl1, yCyl1, 'k')
        ax6.plot(xCyl, yCyl2, 'k')
        ax6.plot(xCyl, yCyl3, 'k')
        ax6.fill(xCyl, yCyl2, color='white')
        ax6.fill(xCyl, yCyl3, color='white')
        ax6.fill(xCyl1, yCyl1, color='white')
        plt.axis('tight')
        plt.axis('equal')
        ax6.set_aspect('equal', adjustable='box')
        # ax3.set_yticks([-1, -0.5, 0, 0.5, 1])
        # ax3.title.set_text('c) GP')
        ax6.text(0.05, 0.8, 'c) GP', transform=ax6.transAxes, color='black', fontsize=11)

        f.text(0.5, 0.02, '$X / D$', ha='center', fontsize = 11, font = 'Times New Roman')
        f.text(0.01, 0.5, '$Y / D$', va='center', rotation='vertical', fontsize = 11, font = 'Times New Roman')
        cbar_ax = f.add_axes([0.93, 0.12, 0.015, 0.8]) # adjust position and size of the colorbar axis
        cb = f.colorbar(im, cax=cbar_ax)
        cb.ax.set_title('$U / U_b$', fontsize = 11, font = 'Times New Roman')
        # cb.ax.tick_params(labelsize=10) # set the font size for the colorbar tick labels
        f.subplots_adjust(left=0.09, right=0.90, bottom=0.1, top=0.95, wspace=0.1, hspace=0.1)
        # f.tight_layout()

        plt.savefig(f"{folder_path}/{sol_path}/Snapshot.png", dpi=300, bbox_inches='tight')

            
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex='col', sharey='row',figsize = (7.4,3))
        im = ax1.pcolormesh(grid['X'],grid['Y'],ORI_LOR['u_rmse'][:,].reshape([param['m'],param['n']]),cmap='jet',clim=[0,1])
        ax1.plot(xCyl1, yCyl1, 'k')
        ax1.plot(xCyl, yCyl2, 'k')
        ax1.plot(xCyl, yCyl3, 'k')
        ax1.fill(xCyl, yCyl2, color='white')
        ax1.fill(xCyl, yCyl3, color='white')
        ax1.fill(xCyl1, yCyl1, color='white')
        plt.axis('tight')
        plt.axis('equal')
        ax1.set_aspect('equal', adjustable='box')
        # ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax1.title.set_text('a) PIV-LOR')
        # ax1.text(0.05, 0.79, 'PIV-LOR', transform=ax1.transAxes, color='white', fontsize=11, fontweight='bold')

        ax2.pcolormesh(grid['X'],grid['Y'],LOR_GP['u_rmse'][:,].reshape([param['m'],param['n']]),cmap='jet',clim=[0,1])
        ax2.plot(xCyl1, yCyl1, 'k')
        ax2.plot(xCyl, yCyl2, 'k')
        ax2.plot(xCyl, yCyl3, 'k')
        ax2.fill(xCyl, yCyl2, color='white')
        ax2.fill(xCyl, yCyl3, color='white')
        ax2.fill(xCyl1, yCyl1, color='white')
        plt.axis('tight')
        plt.axis('equal')
        ax2.set_aspect('equal', adjustable='box')
        # ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax2.title.set_text('b) LOR-GP')
        # ax2.text(0.05, 0.79, 'LOR-GP', transform=ax2.transAxes, color='white', fontsize=11, fontweight='bold')

        ax3.pcolormesh(grid['X'],grid['Y'],ORI_GP['u_rmse'][:,].reshape([param['m'],param['n']]),cmap='jet',clim=[0,1])
        ax3.plot(xCyl1, yCyl1, 'k')
        ax3.plot(xCyl, yCyl2, 'k')
        ax3.plot(xCyl, yCyl3, 'k')
        ax3.fill(xCyl, yCyl2, color='white')
        ax3.fill(xCyl, yCyl3, color='white')
        ax3.fill(xCyl1, yCyl1, color='white')
        plt.axis('tight')
        plt.axis('equal')
        ax3.set_aspect('equal', adjustable='box')
        # ax3.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax3.title.set_text('c) PIV-GP')
        # ax3.text(0.05, 0.79, 'PIV-GP', transform=ax3.transAxes, color='white', fontsize=11, fontweight='bold')

        f.text(0.5, 0.28, '$X / D$', ha='center', fontsize = 11, font = 'Times New Roman')
        f.text(0.02, 0.5, '$Y / D$', va='center', rotation='vertical', fontsize = 11, font = 'Times New Roman')
        cb_ax = f.add_axes([0.07, 0.18, 0.86, 0.03])  # [left, bottom, width, height]
        cb = f.colorbar(im, cax=cb_ax, orientation='horizontal')
        cb.ax.set_xlabel('$RMSE_{u^{\prime}}$', fontsize = 11, font = 'Times New Roman')
        # cb.ax.tick_params(labelsize=10) # set the font size for the colorbar tick labels
        f.subplots_adjust(left=0.09, right=0.90, bottom=0.1, top=0.95, wspace=0.1, hspace=0.1)
        # f.tight_layout()

        plt.savefig(f'{folder_path}/{sol_path}/Error.png', dpi=300, bbox_inches='tight')


        u1 = LOrecon['Urec'][0:param['Np'],Nim].reshape([param['m'],param['n']],order='F')
        u2 = GPrecon['Urec'][0:param['Np'],Nim].reshape([param['m'],param['n']],order='F')
        aa = u1 - u2
        error_aux = np.mean(np.sqrt(np.mean((u1-u2)**2,1)),0)

        if not os.path.exists(f'{folder_path}/Matlab'):
            os.makedirs(f'{folder_path}/Matlab')
        sio.savemat(f'{folder_path}/Matlab/snap10.mat', {'TR': TR_Test['uf'][0:param['Np'],Nim].reshape([param['m'],param['n']]),'LOR': LOrecon['Urec'][0:param['Np'],Nim].reshape([param['m'],param['n']]), 'GP' : GPrecon['Urec'][0:param['Np'],Nim].reshape([param['m'],param['n']])})
        sio.savemat(f'{folder_path}/Matlab/snap16.mat', {'TR': TR_Test['uf'][0:param['Np'],Nim+5].reshape([param['m'],param['n']]), 'LOR': LOrecon['Urec'][0:param['Np'],Nim+5].reshape([param['m'],param['n']]), 'GP' : GPrecon['Urec'][0:param['Np'],Nim+5].reshape([param['m'],param['n']])})
        sio.savemat(f'{folder_path}/Matlab/grid.mat', {'X': grid['X'], 'Y':grid['Y']})
        sio.savemat(f'{folder_path}/Matlab/error.mat', {'ORI_LOR': ORI_LOR['u_rmse'][:,].reshape([param['m'],param['n']]),'LOR_GP': LOR_GP['u_rmse'][:,].reshape([param['m'],param['n']]), 'ORI_GP' : ORI_GP['u_rmse'][:,].reshape([param['m'],param['n']])})
        sio.savemat(f'{folder_path}/Matlab/error_t.mat', {'ORI_LOR': ORI_LOR['u_fom'][:,],'LOR_GP': LOR_GP['u_fom'][:,], 'ORI_GP' : ORI_GP['u_fom'][:,]})


        #%% Plot errors 
        # fig, ax = plt.subplots()
        # im = ax.pcolormesh(grid['X'],grid['Y'],ORI_GP['u_rmse'][:,].reshape([param['m'],param['n']]),cmap='jet',clim=[0,0.4])
        # ax.plot(xCyl1, yCyl1, 'k')
        # ax.plot(xCyl, yCyl2, 'k')
        # ax.plot(xCyl, yCyl3, 'k')
        # ax.fill(xCyl, yCyl2, color='white')
        # ax.fill(xCyl, yCyl3, color='white')
        # ax.fill(xCyl1, yCyl1, color='white')
        # plt.axis('tight')
        # plt.axis('equal')
        # cbar = fig.colorbar(im)
        # cbar.set_label('MSE($u^{\prime})/std(u_f)$')  # Set label and adjust its position
        # cbar.ax.xaxis.set_label_position('top')  # Place the label on top of the colorbar
        # plt.axis('equal')
        # plt.axis('tight')
        # ax.set_aspect('equal', adjustable='box')
        # plt.cm.get_cmap('bwr', 16)
        # plt.ylabel('$Y/D$')
        # plt.xlabel('$X/D$')
        # plt.title('Original vs GP - Streamwise RMSE(u)')
        # if FlagPlots:
        #     plt.savefig(f'{folder_path}/{sol_path}/u_ORI_GP.png', dpi=300, bbox_inches='tight') 


        # fig, ax = plt.subplots()
        # im = ax.pcolormesh(grid['X'],grid['Y'],LOR_GP['u_rmse'][:,].reshape([param['m'],param['n']]),cmap='jet',clim=[0,0.4])
        # ax.plot(xCyl1, yCyl1, 'k')
        # ax.plot(xCyl, yCyl2, 'k')
        # ax.plot(xCyl, yCyl3, 'k')
        # ax.fill(xCyl, yCyl2, color='white')
        # ax.fill(xCyl, yCyl3, color='white')
        # ax.fill(xCyl1, yCyl1, color='white')
        # plt.axis('tight')
        # plt.axis('equal')
        # cbar = fig.colorbar(im)
        # cbar.set_label('MSE($u^{\prime})/std(u_f)$')  # Set label and adjust its position
        # cbar.ax.xaxis.set_label_position('top')  # Place the label on top of the colorbar
        # plt.axis('equal')
        # plt.axis('tight')
        # ax.set_aspect('equal', adjustable='box')
        # plt.cm.get_cmap('bwr', 16)
        # plt.ylabel('$Y/D$')
        # plt.xlabel('$X/D$')
        # plt.title('LOR vs GP - Streamwise RMSE(u)')
        # if FlagPlots:
        #     plt.savefig(f'{folder_path}/{sol_path}/u_LOR_GP.png', dpi=300, bbox_inches='tight') 


        # fig, ax = plt.subplots()
        # im = plt.pcolormesh(grid['X'],grid['Y'],ORI_LOR['u_rmse'][:,].reshape([param['m'],param['n']]),cmap='jet',clim=[0,0.4])
        # ax.plot(xCyl1, yCyl1, 'k')
        # ax.plot(xCyl, yCyl2, 'k')
        # ax.plot(xCyl, yCyl3, 'k')
        # ax.fill(xCyl, yCyl2, color='white')
        # ax.fill(xCyl, yCyl3, color='white')
        # ax.fill(xCyl1, yCyl1, color='white')
        # plt.axis('tight')
        # plt.axis('equal')
        # cbar = fig.colorbar(im)
        # cbar.ax.xaxis.set_label_position('top')  # Place the label on top of the colorbarcbar.set_label('RMSE($u^{\prime}$)')
        # cbar.set_label('MSE($u^{\prime})/std(u_f)$')  # Set label and adjust its position
        # plt.axis('tight')
        # plt.axis('equal')
        # ax.set_aspect('equal', adjustable='box')
        # plt.cm.get_cmap('bwr', 16)
        # plt.ylabel('$Y/D$')
        # plt.xlabel('$X/D$')
        # plt.title('Original vs LOR - Streamwise RMSE(u)')
        # if FlagPlots:
        #     plt.savefig(f'{folder_path}/{sol_path}/u_ORI_LOR.png', dpi=300, bbox_inches='tight') 

    
        # fig, ax = plt.subplots()
        # im = plt.pcolormesh(grid['X'],grid['Y'],ORI_interp['u_rmse'][:,].reshape([param['m'],param['n']]),cmap='jet',clim=[0,0.4])
        # ax.plot(xCyl1, yCyl1, 'k')
        # ax.plot(xCyl, yCyl2, 'k')
        # ax.plot(xCyl, yCyl3, 'k')
        # ax.fill(xCyl, yCyl2, color='white')
        # ax.fill(xCyl, yCyl3, color='white')
        # ax.fill(xCyl1, yCyl1, color='white')
        # plt.axis('tight')
        # plt.axis('equal')
        # cbar = fig.colorbar(im)
        # cbar.ax.xaxis.set_label_position('top')  # Place the label on top of the colorbarcbar.set_label('RMSE($u^{\prime}$)')
        # cbar.set_label('MSE($u^{\prime})/std(u_f)$')  # Set label and adjust its position
        # plt.axis('tight')
        # plt.axis('equal')
        # ax.set_aspect('equal', adjustable='box')
        # plt.cm.get_cmap('bwr', 16)
        # plt.ylabel('$Y/D$')
        # plt.xlabel('$X/D$')
        # plt.title('LOR vs interp- Streamwise RMSE(u)')
        # if FlagPlots:
        #     plt.savefig(f'{folder_path}/{sol_path}/u_LOR_interp.png', dpi=300, bbox_inches='tight') 
            
            


        #%% Plot Figures of merit
        plt.figure()
        plt.plot(LOR_GP['V_fom'],label='LOR-GP', linestyle='-.', color='blue')
        plt.plot(ORI_GP['V_fom'],label='Ori-GP',linestyle='-', color='blue')
        plt.plot(ORI_LOR['V_fom'],label='Ori-LOR', linestyle='-', color='black')
        plt.plot(LOR_interp['V_fom'],label='LOR_interp', linestyle='-.', color='green')
        plt.plot(ORI_interp['V_fom'],label='ORI_interp', linestyle='-', color='green')
        plt.ylabel('$MSE$')
        plt.xlabel('$\Delta N_{snapshot}$')
        # plt.ylim(0,1)
        plt.grid()
        plt.title('Streamwise RMSE vs snapshot separation')

        # Adding a legend
        plt.legend()
        # Showing the plot
        plt.show()

        if FlagPlots:
            plt.savefig(f'{folder_path}/{sol_path}/V_FOM.png', dpi=300, bbox_inches='tight') 
            
            
        # plt.figure()
        # plt.plot(LOR_GP['v_fom'],label='LOR-GP', linestyle='-.', color='blue')
        # plt.plot(ORI_GP['v_fom'],label='Ori-GP',linestyle='-', color='blue')
        # plt.plot(ORI_LOR['v_fom'],label='Ori-LOR', linestyle='-', color='black')
        # plt.plot(LOR_interp['v_fom'],label='LOR_interp', linestyle='-.', color='green')
        # plt.plot(ORI_interp['v_fom'],label='ORI_interp', linestyle='-', color='green')
        # plt.ylabel('$MSE_{v_f}$')
        # plt.xlabel('$\Delta N_{snapshot}$')
        # # plt.ylim(0,1)
        # plt.grid()
        # plt.title('Spanwise RMSE(v) vs snapshot separation')

        # # Adding a legend
        # plt.legend()
        # # Showing the plot
        # plt.show()

        # if FlagPlots:
        #     # plt.savefig(r'{folder_path}\figures\v_FOM.png', dpi=300, bbox_inches='tight') 
        #     plt.savefig(f'{folder_path}/{sol_path}/v_FOM.png', dpi=300, bbox_inches='tight') 
        
        #%% Plot temporal modes
        
        fig, axes = plt.subplots(3, 1, figsize=(8, 12))
        
        time_final = a_recon.shape[0]
        
                
        # Now, axes is a 1-dimensional array containing the subplot axes
        # You can access each subplot using indexing, e.g., axes[0], axes[1], axes[2]
        # Example usage:
        axes[0].plot(TR_Test['t'][0:time_final],TR_Test['a'][modes[0]-1,0:time_final] , linestyle='-', color='black', label = 'PIV')  # Plot on the first subplot
        axes[0].plot(TR_Test['t'][0:time_final],a_recon[0:time_final,modes[0]] , linestyle='-.', color='blue', label = 'GP')  # Plot on the first subplot
        axes[0].plot(TR_Test['t'][0:time_final],a_interp['a'][0:time_final,modes[0]-1] , linestyle='-.', color='green', label = 'interp')  # Plot on the first subplot
        axes[0].plot(NTR_Test['t'][0:int(time_final/param['dsnaps'])],GPrecon['a'][0:int(time_final/param['dsnaps']),modes[0]-1] , linestyle='', marker='*', color='red', label = 'NTR')  # Plot on the first subplot

        # axes[0].set_title('First Subplot')
        axes[0].set_xlabel('')
        axes[0].set_ylabel(f'$a_{modes[0]}$')
        axes[0].legend()
        
        axes[1].plot(TR_Test['t'][0:time_final],TR_Test['a'][modes[1]-1,0:time_final] , linestyle='-', color='black', label = 'PIV')  # Plot on the first subplot
        axes[1].plot(TR_Test['t'][0:time_final],a_recon[0:time_final,modes[1]] , linestyle='-.', color='blue', label = 'GP')  # Plot on the first subplot
        axes[1].plot(TR_Test['t'][0:time_final],a_interp['a'][0:time_final,modes[1]-1] , linestyle='-.', color='green', label = 'interp')  # Plot on the first subplot
        axes[1].plot(NTR_Test['t'][0:int(time_final/param['dsnaps'])],GPrecon['a'][0:int(time_final/param['dsnaps']),modes[1]-1] , linestyle='', marker='*', color='red', label = 'NTR')  # Plot on the first subplot
        # axes[0].set_title('First Subplot')
        axes[1].set_xlabel('')
        axes[1].set_ylabel(f'$a_{modes[1]}$')
        axes[1].legend()
        
        axes[2].plot(TR_Test['t'][0:time_final],TR_Test['a'][modes[2]-1,0:time_final] , linestyle='-', color='black', label = 'PIV')  # Plot on the first subplot
        axes[2].plot(TR_Test['t'][0:time_final],a_recon[0:time_final,modes[2]] , linestyle='-.', color='blue', label = 'GP')  # Plot on the first subplot
        axes[2].plot(TR_Test['t'][0:time_final],a_interp['a'][0:time_final,modes[2]-1] , linestyle='-.', color='green', label = 'interp')  # Plot on the first subplot
        axes[2].plot(NTR_Test['t'][0:int(time_final/param['dsnaps'])],GPrecon['a'][0:int(time_final/param['dsnaps']),modes[2]-1] , linestyle='', marker='*', color='red', label = 'NTR')  # Plot on the first subplot
        # axes[0].set_title('First Subplot')
        axes[2].set_xlabel('')
        axes[2].set_ylabel(f'$a_{modes[2]}$')
        # Add legend
        axes[2].legend()
        # Add a common title to all subplots
        fig.suptitle('Temporal modes', fontsize=16)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        
        # Show the plot
        plt.show()

        if FlagPlots:
            plt.savefig(f'{folder_path}/{sol_path}/modes.png', dpi=300, bbox_inches='tight') 

        #%% Videos
                
        if FlagVideos:
            # def make_animation_3(grid, LOrecon, GPrecon, param):
            #     # Create a figure and subplots for the animation
            #     fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(25, 5))
                
            #     # Define global variables for text annotations
            #     global snapshot_text0, snapshot_text1, snapshot_text2
            #     snapshot_text0 = None
            #     snapshot_text1 = None
            #     snapshot_text2 = None
                
            #     # Define the first plot (left)
            #     fond0 = ax0.pcolormesh(grid['X'], grid['Y'], TR_Test['uf'][0:param['Np'], 0].reshape([param['m'], param['n']]), cmap='jet', clim=[-0.4, 0.4])
            #     snapshot_text0 = ax0.text(0.02, 0.95, '', transform=ax0.transAxes, color='white', fontsize=10)
            #     ax0.plot(xCyl1, yCyl1, 'k')
            #     ax0.plot(xCyl, yCyl2, 'k')
            #     ax0.plot(xCyl, yCyl3, 'k')
            #     ax0.fill(xCyl, yCyl2, color='white')
            #     ax0.fill(xCyl, yCyl3, color='white')
            #     ax0.fill(xCyl1, yCyl1, color='white')
            #     plt.axis('tight')
            #     plt.axis('equal')
            #     cbar0 = fig.colorbar(fond0, ax=ax0)
            #     cbar0.set_label('$u^{\prime}/U_{\infty}$', rotation=90, labelpad=20)
            #     ax0.axis('tight')
            #     ax0.axis('equal')
            #     # ax0.set_title('Cumulative Energy')        
            #     ax0.text(0.05, 0.8, 'a) PIV', transform=ax0.transAxes, color='black', fontsize=11)
            #     plt.ylabel('$Y/D$')
            #     plt.xlabel('$X/D$')
                
            #     # Define the second plot (middle)
            #     fond1 = ax1.pcolormesh(grid['X'], grid['Y'], LOrecon['Urec'][0:param['Np'], 0].reshape([param['m'], param['n']]), cmap='jet', clim=[-0.4, 0.4])
            #     snapshot_text1 = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, color='white', fontsize=10)
            #     ax1.plot(xCyl1, yCyl1, 'k')
            #     ax1.plot(xCyl, yCyl2, 'k')
            #     ax1.plot(xCyl, yCyl3, 'k')
            #     ax1.fill(xCyl, yCyl2, color='white')
            #     ax1.fill(xCyl, yCyl3, color='white')
            #     ax1.fill(xCyl1, yCyl1, color='white')
            #     plt.axis('tight')
            #     plt.axis('equal')
            #     cbar1 = fig.colorbar(fond1, ax=ax1)
            #     cbar1.set_label('$u^{\prime}/U_{\infty}$', rotation=90, labelpad=20)
            #     ax1.axis('tight')
            #     ax1.axis('equal')
            #     ax1.text(0.05, 0.8, 'b) LOR', transform=ax1.transAxes, color='black', fontsize=11)
            #     plt.ylabel('$Y/D$')
            #     plt.xlabel('$X/D$')
                
            #     # Define the third plot (right)
            #     fond2 = ax2.pcolormesh(grid['X'], grid['Y'], GPrecon['Urec'][0:param['Np'], 0].reshape([param['m'], param['n']]), cmap='jet', clim=[-0.4, 0.4])
            #     snapshot_text2 = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, color='white', fontsize=10)
            #     cbar2 = fig.colorbar(fond2, ax=ax2)
            #     cbar2.set_label('$u^{\prime}/U_{\infty}$', rotation=90, labelpad=20)
            #     ax2.plot(xCyl1, yCyl1, 'k')
            #     ax2.plot(xCyl, yCyl2, 'k')
            #     ax2.plot(xCyl, yCyl3, 'k')
            #     ax2.fill(xCyl, yCyl2, color='white')
            #     ax2.fill(xCyl, yCyl3, color='white')
            #     ax2.fill(xCyl1, yCyl1, color='white')
            #     plt.axis('tight')
            #     plt.axis('equal')
            #     ax2.axis('tight')
            #     ax2.axis('equal')
            #     ax2.text(0.05, 0.8, 'c) GP', transform=ax2.transAxes, color='black', fontsize=11)
            #     plt.ylabel('$Y/D$')
            #     plt.xlabel('$X/D$')
                
            #     # Function to update the plots for each frame of the animation
            #     def animate(t):
            #         global snapshot_text0, snapshot_text1, snapshot_text2
            
            #         # Update the first plot
            #         ax0.clear()
            #         # Update the second plot
            #         ax1.clear()
            #         # Update the third plot
            #         ax2.clear()
            
            #         # Update the first plot
            #         fond0 = ax0.pcolormesh(grid['X'], grid['Y'], TR_Test['uf'][0:param['Np'], t].reshape([param['m'], param['n']]), cmap='jet', clim=[-0.4, 0.4])
            #         if  (t ) % param['dsnaps'] == 0:  # Every 10th frame
            #             snapshot_text0 = ax0.text(0.5, 0.9, f'Snapshot: {t + 1}', transform=ax0.transAxes, ha='center', fontsize=12, color='red')
            #         else:
            #             snapshot_text0 = ax0.text(0.5, 0.9, f'Snapshot: {t + 1}', transform=ax0.transAxes, ha='center', fontsize=12, color='black')
            #         ax0.plot(xCyl1, yCyl1, 'k')
            #         ax0.plot(xCyl, yCyl2, 'k')
            #         ax0.plot(xCyl, yCyl3, 'k')
            #         ax0.fill(xCyl, yCyl2, color='white')
            #         ax0.fill(xCyl, yCyl3, color='white')
            #         ax0.fill(xCyl1, yCyl1, color='white')
            #         plt.axis('tight')
            #         plt.axis('equal')
            #         ax0.axis('tight')
            #         ax0.axis('equal')
            #         ax0.text(0.05, 0.8, 'a) PIV', transform=ax0.transAxes, color='black', fontsize=11)
            #         plt.ylabel('$Y/D$')
            #         plt.xlabel('$X/D$')
            
            #         # Update the second plot
            #         fond1 = ax1.pcolormesh(grid['X'], grid['Y'], LOrecon['Urec'][0:param['Np'], t].reshape([param['m'], param['n']]), cmap='jet', clim=[-0.4, 0.4])
            #         if  (t ) %  param['dsnaps'] == 0:  # Every 10th frame
            #             snapshot_text1 = ax1.text(0.5, 0.9, f'Snapshot: {t + 1}', transform=ax1.transAxes, ha='center', fontsize=12, color='red')
            #         else:
            #             snapshot_text1 = ax1.text(0.5, 0.9, f'Snapshot: {t + 1}', transform=ax1.transAxes, ha='center', fontsize=12, color='black')
            #         ax1.plot(xCyl1, yCyl1, 'k')
            #         ax1.plot(xCyl, yCyl2, 'k')
            #         ax1.plot(xCyl, yCyl3, 'k')
            #         ax1.fill(xCyl, yCyl2, color='white')
            #         ax1.fill(xCyl, yCyl3, color='white')
            #         ax1.fill(xCyl1, yCyl1, color='white')
            #         plt.axis('tight')
            #         plt.axis('equal')
            #         ax1.axis('tight')
            #         ax1.axis('equal')
            #         ax1.text(0.05, 0.8, 'b) LOR', transform=ax1.transAxes, color='black', fontsize=11)
            #         plt.ylabel('$Y/D$')
            #         plt.xlabel('$X/D$')
            
            #         # Update the third plot
            #         fond2 = ax2.pcolormesh(grid['X'], grid['Y'], GPrecon['Urec'][0:param['Np'], t].reshape([param['m'], param['n']]), cmap='jet', clim=[-0.4, 0.4])
            #         if  (t ) % param['dsnaps'] == 0:  # Every 10th frame
            #             snapshot_text2 = ax2.text(0.5, 0.9, f'Snapshot: {t + 1}', transform=ax2.transAxes, ha='center', fontsize=12, color='red')
            #         else:
            #             snapshot_text2 = ax2.text(0.5, 0.9, f'Snapshot: {t + 1}', transform=ax2.transAxes, ha='center', fontsize=12, color='black')
            #         ax2.plot(xCyl1, yCyl1, 'k')
            #         ax2.plot(xCyl, yCyl2, 'k')
            #         ax2.plot(xCyl, yCyl3, 'k')
            #         ax2.fill(xCyl, yCyl2, color='white')
            #         ax2.fill(xCyl, yCyl3, color='white')
            #         ax2.fill(xCyl1, yCyl1, color='white')
            #         plt.axis('tight')
            #         plt.axis('equal')
            #         ax2.axis('tight')
            #         ax2.axis('equal')
            #         ax2.text(0.05, 0.8, 'c) GP', transform=ax2.transAxes, color='black', fontsize=11)
            #         plt.ylabel('$Y/D$')
            #         plt.xlabel('$X/D$')
            
            #         return fond0, fond1, fond2, snapshot_text0, snapshot_text1, snapshot_text2
        
            #     # Define the custom interval function
            #     def interval_function(frame):
            #         if (frame ) in NTR_Test['index']:  # Every 10th frame
            #             return 500  # Longer interval (e.g., 500 milliseconds)
            #         else:
            #             return 100   # Shorter interval (e.g., 100 milliseconds)
            
            #     # Create the animation object with the custom interval
            #     # ani = animation.FuncAnimation(fig, animate, frames=400, interval=lambda frame: interval_function(frame))
            #     ani = animation.FuncAnimation(fig, animate, frames=GPrecon['Urec'].shape[1]-1, interval=20)  # 100 milliseconds interval
    
            
            #     # Save the animation as an MP4 video
            #     current_directory = os.getcwd()
            #     video_path = os.path.join(current_directory, f'{folder_path}/{sol_path}/video_3.mp4')
            #     writer = animation.FFMpegWriter(fps=2, codec='h264')
            #     ani.save(video_path, writer=writer)
            
            #     plt.show()
                
            # make_animation_3(grid, LOrecon, GPrecon, param)
            
            #%% 
    
            def make_animation_4(grid, TR_Test, LOrecon, GPrecon, a_interp, param):
                import matplotlib.pyplot as plt
                from matplotlib import animation
                import os
                
                # Create a figure and subplots for the animation
                fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(15, 10))
                
                # Define global variables for text annotations
                global snapshot_text0, snapshot_text1, snapshot_text2, snapshot_text3
                snapshot_text0 = None
                snapshot_text1 = None
                snapshot_text2 = None
                snapshot_text3 = None
                
                # Define the first plot (left)
                fond0 = ax0.pcolormesh(grid['X'], grid['Y'], TR_Test['uf'][0:param['Np'], 0].reshape([param['m'], param['n']]), cmap='jet', clim=[-0.4, 0.4])
                snapshot_text0 = ax0.text(0.02, 0.95, 'a) PIV', transform=ax0.transAxes, color='white', fontsize=10)
                cbar0 = fig.colorbar(fond0, ax=ax0)
                cbar0.set_label('$u^{\prime}/U_{\infty}$', rotation=90, labelpad=20)
                ax0.plot(xCyl1, yCyl1, 'k')
                ax0.plot(xCyl, yCyl2, 'k')
                ax0.plot(xCyl, yCyl3, 'k')
                ax0.fill(xCyl, yCyl2, color='white')
                ax0.fill(xCyl, yCyl3, color='white')
                ax0.fill(xCyl1, yCyl1, color='white')
                plt.axis('tight')
                plt.axis('equal')
                ax0.axis('tight')
                ax0.axis('equal')
                ax0.text(0.05, 0.8, 'a) PIV', transform=ax0.transAxes, color='black', fontsize=11)
                plt.ylabel('$Y/D$')
                plt.xlabel('$X/D$')
                
                # Define the second plot (middle left)
                fond1 = ax1.pcolormesh(grid['X'], grid['Y'], LOrecon['Urec'][0:param['Np'], 0].reshape([param['m'], param['n']]), cmap='jet', clim=[-0.4, 0.4])
                snapshot_text1 = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, color='white', fontsize=10)
                cbar1 = fig.colorbar(fond1, ax=ax1)
                cbar1.set_label('$u^{\prime}/U_{\infty}$', rotation=90, labelpad=20)
                ax1.plot(xCyl1, yCyl1, 'k')
                ax1.plot(xCyl, yCyl2, 'k')
                ax1.plot(xCyl, yCyl3, 'k')
                ax1.fill(xCyl, yCyl2, color='white')
                ax1.fill(xCyl, yCyl3, color='white')
                ax1.fill(xCyl1, yCyl1, color='white')
                ax1.axis('tight')
                ax1.axis('equal')
                ax1.axis('tight')
                ax1.axis('equal')
                ax1.text(0.05, 0.8, 'b) LOR', transform=ax1.transAxes, color='black', fontsize=11)
                plt.ylabel('$Y/D$')
                plt.xlabel('$X/D$')
                
                # Define the third plot (middle right)
                fond2 = ax2.pcolormesh(grid['X'], grid['Y'], GPrecon['Urec'][0:param['Np'], 0].reshape([param['m'], param['n']]), cmap='jet', clim=[-0.4, 0.4])
                snapshot_text2 = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, color='white', fontsize=10)
                cbar2 = fig.colorbar(fond2, ax=ax2)
                cbar2.set_label('$u^{\prime}/U_{\infty}$', rotation=90, labelpad=20)
                ax2.plot(xCyl1, yCyl1, 'k')
                ax2.plot(xCyl, yCyl2, 'k')
                ax2.plot(xCyl, yCyl3, 'k')
                ax2.fill(xCyl, yCyl2, color='white')
                ax2.fill(xCyl, yCyl3, color='white')
                ax2.fill(xCyl1, yCyl1, color='white')
                plt.axis('tight')
                plt.axis('equal')
                ax2.axis('tight')
                ax2.axis('equal')
                ax2.text(0.05, 0.8, 'c) GP', transform=ax2.transAxes, color='black', fontsize=11)
                plt.ylabel('$Y/D$')
                plt.xlabel('$X/D$')
                
                # Define the fourth plot (right)
                fond3 = ax3.pcolormesh(grid['X'], grid['Y'], a_interp['Urec'][0:param['Np'], 0].reshape([param['m'], param['n']]), cmap='jet', clim=[-0.4, 0.4])
                snapshot_text3 = ax3.text(0.02, 0.95, '', transform=ax3.transAxes, color='white', fontsize=10)
                cbar3 = fig.colorbar(fond3, ax=ax3)
                cbar3.set_label('$u^{\prime}/U_{\infty}$', rotation=90, labelpad=20)
                ax3.plot(xCyl1, yCyl1, 'k')
                ax3.plot(xCyl, yCyl2, 'k')
                ax3.plot(xCyl, yCyl3, 'k')
                ax3.fill(xCyl, yCyl2, color='white')
                ax3.fill(xCyl, yCyl3, color='white')
                ax3.fill(xCyl1, yCyl1, color='white')
                plt.axis('tight')
                plt.axis('equal')
                ax3.axis('tight')
                ax3.axis('equal')
                ax3.text(0.05, 0.8, 'd) Interp', transform=ax3.transAxes, color='black', fontsize=11)
                plt.ylabel('$Y/D$')
                plt.xlabel('$X/D$')
                
                # Function to update the plots for each frame of the animation
                def animate(t):
                    global snapshot_text0, snapshot_text1, snapshot_text2, snapshot_text3
            
                    # Update the first plot
                    ax0.clear()
                    # Update the second plot
                    ax1.clear()
                    # Update the third plot
                    ax2.clear()
                    # Update the fourth plot
                    ax3.clear()
            
                    # Update the first plot
                    fond0 = ax0.pcolormesh(grid['X'], grid['Y'], LOrecon['Urec'][0:param['Np'], t].reshape([param['m'], param['n']]), cmap='jet', clim=[-0.4, 0.4])
                    if (t) % 10 == 0:  # Every 10th frame
                        snapshot_text0 = ax0.text(0.5, 0.9, f'Snapshot: {t + 1}', transform=ax0.transAxes, ha='center', fontsize=12, color='red')
                    else:
                        snapshot_text0 = ax0.text(0.5, 0.9, f'Snapshot: {t + 1}', transform=ax0.transAxes, ha='center', fontsize=12, color='black')
                    ax0.plot(xCyl1, yCyl1, 'k')
                    ax0.plot(xCyl, yCyl2, 'k')
                    ax0.plot(xCyl, yCyl3, 'k')
                    ax0.fill(xCyl, yCyl2, color='white')
                    ax0.fill(xCyl, yCyl3, color='white')
                    ax0.fill(xCyl1, yCyl1, color='white')
                    plt.axis('tight')
                    plt.axis('equal')
                    ax0.axis('tight')
                    ax0.axis('equal')
                    ax0.text(0.05, 0.8, 'a) LOR', transform=ax0.transAxes, color='black', fontsize=11)
                    plt.ylabel('$Y/D$')
                    plt.xlabel('$X/D$')
            
                    # Update the second plot
                    fond1 = ax1.pcolormesh(grid['X'], grid['Y'], LOrecon['Urec'][0:param['Np'], t].reshape([param['m'], param['n']]), cmap='jet', clim=[-0.4, 0.4])
                    if (t) % param['dsnaps'] == 0:  # Every 10th frame
                        snapshot_text1 = ax1.text(0.5, 0.9, f'Snapshot: {t + 1}', transform=ax1.transAxes, ha='center', fontsize=12, color='red')
                    else:
                        snapshot_text1 = ax1.text(0.5, 0.9, f'Snapshot: {t + 1}', transform=ax1.transAxes, ha='center', fontsize=12, color='black')
                    
                    ax1.plot(xCyl1, yCyl1, 'k')
                    ax1.plot(xCyl, yCyl2, 'k')
                    ax1.plot(xCyl, yCyl3, 'k')
                    ax1.fill(xCyl, yCyl2, color='white')
                    ax1.fill(xCyl, yCyl3, color='white')
                    ax1.fill(xCyl1, yCyl1, color='white')
                    plt.axis('tight')
                    plt.axis('equal')
                    ax1.axis('tight')
                    ax1.axis('equal')
                    ax1.text(0.05, 0.8, 'b) LOR', transform=ax1.transAxes, color='black', fontsize=11)
                    plt.ylabel('$Y/D$')
                    plt.xlabel('$X/D$')
            
                    # Update the third plot
                    fond2 = ax2.pcolormesh(grid['X'], grid['Y'], GPrecon['Urec'][0:param['Np'], t].reshape([param['m'], param['n']]), cmap='jet', clim=[-0.4, 0.4])
                    if (t) % param['dsnaps'] == 0:  # Every 10th frame
                        snapshot_text2 = ax2.text(0.5, 0.9, f'Snapshot: {t + 1}', transform=ax2.transAxes, ha='center', fontsize=12, color='red')
                    else:
                        snapshot_text2 = ax2.text(0.5, 0.9, f'Snapshot: {t + 1}', transform=ax2.transAxes, ha='center', fontsize=12, color='black')
                    
                    ax2.plot(xCyl1, yCyl1, 'k')
                    ax2.plot(xCyl, yCyl2, 'k')
                    ax2.plot(xCyl, yCyl3, 'k')
                    ax2.fill(xCyl, yCyl2, color='white')
                    ax2.fill(xCyl, yCyl3, color='white')
                    ax2.fill(xCyl1, yCyl1, color='white')
                    plt.axis('tight')
                    plt.axis('equal')
                    ax2.axis('tight')
                    ax2.axis('equal')
                    ax2.text(0.05, 0.8, 'c) GP', transform=ax2.transAxes, color='black', fontsize=11)
                    
                    plt.ylabel('$Y/D$')
                    plt.xlabel('$X/D$')
            
                    # Update the fourth plot
                    fond3 = ax3.pcolormesh(grid['X'], grid['Y'], a_interp['Urec'][0:param['Np'], t].reshape([param['m'], param['n']]), cmap='jet', clim=[-0.4, 0.4])
                    if (t) % param['dsnaps'] == 0:  # Every 10th frame
                        snapshot_text3 = ax3.text(0.5, 0.9, f'Snapshot: {t + 1}', transform=ax3.transAxes, ha='center', fontsize=12, color='red')
                    else:
                        snapshot_text3 = ax3.text(0.5, 0.9, f'Snapshot: {t + 1}', transform=ax3.transAxes, ha='center', fontsize=12, color='black')
                    
                    ax3.plot(xCyl1, yCyl1, 'k')
                    ax3.plot(xCyl, yCyl2, 'k')
                    ax3.plot(xCyl, yCyl3, 'k')
                    ax3.fill(xCyl, yCyl2, color='white')
                    ax3.fill(xCyl, yCyl3, color='white')
                    ax3.fill(xCyl1, yCyl1, color='white')
                    plt.axis('tight')
                    plt.axis('equal')
                    ax3.axis('tight')
                    ax3.axis('equal')
                    ax3.text(0.05, 0.8, 'd) Interp', transform=ax3.transAxes, color='black', fontsize=11)
                    plt.ylabel('$Y/D$')
                    plt.xlabel('$X/D$')
            
                    return fond0, fond1, fond2, fond3, snapshot_text0, snapshot_text1, snapshot_text2, snapshot_text3
                
                # Define the custom interval function
                def interval_function(frame):
                    if (frame) in NTR_Test['index']:  # Every 10th frame
                        return 500  # Longer interval (e.g., 500 milliseconds)
                    else:
                        return 100   # Shorter interval (e.g., 100 milliseconds)
                
                # Create the animation object with the custom interval
                ani = animation.FuncAnimation(fig, animate, frames=GPrecon['Urec'].shape[1]-1, interval=100)  # 100 milliseconds interval
            
                # Save the animation as an MP4 video
                current_directory = os.getcwd()
                video_path = os.path.join(current_directory, f'{folder_path}/{sol_path}/video_4.mp4')
                writer = animation.FFMpegWriter(fps=10, codec='h264')
                ani.save(video_path, writer=writer)
                
                plt.show()
            
            make_animation_4(grid, TR_Test, LOrecon, GPrecon, a_interp, param)
                        


if __name__ == '__main__':

    main()
    
    # # Run mprof to profile memory usage
    # subprocess.run(["mprof", "run", "main_GP.py"])
    
    # # Generate memory usage plot
    # subprocess.run(["mprof", "plot"])
    
    # cProfile.run('main()')

    # Run following lines in the Console to get memory profiler
    # %load_ext memory_profiler
    # %memit main()
    

    # !mprof run main_GP.py
    # %mprof plot
    # mprof plot -t ‘Recorded memory usage’
    # %cd {FP_results}
    # !mprof plot