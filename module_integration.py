# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:41:49 2023

@author: QihongLi
"""
# Import packages

import numpy as np
import scipy.io as sio
import numpy.matlib
# from scipy.integrate import solve_ivp
from scipy.integrate import odeint

from tqdm import tqdm
from functools import partial

from concurrent.futures import ThreadPoolExecutor
from tqdm.contrib.concurrent import thread_map
from multiprocessing import Pool
from memory_profiler import profile

import warnings

import module_GalerkinProjection as gp

#%%

@profile
def integrator(TR_Test,NTR_Test,GPrecon,GPcoeff,FlagPressure,param):
    a = []       # Initialise list
    nt = int(NTR_Test['t'][1]/param['dt']) + 1
    r = param['r']
    
    for i in tqdm(range(0,NTR_Test['index'].shape[0]-1), desc="Integration Progress"):

        # building initial condition for both the integrations
        a_IC_f = GPrecon['Phir'][:,:].T@NTR_Test['uf'][:,i]
        a_IC_f [0,] = 1
        a_IC_b = GPrecon['Phir'][:,:].T@NTR_Test['uf'][:,i+1]
        a_IC_b [0,] = 1
        
        # time vector for the integration
        # tvec_f = np.arange(TR_Test['t'][0,NTR_Test['index'][i]],TR_Test['t'][0,NTR_Test['index'][i+1]]+param['dt'],param['dt'])
        tvec_f = np.linspace(TR_Test['t'][NTR_Test['index'][i]],TR_Test['t'][NTR_Test['index'][i+1]],nt)
        tvec_b = tvec_f[::-1]
        
        
        #collecting results of the ODEs for the specific range
        result_f = odeint(gp.myODESystem, a_IC_f, tvec_f, args=(r, param['nu'], GPcoeff['l'], GPcoeff['lp'], GPcoeff['qc'], FlagPressure))
        result_b = odeint(gp.myODESystem, a_IC_b, tvec_b, args=(r, param['nu'], GPcoeff['l'], GPcoeff['lp'], GPcoeff['qc'], FlagPressure))
        
        #reallocating the initial conditions
        result_f[0,:] = a_IC_f
        result_b[0,:] = a_IC_b
        
        result_b = result_b[::-1]
        
        #  weights for the average of the integrations
        w_b = np.linspace(0,1,result_f.shape[0])#**3
        w_f = w_f = w_b[::-1]
      
        a_aux = np.zeros([result_f.shape[0]-1,result_f.shape[1]])          # Initialise list
        for j in range(0,result_f.shape[1]):
            # weighting the results
            a_aux[:,j] = result_f[0:-1,j]*w_f[0:-1] + result_b[0:-1,j]*w_b[0:-1] 
        # update the list of a
        a.append(a_aux)
        
    a_recon = np.vstack(a)
    
    return a_recon

@profile
def integrator_eins(TR_Test,NTR_Test,GPrecon,GPcoeff,FlagPressure,param):
    a = []       # Initialise list
    nt = int(NTR_Test['t'][1]/param['dt']) + 1
    r = param['r']
    
    model = lambda t, a: gp.galerkin_model(a, GPcoeff['l'], GPcoeff['qc'])
    t_span = (NTR_Test['t'][0], NTR_Test['t'][-1])
    
    # for i in tqdm(range(0,NTR_Test['index'].shape[0]-1), desc="Integration Progress"):
    for i in tqdm(range(0,40), desc="Integration Progress"):

        # building initial condition for both the integrations
        a_IC_f = GPrecon['Phir'][:,:].T@NTR_Test['uf'][:,i]
        a_IC_f [0,] = 1
        a_IC_b = GPrecon['Phir'][:,:].T@NTR_Test['uf'][:,i+1]
        a_IC_b [0,] = 1
        
        # time vector for the integration
        # tvec_f = np.arange(TR_Test['t'][0,NTR_Test['index'][i]],TR_Test['t'][0,NTR_Test['index'][i+1]]+param['dt'],param['dt'])
        tvec_f = np.linspace(TR_Test['t'][NTR_Test['index'][i]],TR_Test['t'][NTR_Test['index'][i+1]],nt)
        tvec_b = tvec_f[::-1]
        t_span_f = (NTR_Test['t'][i], NTR_Test['t'][i+1])
        t_span_b = (NTR_Test['t'][i+1], NTR_Test['t'][i])
        
        #collecting results of the ODEs for the specific range
        a_galerkin_f = solve_ivp(model, t_span_f, a_IC_f, t_eval=tvec_f,
                                **integrator_keywords).y.T
        a_galerkin_b = solve_ivp(model, t_span_b, a_IC_b, t_eval=tvec_b,
                                **integrator_keywords).y.T
        
        # result_f = odeint(gp.myODESystem, a_IC_f, tvec_f, args=(r, param['nu'], GPcoeff['l'], GPcoeff['lp'], GPcoeff['qc'], FlagPressure))
        # result_b = odeint(gp.myODESystem, a_IC_b, tvec_b, args=(r, param['nu'], GPcoeff['l'], GPcoeff['lp'], GPcoeff['qc'], FlagPressure))
        
        # #reallocating the initial conditions
        # result_f[0,:] = a_IC_f
        # result_b[0,:] = a_IC_b
        
        # result_b = result_b[::-1]
        
        #  weights for the average of the integrations
        w_b = np.linspace(0,1,result_f.shape[0])#**3
        w_f = w_f = w_b[::-1]
      
        a_aux = np.zeros([a_galerkin_f.shape[0]-1,a_galerkin_f.shape[1]])          # Initialise list
        for j in range(0,result_f.shape[1]):
            # weighting the results
            a_aux[:,j] = a_galerkin_f[0:-1,j]*w_f[0:-1] + a_galerkin_b[0:-1,j]*w_b[0:-1] 
        # update the list of a
        a.append(a_aux)
        
    a_recon = np.vstack(a)
    
    return a_recon

#%% Paralelised integrator
def integrate_worker(args):
    func, index, nparts, num_parts, TR_Test, NTR_Test, GPrecon, GPcoeff, FlagPressure, param = args
    
    nt = int(NTR_Test['t'][1]/param['dt']) + 1
    r = param['r']
    
    for i in nparts:
        # building initial condition for both the integrations
        a_IC_f = GPrecon['Phir'][:,:].T@NTR_Test['uf'][:,i]
        a_IC_f [0,] = 1
        a_IC_b = GPrecon['Phir'][:,:].T@NTR_Test['uf'][:,i+1]
        a_IC_b [0,] = 1
        
        # time vector for the integration
        # tvec_f = np.arange(TR_Test['t'][0,NTR_Test['index'][i]],TR_Test['t'][0,NTR_Test['index'][i+1]]+param['dt'],param['dt'])
        tvec_f = np.linspace(TR_Test['t'][index[i]],TR_Test['t'][index[i+1]],nt)
        tvec_b = tvec_f[::-1]
        
        
        #collecting results of the ODEs for the specific range
        result_f = odeint(gp.myODESystem, a_IC_f, tvec_f, args=(r, param['nu'], GPcoeff['l'], GPcoeff['lp'], GPcoeff['qc'], FlagPressure))
        result_b = odeint(gp.myODESystem, a_IC_b, tvec_b, args=(r, param['nu'], GPcoeff['l'], GPcoeff['lp'], GPcoeff['qc'], FlagPressure))
        
        #reallocating the initial conditions
        result_f[0,:] = a_IC_f
        result_b[0,:] = a_IC_b
        
        result_b = result_b[::-1]
        
        #  weights for the average of the integrations
        w_b = np.linspace(0,1,result_f.shape[0])#**3
        w_f = w_f = w_b[::-1]
      
        a_aux = np.zeros([result_f.shape[0]-1,result_f.shape[1]])          # Initialise list
        for j in range(0,result_f.shape[1]):
            # weighting the results
            a_aux[:,j] = result_f[0:-1,j]*w_f[0:-1] + result_b[0:-1,j]*w_b[0:-1] 
        # update the list of a
        a.append(a_aux)
    
    
    return a


def parallel_integrate(func, index, num_parts, num_processes,TR_Test, NTR_Test, GPrecon, GPcoeff, FlagPressure, param):

    # Calculate the number of items per part
    nparts, remainder = divmod(index.shape[0], num_parts)
    
    # Create a Pool of workers
    with Pool(processes=num_processes) as pool:
        args = []
        start = 0
        for i in range(num_parts):
            end = start + nparts + (1 if i < remainder else 0)
            args.append((func, index[start:end], nparts, num_parts, TR_Test ,NTR_Test, GPrecon ,GPcoeff ,FlagPressure, param))
            start = end
        
        # Apply the integrate_worker function to each argument asynchronously
        with tqdm(total=len(args)) as pbar:
            results = [pool.apply_async(integrate_worker, arg) for arg in args]
            pbar.update(len(args))

        # Get the results in order
        results = [res.get() for res in results]
    
    # # Sum up the results to get the total integral
    # total_integral = sum(results)
    
    a_recon = np.vstack(results)
    return a_recon

















