# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 18:24:46 2023

@author: QihongLi
"""

import os                                           # For creating folders

#%%

def check_file_existence(file_path):
    if os.path.exists(file_path):
        return 0  
    
    else:
        return 1  
    

    
#%% Flags configuration
def getFlags(results_folder):
    FlagLoadData = 1
    FlagPOD = 1 
    FlagGPcoeff = 1
    FlagInt = 1
        
    config = {}
    data_path = 'data\python'
    FilePath_Data = os.path.join(data_path, 'data_set.pkl')
    FlagLoadData = check_file_existence(FilePath_Data) # Check if the file exists in the target folder
    
    # Load data
    FileName_Train = 'NTR_Train.pkl'
    FilePath_Train = os.path.join(f"{results_folder}\data_set", FileName_Train)
    FlagTrain = check_file_existence(FilePath_Train) # Check if the file exists in the target folder

    FileName_Test = 'NTR_Test.pkl'
    FilePath_Test = os.path.join(f"{results_folder}\data_set", FileName_Test)
    FlagTest = check_file_existence(FilePath_Test) # Check if the file exists in the target folder
    
    if (FlagTrain == 1) or (FlagTest == 1):
        FlagPreprocess = 1
    else:
        FlagPreprocess = 0
        
        
    # Get POD of time coefficients
    Filename_POD = 'POD.pkl'
    FilePath_POD = os.path.join(results_folder, Filename_POD)
    FlagPOD = check_file_existence(FilePath_POD) # Check if the file exists in the target folder
        
    # Get Galerkin Coefficients
    Filename_GP = 'GPcoeff.pkl'
    FilePath_GP = os.path.join(results_folder, Filename_GP)
    FlagGPcoeff = check_file_existence(FilePath_GP) # Check if the file exists in the target folder
    
    # Get integral of time coefficients
    Filename_res = 'GP_Urec.npy'
    FilePath_res = os.path.join(results_folder, Filename_res)
    FlagInt = check_file_existence(FilePath_res) # Check if the file exists in the target folder
    
    

    # return FlagLoadData, FlagPreprocess, FlagGPcoeff, FlagPOD, FlagInt 
    return FlagLoadData, FlagGPcoeff, FlagPOD, FlagInt 



