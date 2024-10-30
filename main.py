# PACKAGES
import warnings
import json
from loguru import logger
import numpy as np

# LOCAL FUNCTIONS
from modules.processing.readers import read_h5, load_results, save_results, check_inputs
from modules.processing.datasets import create_train_set, create_test_set
from modules.processing.POD import get_POD, project_POD, truncate_POD
from modules.dynamics.galerkin_coefs import galerkin_coefs, sparsify_coeffs
from modules.dynamics.integration import integrator, interpolator

def main(INPUTS):

    #%% 0. INPUT READING
    path_grid = INPUTS["path_grid"]                 # Data path for grid dictionary
    path_train = INPUTS["path_train"]               # Data path for train dataset dictionary
    path_test = INPUTS["path_test"]                 # Data path for test dataset dictionary

    Re = INPUTS["Re"]                               # Reynolds number
    truncation_value = INPUTS["truncation_value"]   # POD truncation threshold for selected method
    sparsify_tol = INPUTS["sparsify_tol"]           # Tolerance threshold for matrix of coefficients sparsification
    N_process_int = INPUTS["N_process_int"]         # Number of processes for parallel integration
    ts_train = INPUTS["ts_train"]                   # Irregular time spacing of training set in convective units
    ts_test = INPUTS["ts_test"]                     # Regular time spacing of testing set in convective units
    Dt_test = INPUTS["Dt_test"]                     # Regular time-resolved spacing of integrated testing set in c.u.

    flag_flow = INPUTS["flag_flow"]                 # Flow type ('FP' or 'Jet')
    flag_pressure = INPUTS["flag_pressure"]         # GP pressure coefficients retrieval flag ('none', 'analytical', 'empirical')
    flag_truncation = INPUTS["flag_truncation"]     # POD truncation method ('manual','energy','optimal','elbow','none')
    flag_train_res = INPUTS["flag_train_res"]       # Resolution of training read dataset ('TR', 'NTR')
    flag_test_res = INPUTS["flag_test_res"]         # Resolution of testing read dataset ('TR', 'NTR')
    flag_acceleration = INPUTS["flag_acceleration"] # Boolean to retrieve or not acceleration fields of training and testing
    flag_integration = INPUTS["flag_integration"]   # Type of integration system used ('matrix','einsum')
    flag_sparsify = INPUTS["flag_sparsify"]         # Boolean to threshold or not matrix of coefficients
    flag_loaders = INPUTS["flag_loaders"]           # List indicating entries to load ('stats','GPcoef','POD','PODr')
    flag_savers = INPUTS["flag_savers"]             # List indicating entries to save ('stats','GPcoef','POD','PODr','test_GP','test_interp')

    #%% I. DATA PREPARATION
    logger.info("I. DATA PREPARATION")
    # Read grid
    grid = read_h5(path_grid)
    logger.debug("Read grid...")

    # Read training, only if GP coeffs are to be retrieved and no POD basis is loaded
    if 'GPcoef' not in flag_loaders:
        if 'POD' not in flag_loaders and 'PODr' not in flag_loaders:
            flow = read_h5(path_train)
            logger.debug("Read training set...")

            # Create NTR training set
            train = create_train_set(grid, flow, flag_acceleration, flag_train_res, flag_pressure, ts_train)
            logger.debug("Prepared training set...")
            del flow

    # Get mean flow and stds from not loaded training set
    if 'stats' in flag_loaders:
        train = load_results('stats', flag_flow, Re, truncation_value, flag_pressure, flag_truncation)
        train['dDdt'] = [] # To avoid errors in function callings
        logger.debug("Read training statistics...")

    stats = {'Dm':train['Dm'], 'std_u':train['std_u'], 'std_v':train['std_v'], 'std_D':train['std_D'], 'DP':train['DP']}

    # Read testing
    flow = read_h5(path_test)
    logger.debug("Read testing set...")

    # Create NTR (and TR) testing set, depending on available resolution
    stds = {'u':train['std_u'], 'v':train['std_v'], 'D':train['std_D']}
    if flag_test_res == 'TR':
        test_TR, test_NTR = create_test_set(flow, train['Dm'], stds, flag_acceleration, flag_test_res, Dt_test, ts_test)
    else:
        test_NTR = create_test_set(flow, train['Dm'], stds, flag_acceleration, flag_test_res, Dt_test, ts_test)
    logger.debug("Prepared testing set...")
    del flow

    #%% II. PROPER ORTHOGONAL DECOMPOSITION
    logger.info("II. PROPER ORTHOGONAL DECOMPOSITION")
    # Perform or load complete POD from training set
    if 'POD' in flag_loaders:
        POD = load_results('POD', flag_flow, Re, truncation_value, flag_pressure, flag_truncation)
        logger.debug("Loaded POD complete basis...")
    elif 'PODr' not in flag_loaders:
        POD = get_POD(train['Ddt'], train['dDdt'])
        logger.debug("Performed POD on training dataset...")

    # Truncate or load truncated POD
    if 'PODr' in flag_loaders:
        PODr = load_results('PODr', flag_flow, Re, truncation_value, flag_pressure, flag_truncation)
        logger.debug("Loaded POD truncated basis...")
    else:
        PODr = truncate_POD(POD, flag_truncation, truncation_value)
        logger.debug("Truncated POD basis...")

    # Project POD basis onto testing set and truncate TR flow
    test_NTR = project_POD(test_NTR, PODr)
    if flag_test_res == 'TR':
        test_TR = project_POD(test_TR, PODr)
        test_TR['Ddtr'] = np.dot(PODr['Phi'], test_TR['a'].T)
    logger.debug("Retrieve POD testing temporal coefficients from training POD projection...")

    #%% III. GALERKIN PROJECTION COEFFICIENTS
    logger.info("III. GALERKIN PROJECTION COEFFICIENTS")
    # Load or retrieve Galerkin coefficients
    if 'GPcoef' in flag_loaders:
        GPcoef = load_results('GPcoef', flag_flow, Re, truncation_value, flag_pressure, flag_truncation)
        logger.debug("Loaded Galerkin Projection coefficients...")
    else:
        logger.debug("STARTED retrieval of Galerkin Projection coefficients...")
        GPcoef = galerkin_coefs(grid, PODr['Phi'], train['Dm'], PODr['Sigma'], PODr['Psi'],
                                Re, flag_pressure, flag_integration, train['DP'])
        logger.debug("FINISHED retrieval of Galerkin Projection coefficients...")

    # Sparsify matrix of coefficients
    if flag_sparsify and flag_integration == 'matrix':
        GPcoef['Chi'] = sparsify_coeffs(PODr['Sigma'], PODr['Psi'], PODr['dPsi'], GPcoef['Chi'], sparsify_tol)
        logger.debug("Thresholded matrix of coefficients...")

    #%% IV. INTEGRATION
    logger.info("IV. INTEGRATION")
    # Galerkin projection dynamics backward-forward weighted integration
    logger.debug("STARTED integration of Galerkin Projection dynamical system...")
    test_GP = integrator(test_NTR['a'], test_NTR['t'], test_NTR['Dt'], GPcoef, PODr['Phi'],
                         flag_integration, N_process_int)
    logger.debug("FINISHED integration of Galerkin Projection dynamical system...")

    # Cubic Spline interpolation
    test_interp = interpolator(test_NTR['a'], test_NTR['t'], test_NTR['Dt'], PODr['Phi'])
    logger.debug("Finished cubic spline interpolation process...")

    #%% V. SAVE RESULTS
    logger.info("V. SAVE RESULTS")
    for save_str in flag_savers:
        save_results(eval(save_str), save_str, flag_flow, Re, truncation_value, flag_pressure, flag_truncation)
        logger.debug("Saved " + save_str + "...")

    #%% VI. ERROR COMPUTATION

    #%% VII. PLOTS



if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    with open("INPUTS.json") as json_file:
        INPUTS = json.load(json_file)

    logger.debug("Read inputs...")
    for i,j in zip(INPUTS.keys(), INPUTS.items()):
        logger.info(i + ":" + str(j[1]))

    check_inputs(INPUTS)
    main(INPUTS)
