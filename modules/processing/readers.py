# PACKAGES
import h5py
import os

def read_h5(path):
    """
    Reads .h5 file and creates dictionary

    :param path: string path of file

    :return dict: dictionary containing all entries
    """
    if not os.path.exists(path):
        raise Exception("Path does not exist")

    dict= {}
    with h5py.File(path, 'r') as f:
        for i in f.keys():
            dict[i] = f[i][()]

    return dict

def save_h5(path, dict):
    """
    Saves dictionary into .h5 file

    :param path: string path of file
    :param dict: dictionary containing all entries
    """

    with h5py.File(path, 'w') as h5file:
        for key, item in dict.items():
            h5file.create_dataset(key, data=item)

def save_results(result, flag_save, flag_flow, Re, value_truncation, flag_pressure, flag_truncation, ts_test):
    """
    Saves dictionary of results in folder path

    :param result: dictionary of results to save
    :param flag_save: flag indicating what to save ('POD', 'PODr', 'GPcoeff', 'test_GP' or 'test_interp')
    :param flag_flow: flag indicating flow type ('FP' or 'Jet')
    :param Re: Reynolds numbers
    :param value_truncation: input value for POD truncation
    :param flag_pressure: flag indicating pressure coefficients retrieval ('none', 'analytical' or 'empirical')
    :param flag_truncation: flag indicating type of POD truncation ('elbow', 'manual', 'optimal', or 'energy')
    :param ts_test: normalized separation time for testing set
    """

    dir = r'.\results'
    if not os.path.exists(dir):
        os.makedirs(dir)

    subdir = flag_flow + '_Re' + str(Re)
    subdir_path = os.path.join(dir, subdir)
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path)

    subsubdir = flag_truncation + '_' + str(value_truncation) + '_' + 'pressure' + '_' + flag_pressure
    subsubdir_path = os.path.join(subdir_path, subsubdir)
    if not os.path.exists(subsubdir_path):
        os.makedirs(subsubdir_path)

    # If non-truncated POD wants to be saved, write in same directory of flow type
    if flag_save == 'POD' or flag_save == 'stats':
        path = os.path.join(subdir_path, flag_save)
    if flag_save == 'test_GP' or flag_save == 'test_interp':
        path = os.path.join(subsubdir_path, flag_save+'_ts_'+str(ts_test))
    else:
        path = os.path.join(subsubdir_path, flag_save)
    save_h5(path, result)


def load_results(flag_load, flag_flow, Re, value_truncation, flag_pressure, flag_truncation):
    """
    Loads dictionary of results

    :param flag_load: flag indicating what to save ('POD', 'PODr', 'GPcoeff', 'test_GP' or 'test_interp')
    :param flag_flow: flag indicating flow type ('FP' or 'Jet')
    :param Re: Reynolds numbers
    :param value_truncation: input value for POD truncation
    :param flag_pressure: flag indicating pressure coefficients retrieval ('none', 'analytical' or 'empirical')
    :param flag_truncation: flag indicating type of POD truncation ('elbow', 'manual', 'optimal', or 'energy')

    :return result: dictionary of results
    """

    dir = r'.\results'

    subdir = flag_flow + '_Re' + str(Re)
    subdir_path = os.path.join(dir, subdir)

    subsubdir = flag_truncation + '_' + str(value_truncation) + '_' + 'pressure' + '_' + flag_pressure
    subsubdir_path = os.path.join(subdir_path, subsubdir)

    # If non-truncated POD wants to be saved, write in same directory of flow type
    if flag_load == 'POD' or flag_load == 'stats':
        path = os.path.join(subdir_path, flag_load)
    else:
        path = os.path.join(subsubdir_path, flag_load)

    result = read_h5(path)
    return result

def check_inputs(INPUTS):
    """
    Checks certain inputs and raises errors if choices are not within expected ones

    :param INPUTS: dictionary containing paths, params and flags for GP code
    """

    flag_pressure = INPUTS["flag_pressure"]         # GP pressure coefficients retrieval flag ('none', 'analytical', 'empirical')
    flag_truncation = INPUTS["flag_truncation"]     # POD truncation method ('manual','energy','optimal','elbow','none')
    flag_train_res = INPUTS["flag_train_res"]       # Resolution of training read dataset ('TR', 'NTR')
    flag_test_res = INPUTS["flag_test_res"]         # Resolution of testing read dataset ('TR', 'NTR')
    flag_integration = INPUTS["flag_integration"]   # Type of integration system used ('matrix','einsum')

    if flag_pressure != 'none' and flag_pressure != 'empirical' and flag_pressure != 'analytical':
        raise Exception("flag_pressure needs to be 'none', 'analytical' or 'empirical'")

    if flag_truncation != 'manual' and flag_truncation != 'energy' and flag_truncation != 'optimal' and flag_truncation != 'elbow' and flag_truncation != 'none':
        raise Exception("flag_truncation needs to be 'manual','energy','optimal','elbow' or 'none'")

    if flag_train_res != 'TR' and flag_train_res != 'NTR':
        raise Exception("flag_train_res needs to be 'TR' or 'NTR'")

    if flag_test_res != 'TR' and flag_test_res != 'NTR':
        raise Exception("flag_test_res needs to be 'TR' or 'NTR'")

    if flag_integration != 'matrix' and flag_integration != 'einsum':
        raise Exception("flag_integration needs to be 'matrix' or 'einsum'")