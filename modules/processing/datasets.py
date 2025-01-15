# PACKAGES
import numpy as np
import random

# LOCAL FUNCTIONS
import modules.dynamics.pressure as pressure
import modules.dynamics.differentiation as diff

def create_train_set(grid, flow, flag_acceleration, flag_resolution, flag_separation, flag_presure, ts=0):
    """
    Prepares dictionary with snapshot matrices for training set, always NTR with irregular spacing.

    :param grid: dictionary with X,Y and B entries
    :param flow: read dictionary with u,v and t entries, in snapshot form
    :param flag_acceleration: boolean indicating to compute acceleration of flow (1) or not (0)
    :param flag_resolution: 'NTR' or 'TR', indicating if read dictionary has sufficient time resolution and needs to be
    transformed to NTR or not
    :param flag_separation: 'regular' or 'irregular', indicating if time undersampling shall be reg or irreg
    :param flag_pressure: if different from 'none', requires pressure gradient to be computed, thus requiring acceleration
    :param ts: irregular time separation value in case 'TR' flag is activated

    :return train: dictionary with Ddt, Dm, t and dDdt, Dp entries (the latter if requested) with NTR
    """

    # Create training dictionary
    train = {}

    # Tackle 2D data
    train['std_u'] = np.std(flow['u'])
    train['std_v'] = np.std(flow['v'])
    train['t'] = flow['t'].flatten()
    train['Re'] = flow['Re']

    um = np.mean(flow['u'], axis=1)
    vm = np.mean(flow['v'], axis=1)

    udt = flow['u'] - um[:, None]
    vdt = flow['v'] - vm[:, None]
    del flow

    # Prepare snapshot data
    ns, nt = np.shape(udt)

    train['Ddt'] = np.zeros((ns+ns, nt))
    train['Dm'] = np.zeros((ns+ns))

    train['Ddt'][:ns, :] = udt
    train['Ddt'][ns:, :] = vdt
    del udt, vdt

    train['Dm'][:ns] = um
    train['Dm'][ns:] = vm
    del um, vm

    train['dDdt'] = []
    train['DP'] = []

    train['std_D'] = np.std(train['Ddt'] + train['Dm'][:, None])

    # Create acceleration fields, if requested
    if flag_acceleration or flag_presure != 'none':
        if flag_resolution == 'NTR':
            raise Exception("Acceleration of non-time-resolved data is not possible")
        train['dDdt'] = diff.diff_time(train['Ddt'], train['t'])

    # Change resolution of dataset, if required
    vars = ['Ddt', 'dDdt']
    nt = np.shape(train['Ddt'])[1]
    if flag_resolution == 'TR':
        Dt = train['t'][1] - train['t'][0]

        if flag_separation == 'irregular':
            # Undersample NTR irregularly
            nt_NTR = int(np.floor((nt - 1) / np.floor(ts / Dt)) + 1)
            it = np.sort(random.sample(range(nt), nt_NTR))
        elif flag_separation == 'regular':
            # Undersample NTR regularly
            it = np.arange(0, nt + 1, np.ceil(ts / Dt)).astype(int)

        train['t'] = train['t'][it]
        for i in vars:
            if type(train[i]) == type(np.array([])):
                train[i] = train[i][:, it]

    # Create NTR pressure fields, if requested
    if flag_presure != 'none':
        if flag_resolution == 'NTR':
            raise Exception("Pressure gradient of non-time-resolved data is not possible")
        train['DP'] = pressure.get_pgrad(grid, train['Ddt'] + train['Dm'][:, None], train['dDdt'], train['Re'])

    return train

def create_test_set(flow, Dm, stds, flag_acceleration, flag_resolution, Dt=0, ts=0):
    """
    Prepares dictionary with snapshot matrices for testing set, NTR with regular spacing (and TR)

    :param flow: read dictionary with u,v and t entries, in snapshot form
    :param Dm: mean flow, spatial points x
    :param stds: dictionary containing std_u, std_v and std_D from training set
    :param flag_acceleration: boolean indicating to compute acceleration of flow (1) or not (0)
    :param flag_resolution: 'NTR' or 'TR', indicating if read dictionary has sufficient time resolution and needs to be
    transformed to NTR or not
    :param Dt: time resolution of required NTR, in case 'NTR' flag is activated
    :param ts: regular time separation value in case 'TR' flag is activated

    :return test_TR, test_NTR: dictionaries with Ddt, Dm, t and dDdt entries (the latter if requested) with TR and NTR
    """

    # Create testing dictionary
    test = {}

    D = np.concatenate((flow['u'], flow['v']), axis=0)
    test['Dm'] = Dm
    test['Ddt'] = D - test['Dm'][:, None]
    test['t'] = flow['t'].flatten()
    test['Re'] = flow['Re']
    test['dDdt'] = []

    test['std_u'] = stds['u']
    test['std_v'] = stds['v']
    test['std_D'] = stds['D']
    del D

    # Create acceleration field, if requested
    if flag_acceleration:
        if flag_resolution == 'NTR':
            raise Exception("Acceleration of non-time-resolved data is not possible")
        test['dDdt'] = diff.diff_time(test['Ddt'], test['t'])

    # Change resolution of dataset, if required
    vars = ['Ddt', 'dDdt']
    if flag_resolution == 'TR':
        test_TR = test.copy()
        Dt = test['t'][1] - test['t'][0]
        nt = len(test['t'])

        test_NTR = test.copy()
        test_NTR['Dt'] = Dt

        # Undersample NTR regularly
        it = np.arange(0, nt + 1, np.ceil(ts / Dt)).astype(int)
        test_NTR['t'] = test_NTR['t'][it]
        for i in vars:
            if type(test_NTR[i]) == type(np.array([])):
                test_NTR[i] = test_NTR[i][:, it]

        # End TR at same time instant than NTR
        tf = test_NTR['t'][-1]
        itf = np.where(test_TR['t'] == tf)[0][0]
        test_TR['t'] = test_TR['t'][:itf + 1]
        for i in vars:
            if type(test_TR[i]) == type(np.array([])):
                test_TR[i] = test_TR[i][:, :itf + 1]

        return test_TR, test_NTR

    elif flag_resolution == 'NTR':
        test_NTR = test
        test_NTR['Dt'] = Dt

        return test_NTR