'''
THIS MODULE CONTAINS FUNCTIONS TO SIMULATE THE NETWORK OF REACTIONS
'''

from typing import Callable
import numpy as np; np.set_printoptions(suppress=True)
from multiprocessing.pool import Pool
from tqdm import tqdm

import scipy.integrate as integrate

import libsdesim

# wrapper for the ode model
def simulate_ode(
        ODE_model:      Callable, 
        time_points:    np.ndarray,
        start_state:    list,
        ) ->            np.ndarray:
    
    sol = integrate.solve_ivp(
        ODE_model, 
        [0, time_points[-1]], 
        start_state, 
        t_eval=time_points, 
        method = 'LSODA'  # eq system is mostly stiff, needs a solver that can handle such cases
        ) 
    
    return sol.y


# wrapper for the c++ gillespie simulator module
def gillespie_wrapper(
        vartup:         tuple[np.ndarray, list, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]
        ) ->            np.ndarray:

    # unpack tuple of variables
    time_points, start_state, reactions, react_rates, state_index, birth_update_par, n_birth_updates = vartup
    
    # create arrays which will be modified in place
    sys_state           = np.array(start_state, dtype=np.int32)
    sys_state_sample    = np.zeros((time_points.size, sys_state.size), dtype = np.int32, order = 'F')

    # run c++ module, which modifies 'sys_state_sample' in place
    libsdesim.sim_gillespie(
        time_points, 
        sys_state, 
        sys_state_sample, 
        reactions, 
        react_rates, 
        state_index, 
        birth_update_par, 
        n_birth_updates
        )


    # transpose and return 
    return sys_state_sample.transpose(1,0)

# wrapper to simulate using gillespie
def simulate_gillespie(
        gill_param:     dict,
        time_points:    np.ndarray,
        start_state:    list,
        replicates:     int = 100,
        ) ->            np.ndarray:

    # create array for output
    replicate_results   = np.zeros((replicates, len(start_state), time_points.size), dtype = np.int32)

    # create arrays holding simulation parameters
    time_points         = np.array(time_points, dtype = np.float64)
    react_rates         = np.array(gill_param['reactions']['reaction_rates'], dtype=np.float64)
    state_index         = np.array(gill_param['reactions']['state_index'], dtype=np.int32)
    reactions           = np.array(gill_param['reactions']['reactions'], dtype=np.int32, order = 'F')
    birth_update_par    = np.array(gill_param['update_rate_birth']['rate_update_birth_par'][0], dtype = np.float64)
    n_birth_updates     = int(len(gill_param['update_rate_birth']['rate_update_birth_par'])*2)

    print('simulating...')
    pbar = tqdm(total=replicates)

    with Pool() as pool:
        # prepare arguments as list of tuples
        param = [(time_points, start_state, reactions, react_rates, state_index, birth_update_par, n_birth_updates) for _ in range(replicates)]
        
        # make list that unordered results will be deposited to
        pool_results = []

        # execute tasks
        for result in pool.imap_unordered(gillespie_wrapper, param):
            pool_results.append(result)
            pbar.update(1)

        # write to output array
        for i in range(replicates): replicate_results[i,:,:] = pool_results[i]


    return replicate_results

# wrapper for the c++ gillespie simulator module
def tauleaping_wrapper(
        vartup:         tuple[np.ndarray, float, list, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]
        ) ->            np.ndarray:

    # unpack tuple of variables
    time_points, timestep, start_state, reactions, react_rates, state_index, birth_update_par, n_birth_updates = vartup
    
    # create arrays which will be modified in place
    sys_state           = np.array(start_state, dtype=np.int32)
    sys_state_sample    = np.zeros((time_points.size, sys_state.size), dtype = np.int32, order = 'F')

    # run c++ module, which modifies 'sys_state_sample' in place
    libsdesim.sim_tauleaping(
        time_points,
        timestep, 
        sys_state, 
        sys_state_sample, 
        reactions, 
        react_rates, 
        state_index, 
        birth_update_par, 
        n_birth_updates
        )


    # transpose and return 
    return sys_state_sample.transpose(1,0)

# wrapper to simulate using gillespie
def simulate_tauleaping(
        gill_param:     dict,
        time_points:    np.ndarray,
        start_state:    list,
        replicates:     int = 100,
        timestep:       float = 0.01,
        ) ->            np.ndarray:

    # create array for output
    replicate_results   = np.zeros((replicates, len(start_state), time_points.size), dtype = np.int32)

    # create arrays holding simulation parameters
    time_points         = np.array(time_points, dtype = np.float64)
    react_rates         = np.array(gill_param['reactions']['reaction_rates'], dtype=np.float64)
    state_index         = np.array(gill_param['reactions']['state_index'], dtype=np.int32)
    reactions           = np.array(gill_param['reactions']['reactions'], dtype=np.int32, order = 'F')
    birth_update_par    = np.array(gill_param['update_rate_birth']['rate_update_birth_par'][0], dtype = np.float64)
    n_birth_updates     = int(len(gill_param['update_rate_birth']['rate_update_birth_par'])*2)

    print('simulating...')
    pbar = tqdm(total=replicates)

    with Pool() as pool:
        # prepare arguments as list of tuples
        param = [(time_points, timestep, start_state, reactions, react_rates, state_index, birth_update_par, n_birth_updates) for _ in range(replicates)]
        
        # make list that unordered results will be deposited to
        pool_results = []

        # execute tasks
        for result in pool.imap_unordered(tauleaping_wrapper, param):
            pool_results.append(result)
            pbar.update(1)

        # write to output array
        for i in range(replicates): replicate_results[i,:,:] = pool_results[i]


    return replicate_results