from typing import Callable
import numpy as np; np.set_printoptions(suppress=True)
import pandas as pd
import scipy.integrate as integrate

import libgillespie

# wrapper for the ode model
def ODE_simulate(
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

# wrapper for the gillespie 
def GILL_simulate(
        gill_param:     dict,
        time_points:    np.ndarray,
        start_state:    list,
        replicates:     int = 100,
        ) ->            np.ndarray:

    # create array for output
    replicate_results   = np.zeros((replicates, len(start_state), time_points.size), dtype = np.int64)

    # create arrays holding simulation parameters
    time_points         = np.array(time_points, dtype = np.float64)
    react_rates         = np.array(gill_param['gillespie']['reaction_rates'], dtype=np.float64)
    state_index         = np.array(gill_param['gillespie']['state_index'], dtype=np.int64)
    reactions           = np.array(gill_param['gillespie']['reactions'], dtype=np.int64, order = 'F')
    birth_update_par    = np.array(gill_param['update_rate_birth']['rate_update_birth_par'][0], dtype = np.float64)

    print('simulating...')

    for i in range(replicates):

        # create arrays which will be modified in place
        sys_state   = np.array(start_state, dtype=np.int64)
        sys_state_sample = np.zeros((time_points.size, sys_state.size), dtype = np.int64, order = 'F')

        # run c++ module
        libgillespie.simulate(time_points, sys_state, sys_state_sample, reactions, react_rates, state_index, birth_update_par)
        
        # transpose and write results
        replicate_results[i, :, :] = sys_state_sample.transpose(1,0)
        
        print(f"{round(((i+1)/replicates)*100, 2)}% completed  ", end = "\r")

    return replicate_results
