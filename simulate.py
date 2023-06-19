from typing import Callable
import numpy as np; np.set_printoptions(suppress=True)
import pandas as pd
import scipy.integrate as integrate

from simulate_gilllespie_C_interface import gillespie_simulate

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
        onedynamic:     bool = False,
        ) ->            np.ndarray:
    
    # create array for output
    replicate_results = np.zeros((replicates, len(start_state), time_points.size), dtype = np.int64)

    print('simulating...')
    for i in range(replicates):
        replicate_results[i, :, :] = np.swapaxes(
            gillespie_simulate(
                gill_param, 
                time_points, 
                start_state,
                onedynamic)
        , 0, 1)
        
        print(f"{round(((i+1)/replicates)*100, 2)}% completed  ", end = "\r")

    return replicate_results
