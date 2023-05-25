'''
PYTHON INTERFACE FOR THE C PROGRAMS IMPLEMENTING THE GILLESPIE SIMULATOR
'''

import ctypes
import numpy as np
from copy import deepcopy

#### CUSTOM TYPES ####

# long int
i8   = ctypes.c_long
# long float
f8   = ctypes.c_double
# 1d long float array
np_arr_flt = np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
# 1d long int array
np_arr_int = np.ctypeslib.ndpointer(ctypes.c_long,   flags="C_CONTIGUOUS")
# 2d array of pointers corresponding to a 2d array of long ints
_2dp       = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C')   # 2d array 


#### TWO COMPARTMENT MODEL ####

neuron_gill_2c = ctypes.cdll.LoadLibrary("./neuron_gill_2c.so")
_neruon_gill_2c_loop = neuron_gill_2c.gillespie_2c_loop

'''
void gillespie_loop(
    const f8    MU,     // death rate
    const f8    GAMMA,  // transport rate
    const f8    DELTA,  // mutant deficiency ratio
    const f8    C_B,    // soma birth rate control constant
    const f8    C_T,    // soma to axon transport rate control constant
    const i8    NSS_S,  // carrying capacity of soma
    const i8    NSS_A,  // carrying capacity of axon

    const f8*   time_points,       // array of time points where the state of the system should be recorded.
    const i8    n_time_points,     // number of time points (length of 'time_points')

          i8*   sys_state,         // array holding the current state of the system (number of molecules in each location)
          i8**  sys_state_sample   // The main output. A sample of the system state at each time point specified in 'time_points'
    )
{
'''
# argument types set according to c function
_neruon_gill_2c_loop.argtypes = [f8, f8, f8, f8, f8, i8, i8, np_arr_flt, i8, np_arr_int, _2dp] 
_neruon_gill_2c_loop.restype  = None # no return, the function modifies the output array in-place

def neuron_gill_2c_sim(
        mu, gamma, delta, c_b, c_t, nss_s, nss_a, # model parameters
        time_points,                              # points in time where system state should be recorded
        sys_state                                 # state vector of system (will be modified in place)
        ):

    sys_state = deepcopy(sys_state)
    
    # create empty array for storing the samples fo the state vector over time
    pop_out = np.zeros((time_points.size, sys_state.size), dtype=np.int64, order = 'C')
    pop_out_2dp = (pop_out.__array_interface__['data'][0]  # turn into array of 2d pointers
                   + np.arange(pop_out.shape[0])*pop_out.strides[0]).astype(np.uintp) 
    
    # run the c module
    _neruon_gill_2c_loop(
        mu, gamma, delta, c_b, c_t, nss_s, nss_a, 
        time_points, time_points.size, 
        sys_state, pop_out_2dp)
    
    return pop_out

#### RING MODEL ####

neuron_gill_ring = ctypes.cdll.LoadLibrary("./neuron_gill_ring.so")
_neruon_gill_ring_loop = neuron_gill_ring.gillespie_ring_loop

'''
void gillespie_ring_loop(
    const f8    mu,     // death rate
    const f8    gm,     // transport rate (gamma)
    const f8    delta,  // mutant deficiency ratio
    const f8    c_b,    // soma birth rate control constant
    const i8    nss_s,  // carrying capacity of soma

    const f8*   time_points,       // array of time points where the state of the system should be recorded.
    const i8    n_time_points,     // number of time points (length of 'time_points')

          i8*   sys_state,         // array holding the current state of the system (number of molecules in each location)
          i8**  sys_state_sample   // The main output. A sample of the system state at each time point specified in 'time_points'
    )
{
'''
# argument types set according to c function
_neruon_gill_ring_loop.argtypes = [f8, f8, f8, f8, i8, np_arr_flt, i8, np_arr_int, _2dp] 
_neruon_gill_ring_loop.restype  = None # no return, the function modifies the output array in-place

def neuron_gill_ring_sim(
        mu, gamma, delta, c_b, nss_s, # model parameters
        time_points,                  # points in time where system state should be recorded
        sys_state                     # state vector of system (will be modified in place)
        ):

    sys_state = deepcopy(sys_state)
    
    # create empty array for storing the samples fo the state vector over time
    pop_out = np.zeros((time_points.size, sys_state.size), dtype=np.int64, order = 'C')
    pop_out_2dp = (pop_out.__array_interface__['data'][0]  # turn into array of 2d pointers
                   + np.arange(pop_out.shape[0])*pop_out.strides[0]).astype(np.uintp) 
    
    # run the c module
    _neruon_gill_ring_loop(
        mu, gamma, delta, c_b, nss_s, 
        time_points, time_points.size, 
        sys_state, pop_out_2dp)
    
    return pop_out