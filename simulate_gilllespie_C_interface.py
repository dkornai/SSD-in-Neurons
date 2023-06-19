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
f8_1d = np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
# 1d long int array
i8_1d = np.ctypeslib.ndpointer(ctypes.c_long,   flags="C_CONTIGUOUS")
# 2d array of pointers corresponding to a 2d array of long ints
_2dp       = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C')


#### SET UP C MODULE ####

gillespie_simulate = ctypes.cdll.LoadLibrary("./simulate_gillespie.so")
_simulate = gillespie_simulate.simulate

# argument types set according to c function
'''
void simulate(
    const f8*   time_points,       // array of time points where the state of the system should be recorded.
    const i8    n_time_points,     // number of time points (length of 'time_points')

    const i8    n_pops,
          i8*   sys_state,         // array holding the current state of the system (number of molecules in each location)
          i8**  sys_state_sample,   // The main output. A sample of the system state at each time point specified in 'time_points'

    const i8    n_reactions,
    const i8**  reactions,
          f8*   reaction_rates,
    const i8*   state_index,

    const i8    n_birthrate_updates,
    const f8**  birthrate_updates_par,
    const i8*   birthrate_updates_reaction,
    const i8*   birthrate_state_index
    )
{
'''
_simulate.restype  = None # no return, the function modifies the output array in-place
_simulate.argtypes = [
    f8_1d, 
    i8,

    i8, 
    i8_1d, 
    _2dp,
    
    i8, 
    _2dp, 
    f8_1d, 
    i8_1d, 

    i8,
    _2dp, 
    i8_1d,
    i8_1d,
    ] 


_simulateOD = gillespie_simulate.simulate
_simulateOD.restype = None
_simulateOD.argtypes = [
    f8_1d, 
    i8,

    i8, 
    i8_1d, 
    _2dp,
    
    i8, 
    _2dp, 
    f8_1d, 
    i8_1d, 

    i8,
    _2dp, 
    i8_1d,
    i8_1d,
    ] 


#### SET UP PYTHON INTERFACE ####

def gillespie_simulate(
        reactions_dict,
        time_points,     # list of points in time where system state should be recorded
        sys_state,       # state of the system (number of wt and mt in each compartment)
        onedynamic = False
        ):

    ## UNPACK INPUT PARAMETERS ##
    # TIME POINTS
    n_time_points = time_points.size


    # SYSTEM STATE AND SYSTEM STATE SAMPLE
    sys_state = np.array(deepcopy(sys_state), dtype=np.int64)

    n_pops = sys_state.size

    sys_state_sample = np.zeros((time_points.size, sys_state.size), dtype=np.int64, order = 'C')
        # turn into array of 2d pointers
    sys_state_sample_2dp = (sys_state_sample.__array_interface__['data'][0] + np.arange(sys_state_sample.shape[0])*sys_state_sample.strides[0]).astype(np.uintp)


    # GILLESPIE REACTIONS
    gillespie_reactions = reactions_dict['gillespie']

    n_reactions = gillespie_reactions['n_reactions']

    reactions = gillespie_reactions['reactions']
        # turn into array of 2d pointers
    reactions_2dp = (reactions.__array_interface__['data'][0] + np.arange(reactions.shape[0])*reactions.strides[0]).astype(np.uintp)
    
    reaction_rates = gillespie_reactions['reaction_rates']

    state_index = gillespie_reactions['state_index']

    # BIRTHRATE UPDATES
    update_rate_birth = reactions_dict['update_rate_birth']

    n_birthrate_updates = update_rate_birth['n_rate_update_birth']

    birthrate_updates_par = update_rate_birth['rate_update_birth_par']
        # turn into array of 2d pointers
    birthrate_updates_par_2dp = (birthrate_updates_par.__array_interface__['data'][0] + np.arange(birthrate_updates_par.shape[0])*birthrate_updates_par.strides[0]).astype(np.uintp)
    
    birthrate_updates_reaction = update_rate_birth['rate_update_birth_reaction']

    birthrate_state_index = update_rate_birth['birthrate_state_index']

    # run the c module
    if onedynamic == False:
        _simulate(
            time_points,
            n_time_points,

            n_pops,
            sys_state,
            sys_state_sample_2dp,

            n_reactions,
            reactions_2dp,
            reaction_rates,
            state_index,

            n_birthrate_updates,
            birthrate_updates_par_2dp,
            birthrate_updates_reaction,
            birthrate_state_index,
            )
    elif onedynamic == True:
        _simulateOD(
            time_points,
            n_time_points,

            n_pops,
            sys_state,
            sys_state_sample_2dp,

            n_reactions,
            reactions_2dp,
            reaction_rates,
            state_index,

            n_birthrate_updates,
            birthrate_updates_par_2dp,
            birthrate_updates_reaction,
            birthrate_state_index,
            )
    
    return sys_state_sample