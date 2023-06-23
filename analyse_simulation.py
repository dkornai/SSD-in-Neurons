import numpy as np

def get_result_statistics(
        replicate_results,  # variable values over time (number of wildtype and mutant in each compartment)
        time_points,        # time points where system is sampled
        delta,              # mutant deficiency, used in calculating effective population sizes
        vars,               # name of the variables being tracked (compartment name + wt/mt)
        comp                # name of the compartments (e.g. soma, axon, etc.)
        ):
    
    n_vars = len(vars)
    n_comp = len(comp)
    
    # separate out wt and mt counts
    wt_counts = replicate_results[:,np.arange(0, n_vars, 2),:]
    mt_counts = replicate_results[:,np.arange(1, n_vars, 2),:]

    # get mean counts of each variable over time
    mean_var_value = np.mean(replicate_results, axis=0)

    # get the mean heteroplasmy of each compartment
    mean_per_comp_= np.mean(mt_counts/(wt_counts+mt_counts), axis = 0)