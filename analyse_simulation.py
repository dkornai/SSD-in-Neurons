import numpy as np




def get_result_statistics(
        replicate_results,  # variable values over time (number of wildtype and mutant in each compartment)
        time_points,        # time points where system is sampled
        delta,              # mutant deficiency, used in calculating effective population sizes
        vari,               # name of the variables being tracked (compartment name + wt/mt)
        comp                # name of the compartments (e.g. soma, axon, etc.)
        ):
    
    n_vars = len(vari)
    n_comp = len(comp)
    n_samp = replicate_results.shape[0]
    
    # separate out wt and mt counts per compartment
    wt_counts = replicate_results[:,np.arange(0, n_vars, 2),:]
    mt_counts = replicate_results[:,np.arange(1, n_vars, 2),:]
    
    # get the number of wt and mt in the full system (summed over compartments)
    wt_totals = np.sum(wt_counts, axis = 1)
    mt_totals = np.sum(mt_counts, axis = 1)

    # get mean counts of each variable over time
    mean_var_value = np.mean(replicate_results, axis=0)

    # get the mean heteroplasmy of each compartment
    per_comp_het = mt_counts/(wt_counts+mt_counts)
    mean_per_comp_het= np.mean(per_comp_het, axis = 0)
    
    # get the mean effective population size of each compartment
    mean_per_comp_eps= np.mean(mt_counts*delta + wt_counts, axis = 0)
    
    # get the total heteroplasmies of all the replicates
    total_het = mt_totals/(wt_totals+mt_totals)
    
    # get the total heteroplasmies in the last time point
    final_total_het = total_het[:,-1]
    
    return final_total_het