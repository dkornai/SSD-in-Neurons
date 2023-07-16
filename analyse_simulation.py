import numpy as np; np.seterr(divide='ignore')
import pandas as pd
import scipy.stats as stats

import warnings
warnings.filterwarnings("ignore")

def two_component_statistics(
        replicate_results,  # variable values over time (number of wildtype and mutant in each compartment)
        delta,              # mutant deficiency, used in calculating effective population sizes
        ):
    
    n_samp = replicate_results.shape[0]
    
    # wt in soma
    s_wt = replicate_results[:,0,:]
    s_wt_mean = np.nanmean(s_wt, axis=0)
    s_wt_sem = np.nanstd(s_wt, axis=0)/np.sqrt(n_samp)

    # mt in soma
    s_mt = replicate_results[:,1,:]
    s_mt_mean = np.nanmean(s_mt, axis=0)
    s_mt_sem = np.nanstd(s_mt, axis=0)/np.sqrt(n_samp)

    # wt in axon
    a_wt = replicate_results[:,2,:]
    a_wt_mean = np.nanmean(a_wt, axis=0)
    a_wt_sem = np.nanstd(a_wt, axis=0)/np.sqrt(n_samp)

    # mt in axon
    a_mt = replicate_results[:,3,:]
    a_mt_mean = np.nanmean(a_mt, axis=0)
    a_mt_sem = np.nanstd(a_mt, axis=0)/np.sqrt(n_samp)

    
    # get the number of wt and mt in the full system (summed over compartments)
    wt_totals = s_wt + a_wt
    mt_totals = s_mt + a_mt


    # get total effective population size
    eps = wt_totals + (mt_totals*delta)
    eps_mean = np.nanmean(eps, axis = 0)
    eps_sem = np.nanstd(eps,axis=0)/np.sqrt(n_samp)

    # get total population size
    ps = wt_totals + mt_totals
    ps_mean = np.nanmean(ps, axis=0)
    ps_sem = np.nanstd(ps,axis=0)/np.sqrt(n_samp)
    
    # proportion of exhausted cells
    prop_exhaust = np.mean(ps == 0, axis=0)

    # get total heteroplasmy
    het = mt_totals/(mt_totals+wt_totals)
    het_mean = np.nanmean(het, axis=0)
        # heteroplasmy is not defined for exhausted cells, so sample size must be adjusted down accordingly
    het_sem = np.nanstd(het, axis=0)/np.sqrt((1-prop_exhaust)*n_samp) 
    
    prop_het1 = np.mean(het == 1, axis=0)
    prop_het0 = np.mean(het == 0, axis=0)

    res = {
        'pop_mean':     np.round(ps_mean,2),
        'pop_sem':      np.round(ps_sem,2),
        'eps_mean':     np.round(eps_mean,2),
        'eps_sem':      np.round(eps_sem,2),
        'het_mean':     np.round(het_mean,4),
        'het_sem':      np.round(het_sem,4),
        'p_het_1':      np.round(prop_het1,4),
        'p_het_0':      np.round(prop_het0,4),
        'p_pop0':       np.round(prop_exhaust,4),
        's_wt_mean':    np.round(s_wt_mean,2),
        's_wt_sem':     np.round(s_wt_sem,2),
        's_mt_mean':    np.round(s_mt_mean,2),
        's_mt_sem':     np.round(s_mt_sem,2),
        'a_wt_mean':    np.round(a_wt_mean,2),
        'a_wt_sem':     np.round(a_wt_sem,2),
        'a_mt_mean':    np.round(a_mt_mean,2),
        'a_mt_sem':     np.round(a_mt_sem,2),
        }
    
    df = pd.DataFrame.from_dict(res)

    # u test on the original distributions
    mw_test_p = round(stats.mannwhitneyu(het[:,-1], het[:,0], nan_policy='omit', alternative='greater',).pvalue, 4)

    # wilcox test
    wilcox_test_p = round(stats.wilcoxon(het[:,-1], het[:,0], nan_policy='omit', alternative='greater',).pvalue, 4)

    # pooling values and averageing to get normal distributions, before doing a t-test
    final_het_means = np.nanmean(het[:,-1].reshape(20,-1), axis = 0)
    start_het_means = np.nanmean(het[:, 0].reshape(20,-1), axis = 0)
    t_test_p = round(stats.ttest_ind(final_het_means,start_het_means, equal_var=False, alternative='greater', nan_policy='omit').pvalue,4)

    print("\n> Statistical tests: end heteroplasmy > start heteroplasmy:")
    print(" mann-whitney p-value:", mw_test_p)
    print("  wilcox test p-value:", wilcox_test_p)
    print("pooled t-test p-value:", t_test_p)
    print()

    return df