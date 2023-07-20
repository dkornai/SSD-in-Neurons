import numpy as np; np.seterr(divide='ignore')
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pymannkendall as mk
from statsmodels.tsa.stattools import kpss

import warnings
warnings.filterwarnings("ignore")

# merge a dict and add a prefix to the variable names
def merge_with_prefix(x, y, var):
    y = {f"{var}{k}": v for k, v in y.items()}
    x.update(y)
    return x

# transform a statistic calculated for each replicate at each time point into two arrays
def stat_to_x_y(time_points, stat):
    n_replicates, n_timepoints = stat.shape
    x = []; y = []

    for i in range(n_timepoints):
        for j in range(n_replicates):
            if not np.isnan(stat[j, i]):
                x.append(time_points[i])
                y.append(stat[j, i])

    return np.array(x), np.array(y)


def fixed_intercept_regression(x, y, intercept_value):
    # Subtract the intercept from y
    y = y - intercept_value
    
    # Create a dataframe from x
    df = pd.DataFrame({'x': x, 'y':y})

    # Define your model, specifying that we want to fit without adding an automatic intercept
    model = smf.ols(formula='y ~ x - 1', data=df)

    # Fit your model
    results = model.fit()
    x = round(results.params[0], 16)
    p = round(results.pvalues[0], 4)
    
    return x, p

# time series tests for the mean of a parameter value
def ts_tests(x, data_mean):
    
    # reg_x, reg_p = fixed_intercept_regression(x, data_mean, data_mean[0])
    # if reg_p >= 0.05:
    #     slope = 'none'
    # elif reg_x > 0:
    #     slope = 'positive'
    # else:
    #     slope = 'negative'
    
    mktest = mk.original_test(data_mean)
    mktest_trend = mktest[0]
    mktest_p = round(mktest[2], 4)

    kpss_p = round(kpss(data_mean)[1], 4)
    kpss_stationary = 'non-stationary' if kpss_p < 0.05 else 'stationary'
    
    #print('regresssion slope:', slope, "p=",reg_p)
    print('mk trend:',mktest_trend, 'p=', mktest_p)
    print('kpss:',kpss_stationary, 'p=', kpss_p)

    return {'mk_trend':mktest_trend, 'mk_p':mktest_p, 'kpss_trend':kpss_stationary, 'kpss_p':kpss_p}

# wilcox test and t-test between complete starting and final distributions of a parameter value
def change_tests(full_values):
    start_val, final_val = full_values[:,0], full_values[:,-1]

    # wilcox test on original distributions
    s2_wilcox_test = stats.wilcoxon(final_val, start_val, nan_policy='omit')
    s2_wilcox_test_p = round(s2_wilcox_test.pvalue, 4)
    
    if s2_wilcox_test_p >= 0.05:
        s2_wilcox_test_dir = 'none'
    elif np.nanmean(final_val) > np.nanmean(start_val):
        s2_wilcox_test_dir = 'greater'
    else:
        s2_wilcox_test_dir = 'less'

    # # wilcox test on original distributions
    # s1_wilcox_test = stats.wilcoxon(final_val - start_val[0], nan_policy='omit')
    # s1_wilcox_test_p = round(s1_wilcox_test.pvalue, 16)
    
    # if s1_wilcox_test_p >= 0.05:
    #     s1_wilcox_test_dir = 'none'
    # elif np.nanmean(final_val) > start_val[0]:
    #     s1_wilcox_test_dir = 'greater'
    # else:
    #     s1_wilcox_test_dir = 'less'    

    # pooling values and averageing to get normal distributions, before doing a t-test
    final_val_means = np.nanmean(final_val.reshape(20,-1), axis = 0)
    start_val_means = np.nanmean(start_val.reshape(20,-1), axis = 0)
    
    # two sample t-test
    s2_t_test = stats.ttest_ind(final_val_means, start_val_means, equal_var=False, nan_policy='omit')
    s2_t_test_p = np.round(s2_t_test.pvalue, 4)
    
    if s2_t_test_p >= 0.05:
        s2_t_test_dir = 'none'
    elif np.nanmean(final_val_means) > np.nanmean(start_val_means):
        s2_t_test_dir = 'greater'
    else:
        s2_t_test_dir = 'less'

    # # one sample t-test
    # s1_t_test = stats.ttest_1samp(final_val_means, start_val_means[0], nan_policy='omit')
    # s1_t_test_p = np.round(s1_t_test.pvalue, 16)

    # if s1_t_test_p >= 0.05:
    #     s1_t_test_dir = 'none'
    # elif np.nanmean(final_val_means) > start_val_means[0]:
    #     s1_t_test_dir = 'greater'
    # else:
    #     s1_t_test_dir = 'less'


    print('wilcox. test:', s2_wilcox_test_dir, 'p=', s2_wilcox_test_p)
    #print('wilcox. 1 s. test:', s1_wilcox_test_dir, 'p=', s1_wilcox_test_p)
    print('pooled t test:', s2_t_test_dir, 'p=', s2_t_test_p)
    #print('pooled 1 s. t test:', s1_t_test_dir, 'p=', s1_t_test_p)

    return {'w_test_dir': s2_wilcox_test_dir,'w_test_p':s2_wilcox_test_p, 't_test_dir':s2_t_test_dir, 't_test_p':s2_t_test_p}


def two_component_statistics(
        replicate_results,  # variable values over time (number of wildtype and mutant in each compartment)
        time_points,        # time points where the measurements were taken
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

    # # get effective population size for soma and axon
    # eps_soma = s_wt + delta*s_mt
    # eps_soma_mean = np.nanmean(eps_soma, axis = 0)
    # eps_soma_sem = np.nanstd(eps_soma, axis = 0)/np.sqrt(n_samp)

    # eps_axon = a_wt + delta*a_mt
    # eps_axon_mean = np.nanmean(eps_axon, axis = 0)
    # eps_axon_sem = np.nanstd(eps_axon, axis = 0)/np.sqrt(n_samp)


    # get total population size
    ps = wt_totals + mt_totals
    ps_mean = np.nanmean(ps, axis=0)
    ps_sem = np.nanstd(ps,axis=0)/np.sqrt(n_samp)
    
    # # get total population size in soma and axon
    # ps_soma = s_wt + s_mt
    # ps_soma_mean = np.nanmean(ps_soma, axis = 0)
    # ps_soma_sem = np.nanstd(ps_soma, axis = 0)/np.sqrt(n_samp)

    # ps_axon = a_wt + a_mt
    # ps_axon_mean = np.nanmean(ps_axon, axis = 0)
    # ps_axon_sem = np.nanstd(ps_axon, axis = 0)/np.sqrt(n_samp)


    # proportion of exhausted cells
    prop_exhaust = np.mean(ps == 0, axis=0)

    # get total heteroplasmy
    het = mt_totals/(mt_totals+wt_totals)
    het_mean = np.nanmean(het, axis=0)
        # heteroplasmy is not defined for exhausted cells, so sample size must be adjusted down accordingly
    het_sem = np.nanstd(het, axis=0)/np.sqrt((1-prop_exhaust)*n_samp) 

    # proportion of 0 or 1 heteroplasmy
    prop_het1 = np.mean(het == 1, axis=0)
    prop_het0 = np.mean(het == 0, axis=0)

    res = {
        't':            np.round(time_points, 1),
        'ps_mean':      np.round(ps_mean,2),
        'ps_sem':       np.round(ps_sem,2),
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
    
    time_df = pd.DataFrame.from_dict(res)

    stat_res = {}
    print('\nCopy number:')
    merge_with_prefix(stat_res, change_tests(ps), 'cn_')
    merge_with_prefix(stat_res, ts_tests(time_points, ps_mean), 'cn_')

    print('\nEff. pop. size:')
    merge_with_prefix(stat_res, change_tests(eps), 'eps_')
    merge_with_prefix(stat_res, ts_tests(time_points, eps_mean), 'eps_')

    print('\nHeteroplasmy:')
    merge_with_prefix(stat_res, change_tests(het), 'het_')
    merge_with_prefix(stat_res, ts_tests(time_points, het_mean), 'het_')
    print()

    return time_df, stat_res