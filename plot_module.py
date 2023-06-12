import numpy as np
import scipy.integrate as integrate
from scipy.signal import resample
import matplotlib.pyplot as plt
import pandas as pd

'''
Plot the results of a numerical solution to an ODE.
'''
def plot_ODE(
        results,        # variable values over time (number of wildtype and mutant in each compartment)
        time_points,    # time points where system is sampled
        delta,          # mutant deficiency, used in calculating effective population sizes
        vars,           # name of the variables being tracked (compartment name + wt/mt)
        comp            # name of the compartments (e.g. soma, axon, etc.)
        ):
    
    n_vars = len(vars)
    n_comp = len(comp)
    
    # plot wildtype and mutant counts in each compartment over time
    plt.subplots(figsize=(10, 5))
    for i in range(n_vars):
        plt.plot(time_points, results[i], label = vars[i])
    plt.legend()
    plt.title('wt and mt counts in each compartment over time')

    # plot heteroplasmy levels in each compartment over time
    plt.subplots(figsize=(10, 5))
    for i in range(n_comp):
        het = results[(i*2)+1]/(results[(i*2)+1]+results[i*2])
        plt.plot(time_points, het, label = f'{comp[i]} het')
    plt.ylim([0, 1])
    plt.legend()
    plt.title('heteroplasmy in each compartment over time')

    # plot effective population sizes over time
    plt.subplots(figsize=(10, 5))
    for i in range(n_comp):
        eps = results[(i*2)+1]*delta + results[i*2]
        plt.plot(time_points, eps, label = f'{comp[i]} eff. pop. size')
    plt.legend()
    plt.title('effective population size in each compartment over time')

    # print parameter values in the final time point
    print("Final counts of mt and wt in each compartment:")
    for i in range(n_vars):
        print(f'{vars[i]}\t{round(results[i,-1], 4)}\t')

    print("\nFinal effective population sizes in each compartment:")
    for i in range(n_comp):
        eps = results[(i*2)+1,-1]*delta + results[i*2,-1]
        print(f'{comp[i]}\t{round(eps, 4)}\t')


'''
Plot the mean values across many replicate gillespie simulations of the system
'''
def plot_gillespie(
        replicate_results,  # variable values over time (number of wildtype and mutant in each compartment)
        time_points,    # time points where system is sampled
        delta,          # mutant deficiency, used in calculating effective population sizes
        vars,           # name of the variables being tracked (compartment name + wt/mt)
        comp            # name of the compartments (e.g. soma, axon, etc.)
        ):
    
    n_vars = len(vars)
    n_comp = len(comp)
    
    # plot wildtype and mutant counts in each compartment over time
    mean_per_var_counts = []
    plt.subplots(figsize=(10, 5))
    for i in range(n_vars):
        counts = np.nanmean(replicate_results[:,i,:], axis = 0)
        mean_per_var_counts.append(counts)

        plt.plot(time_points, counts, label = vars[i])
    plt.legend()
    plt.title('mean wt and mt counts in each compartment over time')

    # plot heteroplasmy levels in each compartment over time
    mean_per_comp_het = []
    plt.subplots(figsize=(10, 5))
    for i in range(n_comp):
        # calculate mean heteroplasmy as a mean of ratios
        het = np.nanmean(replicate_results[:,(i*2)+1,:]/(replicate_results[:,(i*2)+1,:]+replicate_results[:,i*2,:]), axis = 0)
        mean_per_comp_het.append(het)

        plt.plot(time_points, het, label = f'{comp[i]} het')
    plt.ylim([0, 1])
    plt.legend()
    plt.title('mean heteroplasmy in each compartment over time')

    # plot effective population sizes over time
    mean_per_comp_eps = []
    plt.subplots(figsize=(10, 5))
    for i in range(n_comp):
        eps = np.nanmean(replicate_results[:,(i*2)+1,:]*delta + replicate_results[:,i*2,:], axis = 0)
        mean_per_comp_eps.append(eps)

        plt.plot(time_points, eps, label = f'{comp[i]} eff. pop. size')
    plt.legend()
    plt.title('mean effective population size in each compartment over time')

    # print parameter values in the final time point
    print("Final mean counts of mt and wt in each compartment:")
    for i in range(n_vars):
        print(f'{vars[i]}\t{round(mean_per_var_counts[i][-1], 4)}\t')

    print("\nFinal mean heteroplasmy in each compartment:")
    for i in range(n_comp):
        print(f'{comp[i]}\t{round(mean_per_comp_het[i][-1], 4)}\t')

    print("\nFinal mean effective population sizes in each compartment:")
    for i in range(n_comp):
        print(f'{comp[i]}\t{round(mean_per_comp_eps[i][-1], 4)}\t')