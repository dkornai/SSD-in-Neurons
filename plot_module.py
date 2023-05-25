import numpy as np
import scipy.integrate as integrate
from scipy.signal import resample
import matplotlib.pyplot as plt
import pandas as pd

def plot_ODE(results, time_points, delta, vars):
    plt.subplots(figsize=(10, 5))
    plt.plot(time_points, results[0], label = 'soma w')
    plt.plot(time_points, results[1], label = 'soma m')
    plt.plot(time_points, results[2], label = 'axon w')
    plt.plot(time_points, results[3], label = 'axon m')
    plt.legend()

    soma_het = results[1]/(results[1]+results[0])
    axon_het = results[3]/(results[3]+results[2])
    plt.subplots(figsize=(10, 5))
    plt.plot(time_points, soma_het, label = 'soma het')
    plt.plot(time_points, axon_het, label = 'axon het')
    #plt.ylim([0, 1])
    plt.legend()

    soma_count = results[1]*delta+results[0]
    axon_count = results[3]*delta+results[2]
    plt.subplots(figsize=(10, 5))
    plt.plot(time_points, soma_count, label = 'soma eff. pop. size')
    plt.plot(time_points, axon_count, label = 'axon eff. pop. size')
    plt.ylim(min(0, min(soma_count), min(axon_count)) - 5, max(max(soma_count), max(axon_count)) + 5)
    plt.legend()

    print("Final counts:")
    for i, res in enumerate(results[:,-1]): print(f'{vars[i]}\t{round(res,2)}\t')

    print("Final effective population size:")
    print("soma:",round(soma_count[-1],2))
    print("axon:",round(axon_count[-1],2))

