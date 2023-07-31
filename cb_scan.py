import argparse
import os
import pickle

import numpy as np; np.set_printoptions(suppress=True, linewidth=180); np.seterr(divide='ignore')
import pandas as pd; pd.set_option('display.width', 500)

from sim_param_from_network import sde_param_from_network, start_state_from_nodes
from neuron_graph_helper import load_pickled_neuron_graph
from simulate import simulate_gillespie, simulate_tauleaping


# folder names for delta values
delta_folders = {0.25:'D25', 0.40:'D40', 0.50:'D50', 0.68:'D68', 0.75:'D75'}

# c_b values to scan through
C_VALS = sequence = [round(i * 10**-decimals, 14) for decimals in range(2, 13) for i in range(10, 0, -1)]
#C_VALS = [element for i, element in enumerate(C_VALS) if i % 10 == 0] # sparseify for testing
#print("- Simulations will be performed with the following C values:")
#print(C_VALS, '\n')

# check that output directories are already present, and create if not
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)



def cb_scan(model, delta, replicates, time):
    print(f"\n**** preparing to simulate {model} with delta = {delta} ****")
        
    # load graph with attributes
    G = load_pickled_neuron_graph(f'neuron_graphs/{model}.pkl')

    # set delta and NSS to desired value
    NSS = 210 # this NSS value divides nicely into even integers with the given delta values
    for node, data in G.nodes(data = True):
        if data['nodetype'] == 1:
            data['delta'] = delta
        data['nss'] = NSS

    # get node names and start state
    START_STATE = start_state_from_nodes(G, heteroplasmy=0.5, delta=delta)

    # set the timesteps for the simulation
    TIME_POINTS = np.linspace(0, time, 1001)
    
    # perform parameter scan
    for i, c_b in enumerate(C_VALS):
        print(f"\n<<<< STARTING SIMULATION {i} WITH C_B = {c_b} >>>>\n")
        
        # set c_b value
        for node, data in G.nodes(data = True):
            if data['nodetype'] == 1:
                data['c_b'] = c_b

        # infer the parameterse of the sde systems
        SDE_PARAM = sde_param_from_network(G, prnt=False)


        # run the gillespie simulation
        gillespie_results = simulate_gillespie(
            SDE_PARAM, TIME_POINTS, START_STATE, 
            replicates=replicates, n_cpu=254
            )
        # export the results
        print(f"Wrote full simulation state to a .pkl file")
        with open(f'sim_out/{model}/{delta_folders[delta]}/gillespie_results_{i}.pkl', 'wb') as f:
            pickle.dump(gillespie_results, f)
        

        # run the tau leaping simulation
        tauleaping_results = simulate_tauleaping(
            SDE_PARAM, TIME_POINTS, START_STATE, timestep=0.005,
            replicates=replicates, n_cpu=254
            )
        # export the results
        print(f"Wrote full simulation state to a .pkl file")
        with open(f'sim_out/{model}/{delta_folders[delta]}/tauleaping_results_{i}.pkl', 'wb') as f:
            pickle.dump(tauleaping_results, f)

        

        print("\n ----- \n")




# run cb_scan function with parameters from command line
if __name__ == '__main__':  # parse command line arguments
    parser = argparse.ArgumentParser(description='Run cb_scan function')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--delta', type=float, required=True, help='Delta value')
    parser.add_argument('--replicates', type=int, required=True, help='Number of replicates')
    parser.add_argument('--time', type=int, required=True, help='Amount of time to simulate')
    args = parser.parse_args()

    ensure_dir(f'sim_out/{args.model}/{delta_folders[args.delta]}/')

    cb_scan(args.model, args.delta, args.replicates, args.time)