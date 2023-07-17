import networkx as nx
import pickle

import numpy as np; np.set_printoptions(suppress=True, linewidth=180); np.seterr(divide='ignore')
import pandas as pd; pd.set_option('display.width', 500)


from plot_module import plot_ode_results, plot_sde_results, plot_simulator_graph, plot_neuron_graph_subset
from sim_param_from_network import names_from_network, sde_param_from_network, ode_from_network, start_state_from_nodes
from neuron_graph_process import neuron_graph_process
from neuron_graph_helper import load_pickled_neuron_graph
from analyse_simulation import two_component_statistics

from simulate import simulate_ode, simulate_gillespie, simulate_tauleaping

G = load_pickled_neuron_graph('neuron_graphs/2_compartment_model.pkl')

DELTA = 0.25
G.nodes()['S0B']['delta'] = DELTA

VARS, NODES = names_from_network(G)
START_STATE = start_state_from_nodes(G, heteroplasmy=0.5, delta=DELTA)


C_B_val = [
    0.1,
    0.0001, 
    0.0000001, 
]

TIME_POINTS = np.linspace(0, 4000, 1001)
REP = 5000


for i, c_b in enumerate(C_B_val):
    print(f"\n<<<< STARTING SIMULATION WITH C_B = {c_b} >>>>\n")
    
    G.nodes()['S0B']['c_b'] = c_b

#     # infer the ode model
#     ode_model = ode_from_network(G, prnt=True)

#     # run the ode model
#     ode_results = simulate_ode(ode_model, TIME_POINTS, START_STATE)
#     plot_ode_results(ode_results, TIME_POINTS, DELTA, VARS, NODES, prnt=False)

    # infer the parameterse of the sde systems
    SDE_PARAM = sde_param_from_network(G, prnt=False)

    # run the gillespie simulation
    gillespie_results = simulate_gillespie(SDE_PARAM, TIME_POINTS, START_STATE, replicates=REP)
    #plot_sde_results(gillespie_results, TIME_POINTS, DELTA, VARS, NODES, prnt = False)
    df_g = two_component_statistics(gillespie_results, DELTA)
    df_g.to_csv(f'hpc/results_{i}_gillespie.csv')
    print(df_g.iloc[[0, -1]]);print()
    
    # run the tau leaping simulation
    tauleaping_results = simulate_tauleaping(SDE_PARAM, TIME_POINTS, START_STATE, replicates=REP, timestep=0.01)
    #plot_sde_results(tauleaping_results, TIME_POINTS, DELTA, VARS, NODES, prnt = False)
    df_t = two_component_statistics(tauleaping_results, DELTA)
    df_t.to_csv(f'hpc/results_{i}_tauleap.csv')
    print(df_t.iloc[[0, -1]]);print()
    
    print("\n ----- \n")

