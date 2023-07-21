print("<<<< C PARAMETER SWEEP >>>>")

import numpy as np; np.set_printoptions(suppress=True, linewidth=180); np.seterr(divide='ignore')
import pandas as pd; pd.set_option('display.width', 500)


from sim_param_from_network import names_from_network, sde_param_from_network, ode_from_network, start_state_from_nodes
from neuron_graph_helper import load_pickled_neuron_graph
from analyse_simulation import two_component_statistics

from simulate import simulate_ode, simulate_gillespie, simulate_tauleaping


# load graph with attributes
G = load_pickled_neuron_graph('neuron_graphs/2_compartment_model.pkl')

# set delta to desired value
DELTA = 0.25
G.nodes()['S0B']['delta'] = DELTA

# get node names and start state
VARS, NODES = names_from_network(G)
START_STATE = start_state_from_nodes(G, heteroplasmy=0.5, delta=DELTA)

# set the timesteps for the simulation
TIME_POINTS = np.linspace(0, 4000, 1001)

# number of replicates
REP = 10000

# get the parameter values for which the simulations will be run
C_B_val = sequence = [round(i * 10**-decimals, 14) for decimals in range(2, 13) for i in range(10, 0, -1)]
#C_B_val = [element for i, element in enumerate(C_B_val) if i % 10 == 0] # sparseify for testing
print("preparing to simulate with the following parameters:")
print(C_B_val, '\n')


# create dataframe for storeing results
stats_df = pd.DataFrame()

# perform parameter scan
for i, c_b in enumerate(C_B_val):
    print(f"\n<<<< STARTING SIMULATION {i} WITH C_B = {c_b} >>>>\n")
    
    G.nodes()['S0B']['c_b'] = c_b

#     # infer the ode model
#     ode_model = ode_from_network(G, prnt=True)

#     # run the ode model
#     ode_results = simulate_ode(ode_model, TIME_POINTS, START_STATE)
#     plot_ode_results(ode_results, TIME_POINTS, DELTA, VARS, NODES, prnt=False)

    # infer the parameterse of the sde systems
    SDE_PARAM = sde_param_from_network(G, prnt=False)

    # run the gillespie simulation
    gillespie_results = simulate_gillespie(SDE_PARAM, TIME_POINTS, START_STATE, replicates=REP, n_cpu=190)
    
    df_g, stats_g = two_component_statistics(gillespie_results, TIME_POINTS, DELTA)
    df_g.to_csv(f'hpc/model_1/paramdf_{i}_gillespie.csv')

    stats_g['simtype'] = 'g'
    stats_g['cb'] = c_b
    stats_g['i'] = i
    stats_df = stats_df.append(pd.Series(stats_g), ignore_index=True)
    
    print(df_g.iloc[[0, -1]]);print()
    
    
    # run the tau leaping simulation
    tauleaping_results = simulate_tauleaping(SDE_PARAM, TIME_POINTS, START_STATE, replicates=REP, timestep=0.005, n_cpu=190)
    
    df_t, stats_t = two_component_statistics(tauleaping_results, TIME_POINTS, DELTA)
    df_t.to_csv(f'hpc/model_1/paramdf_{i}_gillespie.csv')

    stats_t['simtype'] = 't'
    stats_t['cb'] = c_b
    stats_t['i'] = i
    stats_df = stats_df.append(pd.Series(stats_t), ignore_index=True)
    
    print(df_t.iloc[[0, -1]]);print()
    
    print("\n ----- \n")
    
# print and write collected stats
print(stats_df)
stats_df.to_csv("hpc/model_1/statsdf.csv")
