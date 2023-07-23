print("<<<< C AND DELTA PARAMETER SWEEP >>>>")

import numpy as np; np.set_printoptions(suppress=True, linewidth=180); np.seterr(divide='ignore')
import pandas as pd; pd.set_option('display.width', 500)


from sim_param_from_network import names_from_network, sde_param_from_network, ode_from_network, start_state_from_nodes
from neuron_graph_helper import load_pickled_neuron_graph
from analyse_simulation import two_component_statistics

from simulate import simulate_ode, simulate_gillespie, simulate_tauleaping


DELTA_VALUES = [0.25, 0.4,  0.5,  0.68, 0.75]
DELTA_NAMES =  ['D25','D40','D50','D68','D75']
MODEL_NAMES = ['model_0', 'model_1']



for DELTA, DELTA_NAME in zip(DELTA_VALUES, DELTA_NAMES):
    for MODEL in MODEL_NAMES:
        print(f"\n**** preparing to simulate {MODEL} with delta = {DELTA} ****")
        
        # load graph with attributes
        G = load_pickled_neuron_graph(f'neuron_graphs/{MODEL}.pkl')

        # set delta and NSS to desired value
        DELTA = DELTA
        NSS = 210 # this NSS value divides nicely into even integers with the given delta values
        for node, data in G.nodes(data = True):
            if data['nodetype'] == 1:
                data['delta'] = DELTA
            data['nss'] = NSS

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
        print("- Simulations will be performed with the following C values:")
        print(C_B_val, '\n')


        # create dataframe for storeing results
        stats_df = pd.DataFrame()

        # perform parameter scan
        for i, c_b in enumerate(C_B_val):
            print(f"\n<<<< STARTING SIMULATION {i} WITH C_B = {c_b} >>>>\n")
            
            # set c_b value
            for node, data in G.nodes(data = True):
                if data['nodetype'] == 1:
                    data['c_b'] = c_b

            # infer the parameterse of the sde systems
            SDE_PARAM = sde_param_from_network(G, prnt=False)

            # run the gillespie simulation
            gillespie_results = simulate_gillespie(SDE_PARAM, TIME_POINTS, START_STATE, replicates=REP, n_cpu=190)
            
            df_g, stats_g = two_component_statistics(gillespie_results, TIME_POINTS, DELTA)
            df_g.to_csv(f'sim_out/{MODEL}/{DELTA_NAME}/paramdf_{i}_gillespie.csv')

            stats_g['simtype'] = 'g'; stats_g['cb'] = c_b; stats_g['i'] = i
            stats_df = stats_df.append(pd.Series(stats_g), ignore_index=True)
            
            print(df_g.iloc[[0, -1]]);print()
            
            
            # run the tau leaping simulation
            tauleaping_results = simulate_tauleaping(SDE_PARAM, TIME_POINTS, START_STATE, replicates=REP, timestep=0.005, n_cpu=190)
            
            df_t, stats_t = two_component_statistics(tauleaping_results, TIME_POINTS, DELTA)
            df_t.to_csv(f'sim_out/{MODEL}/{DELTA_NAME}/paramdf_{i}_tauleaping.csv')

            stats_t['simtype'] = 't'; stats_t['cb'] = c_b; stats_t['i'] = i
            stats_df = stats_df.append(pd.Series(stats_t), ignore_index=True)
            
            print(df_t.iloc[[0, -1]]);print()
            
            print("\n ----- \n")
            
        # print and write collected stats
        print(stats_df)
        stats_df.to_csv(f"sim_out/{MODEL}/{DELTA_NAME}/statsdf.csv")
