from copy import deepcopy

import networkx as nx
import numpy as np; np.set_printoptions(suppress=True)
import pandas as pd
from scipy.optimize import minimize

# find forward and reverse edge pairs in a subgraph
def find_fr_pairs(G):
    fr_pairs = {}
    for u, v, data in G.edges(data=True):
        if data['edgetype'] != 5:
            # exclude reverse directed edges as they will be covered when their corresponding forward edge is processed
            if data['direction'] == 'reverse':
                continue
            # get base names of the nodes (without 'F', 'R', 'T', 'B' suffix)
            u_base = u.rstrip("FRTB")
            v_base = v.rstrip("FRTB")
            # find the corresponding reverse edge (if exists)
            reverse_u = v_base + 'R' if v.endswith('F') else (v_base + 'T' if v.endswith('T') else v_base + 'B')
            reverse_v = u_base + 'R' if u.endswith('F') else (u_base + 'T' if u.endswith('T') else u_base + 'B')
            # if the reverse edge exists and the pair has not been added before
            if G.has_edge(reverse_u, reverse_v):
                fr_pairs[(u, v)] = (reverse_u, reverse_v)
    return fr_pairs

#
def flux_map_df_from_subnetwork(G, bio_param):
    terminal_inflix = 2*float(bio_param['death_rate'])
    terminal_efflux = float(bio_param['death_rate'])
    
    map_df = pd.DataFrame()

    nodes = list(G.nodes())
    node_to_index = {n: i for i, n in enumerate(nodes)}

    edges = list(G.edges())
    enumerated_edges_with_data = list(enumerate(G.edges(data = True)))

    n_fluxes = len(edges)

    # name of the source and destination nodes
    map_df['(u, v)'] = [(u,v) for (u,v) in edges]
    
    # index of the reverse edge, if it exists
    f_r_pairs = find_fr_pairs(G)
    map_df['reverse_pair'] = [f_r_pairs[edge] if edge in f_r_pairs.keys() else None for edge in edges]

    # location of the rate in the flux matrix
    map_df['flux_mat_i_j'] = [(node_to_index[u],node_to_index[v]) for (u,v) in edges]

    rate = []
    # iterate through the edges in the graph, and set any known terminal branch flux values
    for i, (u, v, data) in enumerated_edges_with_data:
        dest_data = G.nodes(data = True)[v]
        src_data  = G.nodes(data = True)[u]
        
        # if the edge is pointing from a non-terminal into a branch terminal, set the rate of the corresponding flux to the influx rate
        if dest_data['nodetype'] != 1 and dest_data['terminal'] == True:
            rate.append(terminal_inflix)
        
        # if the edge is pointing away from the branch terminal to a non-terminal, set the rate of the corresponding flux to the efflux rate
        elif src_data['nodetype'] != 1 and src_data['terminal'] == True:
            rate.append(terminal_efflux)
        
        # otherwise the edge has an unknown rate which must be inferred
        else:
            rate.append(np.nan)

    # rate given by the user
    map_df['specified_rate'] = rate

    # names of the 
    unknown_name = []
    unknown_count = 0
    for i in range(n_fluxes):
        if np.isnan(map_df['specified_rate'][i]):
            unknown_name.append(f'x[{unknown_count}]')
            unknown_count += 1
        else:
            unknown_name.append(None)
    map_df['unknown_name'] = unknown_name

    return map_df

# fluxes between each node in the subgraph
def flux_matrix_from_subnetwork(G, map_df):
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    
    flux_matrix = np.zeros((n_nodes, n_nodes))
    
    for i, row in map_df.iterrows():
        (i, j) = row['flux_mat_i_j']
        flux_matrix[i,j] = row['specified_rate']
        
    return flux_matrix
    


# get constraints that arise from the fact that the inflow and outflow from each node must sum to 0
def get_mass_conserve_constraints(
        G, 
        flux_map_df, 
        flux_matrix, 
        steady_state_pop, 
        steady_state_flux,
        prnt = False,
        ):
    
    nodes = list(G.nodes()); n_nodes = len(nodes)
    
    # for each node, calculate the influx and efflux terms. If they include unkowns, include them in a list of constraints
    
    constr = ''
    # get the sum of inflows and outflows into a node, if the variables are unknown, represent as such
    for i, node in enumerate(nodes):
        # infer the node type (no constraints on soma nodes)
        node_type = G.nodes(data = True)[node]['nodetype']
        
        # does the flux contain variables that are unknown?
        contains_unknown = False

        # begin the text for the flux constraint with a comment about the node it affects
        flux_at_node = f"\t\t# flux at {node}\n\t\t"
        
        # calculate influx rates 
        influx = ""
        for j in range(n_nodes):
            flux = flux_matrix[j,i]
            if flux != 0:
                # if flux rate is unknown
                if np.isnan(flux):
                    contains_unknown = True
                    
                    unknown_name = flux_map_df.loc[flux_map_df['flux_mat_i_j'] == (j,i)]["unknown_name"].values[0]
                    influx += f'+({unknown_name}*{steady_state_pop[j]})'

                # if flux rate is known 
                else:
                    influx += f'+({float(flux)*steady_state_pop[j]})'
        # add the calculated terms to the flux statement
        flux_at_node += influx
        
        # calculate efflux rates 
        efflux = ""
        for j in range(n_nodes):
            flux = flux_matrix[i,j]
            if flux != 0:
                # if flux rate is unknown
                if np.isnan(flux):
                    contains_unknown = True
                    
                    unknown_name = flux_map_df.loc[flux_map_df['flux_mat_i_j'] == (i,j)]["unknown_name"].values[0]
                    efflux += f'-({unknown_name}*{steady_state_pop[j]})'

                # if flux rate is known 
                else:
                    efflux += f'-({float(flux)*steady_state_pop[j]})'
        # add the calculated terms to the flux statement                    
        flux_at_node += efflux
        
        # subtract the steady state flux (the deaths at each node)
        flux_at_node += f' - {steady_state_flux[i]},\n'
        
        # only add the constraint to the set of constraints if it contains an unknown, and the node is not somal
        if contains_unknown and node_type != 1: constr += flux_at_node

    # make the text for the constraint program
    constr_prog = 'global flux_constraints\ndef flux_constraints(x):\n\tarray = [\n'
    constr_prog += constr
    constr_prog += '\t\t]\n\treturn np.array(array)'
    
    print(constr_prog)

    # execute the text created above as a program, to yield the flux_constraint function
    global flux_constraints
    flux_constraints = None
    exec(constr_prog)

    return flux_constraints


# get constraints from the anterograde retrograde flux
def get_flux_ratio_constraints(flux_map_df, ratio, epsilon):
    # get the list of forward-reverse names for unknown variables
    f_r_varname_pairs = []
    for i, row in flux_map_df.iterrows():
        if row['reverse_pair'] != None:
            reverse_pair_unknown_name = flux_map_df[flux_map_df['(u, v)'] == row['reverse_pair']]['unknown_name'].values[0]
            if reverse_pair_unknown_name != None:
                f_r_varname_pairs.append((row['unknown_name'], reverse_pair_unknown_name))
    
    # construct the text of the ratio constraints. These inequaliteies guarantee that the forward/reverse flux ratio is within 'epsilon' of 'ratio'
    constr = ''
    for forward, reverse in f_r_varname_pairs:
        constr += f'\t\t({forward} - {reverse}*{np.round(ratio * (1 - epsilon), 4)}),  # first inequality \n'
        constr += f'\t\t({reverse}*{np.round(ratio * (1 + epsilon),4)} - {forward}),  # second inequality\n'
                      
            
    # make the text for the constraint program
    constr_prog = 'global ratio_constraints\ndef ratio_constraints(x):\n\tarray = [\n'
    constr_prog += constr
    constr_prog += '\t\t]\n\treturn np.array(array)'
    
    print(constr_prog)

    # execute the text created above as a program, to yield the flux_constraint function
    global ratio_constraints
    ratio_constraints = None
    exec(constr_prog)
    
    return ratio_constraints

# objective function is to minimize is the total flux in the system
def total_flux(x):
    return np.sum(x)

# objective function is linear so hessian is all 0s
def total_flux_hessian(x):
    return np.zeros((len(x), len(x)))

# get the lb and ub for each of the flux parameters solved by the minimizer
def get_minimizer_bounds(flux_map_df, lb = 0.1, ub = 100):
    bounds = []
    
    for i, row in flux_map_df.iterrows():
        if row['unknown_name'] != None:
            bounds.append((lb, ub))
            
    return bounds

# get the starting values of the flux parameters solved by the minimizer
def get_minimizer_startval(flux_map_df, val = 1):
    startval = []
    
    for i, row in flux_map_df.iterrows():
        if row['unknown_name'] != None:
            startval.append(val)
            
    return startval


# update the flux map dataframe with solved values
def update_flux_map_df(flux_map_df, solved_fluxes):
    solved_rate = []
    deposited = 0
    for i, row in flux_map_df.iterrows():
        if row['unknown_name'] != None:
            solved_rate.append(solved_fluxes[deposited])
            deposited += 1
        else:
            solved_rate.append(np.nan)
            
    flux_map_df['solved_rate'] = solved_rate
    
    final_rate = []
    for i, row in flux_map_df.iterrows():
        if np.isnan(row['specified_rate']):
            final_rate.append(row['solved_rate'])
        else:
            final_rate.append(row['specified_rate'])
            
    flux_map_df['final_rate'] = final_rate
    
    ant_ret_ratio = []
    for i, row in flux_map_df.iterrows():
        if row['reverse_pair'] != None:
            retrograde_flux = flux_map_df[flux_map_df['(u, v)'] == row['reverse_pair']]['final_rate'].values[0]
            ant_ret_ratio.append(np.round(float(row['final_rate'])/retrograde_flux, 1))
        else:
            ant_ret_ratio.append(None)
            
    flux_map_df['ant_ret_ratio'] = ant_ret_ratio
    
    return flux_map_df

# update the flux matrix with solved values 
def update_flux_matrix(flux_matrix, flux_map_df):
    for i, row in flux_map_df.iterrows():
        (i, j) = row['flux_mat_i_j']
        flux_matrix[i,j] = row['final_rate']
    
    return flux_matrix

# main wrapper to solve fluxes in a subnetwork
def solve_fluxes(G, bio_param, steady_state_pop, steady_state_flux):
    
    print('> Flux dataframe:')
    flux_map_df = flux_map_df_from_subnetwork(G, bio_param)
    print(flux_map_df)

    print('\n> Flux matrix:')
    flux_matrix = flux_matrix_from_subnetwork(G, flux_map_df)
    print(flux_matrix)

    print('\n> Flux constraints:')
    flux_constraints = get_mass_conserve_constraints(G, flux_map_df, flux_matrix, steady_state_pop, steady_state_flux)

    print('\n> Ratio constraints:')
    ratio_constraints = get_flux_ratio_constraints(flux_map_df, 2, 0.2)
    
    flux_cons = {'type': 'eq', 'fun': flux_constraints}
    ratio_cons = {'type': 'ineq', 'fun': ratio_constraints}
    cons = [flux_cons, ratio_cons]

    bnds = get_minimizer_bounds(flux_map_df, lb = 0.01, ub = 100)
    strt = get_minimizer_startval(flux_map_df)

    print('\n> Optimizing flux values...')
    solution = minimize(
        method='trust-constr',
        fun = total_flux, 
        hess = total_flux_hessian,
        x0 = strt, 
        bounds = bnds, 
        constraints=cons,
        options={'disp':True, 'maxiter':1000},
        #tol = 0.0001,
        )

    solved_fluxes = np.round(solution.x, 4)

    print('\n> Solved flux dataframe:')
    flux_map_df = update_flux_map_df(flux_map_df, solved_fluxes)
    print(flux_map_df)

    print('\n> Solved flux matrix:')
    flux_matrix = update_flux_matrix(flux_matrix, flux_map_df)
    print(np.round(flux_matrix, 2))

    flux_dict = {row['(u, v)']:row['final_rate'] for i, row in flux_map_df.iterrows()} 

    return flux_dict