from copy import deepcopy

import networkx as nx
import numpy as np; np.set_printoptions(suppress=True)
import pandas as pd
from scipy.optimize import minimize

def flux_map_df_from_subnetwork(G):

    map_df = pd.DataFrame()

    nodes = list(G.nodes())
    node_to_index = {n: i for i, n in enumerate(nodes)}

    edges = list(G.edges())
    enumerated_edges_with_data = list(enumerate(G.edges(data = True)))

    n_fluxes = len(edges)

    # name of the source and destination nodes
    map_df['u'] = [u for (u,v) in edges]
    map_df['v'] = [v for (u,v) in edges]

    # location of the rate in the flux matrix
    map_df['flux_mat_i_j'] = [(node_to_index[u],node_to_index[v]) for (u,v) in edges]

    # individual name of each flux variable
    map_df['flux_name'] = [f'flux_{i}' for i in range(n_fluxes)]

    rate = []
    # iterate through the edges in the graph, and set any known terminal branch flux values
    for i, (u, v, data) in enumerated_edges_with_data:
        dest_data = G.nodes(data = True)[v]
        src_data  = G.nodes(data = True)[u]
        
        # if the edge is pointing into a branch terminal, set the rate of the corresponding flux to the influx rate
        if dest_data['nodetype'] != 1 and dest_data['terminal'] == True:
            rate.append(TERMINAL_INFLUX)
        
        # if the edge is pointing away from the branch terminal, set the rate of the corresponding flux to the efflux rate
        elif src_data['nodetype'] != 1 and src_data['terminal'] == True:
            rate.append(TERMINAL_EFFLUX)
        
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

    
for i, flux in enumerate(flux_vector):
    u,v = index_to_edge[i]
    flux_matrix[node_to_index[u],node_to_index[v]] = flux

print('Current flux matrix:')
print(flux_matrix)



unknown_flux_indeces = [i for i, flux in enumerate(flux_vector) if np.isnan(flux)]

unknown_flux_varnames = [fluxnames[i] for i in unknown_flux_indeces]

unknown_flux_array_indeces = {varname:i for i, varname in enumerate(unknown_flux_varnames)}

remap_unknown_flux_array = {v: k for k, v in unknown_flux_array_indeces.items()}



# for each node, calculate the influx and efflux terms. If they include unkowns, include them in a list of constraints
constr = '\tarray = [\n'
for i, node in enumerate(nodelist):
    contains_unknown = False
    
    flux = f"\t\t# flux at {node}\n\t\t"
    # calculate influx rates 
    influx = ""
    for j in range(N):
        flux_ratio = flux_matrix[i,j]
        if flux_ratio != 0:
            #influx += f'+{(index_to_node[j],index_to_node[i])}'
            
            if np.isnan(flux_ratio):
                contains_unknown = True
                
                flux_varname = edge_to_fluxname[(index_to_node[j],index_to_node[i])]
                influx += f'+(x[{unknown_flux_array_indeces[flux_varname]}]*{stedstate_pop[j]})'
            
            else:
                influx += f'+({flux_ratio*stedstate_pop[j]})'
    
    flux += influx
    
    # calculate efflux rates 
    efflux = ""
    for j in range(N):
        flux_ratio = flux_matrix[j,i]
        if flux_ratio != 0:
            #efflux += f'-{(index_to_node[i],index_to_node[j])}'
            
            if np.isnan(flux_ratio):
                contains_unknown = True
                
                flux_varname = edge_to_fluxname[(index_to_node[i],index_to_node[j])]
                efflux += f'-(x[{unknown_flux_array_indeces[flux_varname]}]*{stedstate_pop[j]})'
            
            else:
                efflux += f'-({flux_ratio*stedstate_pop[j]})'
    
    flux += efflux
    
    flux += f' - {steady_state_flux[i]},\n'
    
    if contains_unknown:
        constr += flux
    
constr += '\t\t]'
        
constr_prog = 'global flux_constrints\ndef flux_constraints(x):\n'
constr_prog += constr
constr_prog += '\n\treturn np.array(array)'

print(constr_prog)

global flux_constraints
flux_constraints = None
exec(constr_prog)

bnds = [(0.1, 100) for _ in range(len(unknown_flux_varnames))]

def obj_to_minim(x):
    return np.sum(x)


cons = {
    'type': 'eq', 
    'fun': flux_constraints
    }

start_x = [1 for _ in range(len(unknown_flux_varnames))]

solution = minimize(
    fun = obj_to_minim, 
    x0 = start_x, 
    bounds = bnds, 
    constraints=cons,
    options={'disp':True},
    tol = 0.001,
    method='trust-constr'
    )

optimized_fluxes = solution.x

for i, flux in enumerate(optimized_fluxes):
    u, v = fluxname_to_edge[remap_unknown_flux_array[i]]
    print(u, v, flux)
    flux_matrix[node_to_index[v],node_to_index[u]] = flux

print(np.round(flux_matrix, 3))