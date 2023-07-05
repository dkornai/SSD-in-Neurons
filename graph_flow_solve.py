import numpy as np; np.set_printoptions(suppress=True, linewidth=180)
import pandas as pd; pd.set_option('display.width', 500)
from scipy.optimize import minimize


def mat_print(data):
    # df=pd.DataFrame(
    #     data=data[0:,0:],
    #     index=[i for i in range(data.shape[0])],
    #     columns=['f'+str(i) for i in range(data.shape[1])])
    # print(df)
    
    def pad(element, target_len):
        out = str(element)
        while len(out) < target_len:
            out += ' '
        return out
    def longest_element(data):
        return max([max([len(str(element)) for element in row]) for row in data])

    target_len = longest_element(data)
                   
    for row in data:
        string = ''
        for element in row:
            string += f'{pad(element, target_len)} '
        print(string)

def replace_text(input_text, replacement_dict):
    for key, value in replacement_dict.items():
        input_text = input_text.replace(key, value)
    return input_text

        
        
# find forward and reverse edge pairs in a subgraph
def find_fr_node_pairs(G):
    fr_pairs = {}
    for node, data in G.nodes(data=True):
        if node[-1] == 'F':
            fr_pairs[node] = f'{node[:-1]}R'
    return fr_pairs


# find forward and reverse edge pairs in a subgraph
def find_fr_edge_pairs(G):
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

def get_node_pop_dict(G):
    node_pop_dict = {}
    unknowns = 0
    for node, data in G.nodes(data=True):
        if node[-1] not in ['T', 'B']:
            node_pop_dict[node] = f'p[{unknowns}]'
            unknowns += 1
        else:
            node_pop_dict[node] = data['nss']
    
    return node_pop_dict

def find_reverse_edge_index(G, map_df):
    reverse_index = []
    pairs_dict = find_fr_edge_pairs(G)
    for i, row in map_df.iterrows():
        if (row['u'], row['v']) in pairs_dict:
            u, v = pairs_dict[(row['u'], row['v'])]
            reverse_index.append(int(map_df[(map_df['u'] == u) & (map_df['v'] == v)].index[0]))
        else:
            reverse_index.append(' ')
            
    return reverse_index



# read any user specified net fluxes (applies to axon and dendrite terminals)
def read_given_net_flux(G):
    unknowns = 0
    
    net_flux = []
    # iterate through the edges in the graph, and set any known terminal branch flux values
    for u, v, data in G.edges(data = True):
        dest_data = G.nodes(data = True)[v]
        src_data  = G.nodes(data = True)[u]
        
        # if the edge is pointing from a non-terminal into a branch terminal, set the rate of the corresponding flux to the influx rate
        if dest_data['nodetype'] != 1 and dest_data['terminal'] == True:
            net_flux.append(data['net_flux'])
        
        # if the edge is pointing away from the branch terminal to a non-terminal, set the rate of the corresponding flux to the efflux rate
        elif src_data['nodetype'] != 1 and src_data['terminal'] == True:
            net_flux.append(data['net_flux'])
        
        # otherwise the edge has an unknown rate which must be inferred
        else:
            net_flux.append(None)
            unknowns += 1
            
    return net_flux

# infer any inferrable flux rates (doable when both the population size, and the net flux is specified)
def infer_flux_rate(in_u_popsize, in_net_flux):
    flux_rate = []
    unknowns = 0
    for i in range(len(in_u_popsize)):
        net_flux = in_net_flux[i]
        u_popsize = in_u_popsize[i]
        
        # if the fluxrate can be fully inferred from the starting data
        if net_flux != None and type(u_popsize) != str:
            flux_rate.append(np.round(net_flux/u_popsize, 6))
        else:
            flux_rate.append(f'r[{unknowns}]')
            unknowns += 1
            
    return flux_rate

# generate the dicts that remap p[i] and r[j] to a single vector x[k]
def get_variable_name_remap(map_df):
    u_popsize = map_df['u_popsize'].tolist()
    flux_rate = map_df['flux_rate'].tolist()
    
    orig_to_x = {}
    unknowns = 0
    for element in flux_rate:
        if type(element) == str:
            if element not in orig_to_x:
                orig_to_x[element] = f'x[{unknowns}]'
                unknowns += 1
    for element in u_popsize:
        if type(element) == str:
            if element not in orig_to_x:
                orig_to_x[element] = f'x[{unknowns}]'
                unknowns += 1
    
    x_to_orig = {v:k for k, v in orig_to_x.items()}
    
    return orig_to_x, x_to_orig

# define any remaining net fluxes in terms of the source population size, and the flux rate
def define_unknown_net_fluxes(in_u_popsize, in_flux_rate, in_net_flux):
    for i, net_flux in enumerate(in_net_flux):
        flux_rate = in_flux_rate[i]
        u_popsize = in_u_popsize[i]
        
        if net_flux == None:
            in_net_flux[i] = f'{u_popsize}*{flux_rate}'
            
    return in_net_flux

def flux_map_df_from_subnetwork(G):
    
    nodes = list(G.nodes())
    node_to_index = {n: i for i, n in enumerate(nodes)}

    edges = list(G.edges())
    enumerated_edges_with_data = list(enumerate(G.edges(data = True)))
    n_fluxes = len(edges)
    
    
    map_df = pd.DataFrame()
    
    # name of the source and destination nodes
    map_df['u'] = [u for (u,v) in edges]
    map_df['v'] = [v for (u,v) in edges]
    
    # direction of the flux
    map_df['direction'] = [data['direction'] for u, v, data in G.edges(data=True)]
    
    # location of the rate in the flux matrix
    map_df['flux_mat_i_j'] = [(node_to_index[u],node_to_index[v]) for (u,v) in edges]
    
    # find the index of the reverse pair
    map_df['reverse_pair'] = find_reverse_edge_index(G, map_df)
    
    # get the population sizes (if known) for each of the source nodes
    node_pop_dict = get_node_pop_dict(G)
    u_popsize = [node_pop_dict[row['u']] for i, row in map_df.iterrows()]
    map_df['u_popsize'] = u_popsize
    
    # read given net fluxes
    net_flux = read_given_net_flux(G)
    # infer any flux rates if possible
    flux_rate = infer_flux_rate(u_popsize, net_flux)
    # update the net flux section 
    net_flux = define_unknown_net_fluxes(u_popsize, flux_rate, net_flux)
    
    
    map_df['flux_rate'] = flux_rate
    map_df['net_flux'] = net_flux

    return map_df

# fluxes between each node in the subgraph
def flux_matrix_from_subnetwork(G, map_df):
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    
    flux_matrix = np.zeros((n_nodes, n_nodes)).tolist()
    
    for i, row in map_df.iterrows():
        (i, j) = row['flux_mat_i_j']
        flux_matrix[i][j] = row['net_flux']
        
    return flux_matrix

# get constraints that guarantee that populations in forwards and backwards node pairs sum to the desired axon node population
def get_pop_sum_constraints(
        orig_to_x,
        G,
        ):
    
    nodes = list(G.nodes()); n_nodes = len(nodes)
    desired_populations = {node:data.get('nss', None) for node, data in G.nodes(data = True)}
    
    # get the population size, or the variable representing population size for the node
    node_pop_dict = get_node_pop_dict(G)
    
    # get populatoins that are forward-reverse pairs
    fr_node_pairs = find_fr_node_pairs(G)
    
    constr = ''
    # write the terms for each node pair
    for forward, reverse in fr_node_pairs.items():
        constr += f'\t\t# pops. of {forward} and {reverse} must sum to {desired_populations[forward]}\n'
        constr += f'\t\t{node_pop_dict[forward]}+{node_pop_dict[reverse]}-{desired_populations[forward]},\n'
    
    
    # make the text for the constraint program
    constr_prog = 'global pop_sum_constraints\ndef pop_sum_constraints(x):\n\tarray = [\n'
    constr_prog += constr
    constr_prog += '\t\t]\n\treturn np.array(array)'
    
    print(constr)
    # replace human readable variable names with those that mapp all categories to a single vector
    constr_prog = replace_text(constr_prog, orig_to_x)
    #print(constr_prog)
    
    # execute the text created above as a program, to yield the flux_constraint function
    global pop_sum_constraints
    pop_sum_constraints = None
    exec(constr_prog)

    return pop_sum_constraints

# get constraints that guarantee that populations in forwards and backwards node pairs sum to the desired axon node population
def get_net_flux_constraints(
        orig_to_x,
        map_df,
        ):
    
    in_u_popsize = map_df['u_popsize'].tolist()
    in_flux_rate = map_df['flux_rate'].tolist()
    in_net_flux = map_df['net_flux'].tolist()
    
    constr = ''
    # write the terms for each node pair
    for u_popsize, flux_rate, net_flux in zip(in_u_popsize, in_flux_rate, in_net_flux):
        if type(net_flux) == float and type(flux_rate) == str and type(u_popsize) == str:
            constr += f'\t\t# source population ({u_popsize}) * flux rate ({flux_rate}) must equal to {net_flux}\n'
            constr += f'\t\t({u_popsize}*{flux_rate})-{net_flux},\n'
    
    
    # make the text for the constraint program
    constr_prog = 'global net_flux_constraints\ndef net_flux_constraints(x):\n\tarray = [\n'
    constr_prog += constr
    constr_prog += '\t\t]\n\treturn np.array(array)'
    
    print(constr)
    # replace human readable variable names with those that mapp all categories to a single vector
    constr_prog = replace_text(constr_prog, orig_to_x)
    #print(constr_prog)
    
    # execute the text created above as a program, to yield the flux_constraint function
    global net_flux_constraints
    net_flux_constraints = None
    exec(constr_prog)

    return net_flux_constraints


# get constraints that arise from the fact that the inflow and outflow from each node must sum to 0
def get_mass_conserve_constraints(
        orig_to_x,
        G,
        flux_matrix,
        ):
    nodes = list(G.nodes()); n_nodes = len(nodes)
    
    # get the population size, or the variable representing population size for the node
    node_pop_dict = get_node_pop_dict(G)

    
    constr = ''
    # get the sum of inflows and outflows into a node, if the variables are unknown, represent as such
    for i, node in enumerate(nodes):
        node_data = G.nodes(data = True)[node]
        
        node_type = node_data['nodetype']
        node_pop = node_pop_dict[node]
        node_death_rate = node_data['death_rate']

        
        # begin the text for the flux constraint with a comment about the node it affects
        flux_at_node = f"\t\t# flux at {node}\n\t\t"
        
        contains_unknown = False  # does the flux contain variables that are unknown?
    
    
        # add any influx terms
        influx = ""
        for j in range(n_nodes):
            flux = flux_matrix[j][i]
            if flux != 0:
                if type(flux) == str: contains_unknown = True
                influx += f'+({flux})'
        flux_at_node += influx
        
        
        # subtract any efflux terms
        efflux = " "
        for j in range(n_nodes):
            flux = flux_matrix[i][j]
            if flux != 0:
                if type(flux) == str: contains_unknown = True
                efflux += f'-({flux})'             
        flux_at_node += efflux
        
        
        # subtract the the deaths at each node
        flux_at_node += f' -({node_pop}*{node_death_rate}),\n'
        
        
        # only add the constraint if it contains an unknown, 
        if contains_unknown:
            if node_type != 1: # and the node is not in the soma
                constr += flux_at_node

                
    # make the text for the constraint program
    constr_prog = 'global mass_conserve_constraints\ndef mass_conserve_constraints(x):\n\tarray = [\n'
    constr_prog += constr
    constr_prog += '\t\t]\n\treturn np.array(array)'
    
    print(constr)
    # replace human readable variable names with those that mapp all categories to a single vector
    constr_prog = replace_text(constr_prog, orig_to_x)
    #print(constr_prog)
    
    # execute the text created above as a program, to yield the flux_constraint function
    global mass_conserve_constraints
    mass_conserve_constraints = None
    exec(constr_prog)

    return mass_conserve_constraints


# get constraints from the anterograde retrograde flux
def get_flux_ratio_constraints(
        orig_to_x,
        flux_map_df,  
        ratio, 
        epsilon
        ):
    
    # get the list of forward-reverse names for unknown variables
    f_r_varname_pairs = []
    for i, row in flux_map_df.iterrows():
        reverse_row_i = row['reverse_pair']
        if type(reverse_row_i) == int:
            reverse_pair_row = flux_map_df.iloc[[reverse_row_i]]
            reverse_flux_rate = reverse_pair_row['flux_rate'].values[0]
            reverse_u_population = reverse_pair_row['u_popsize'].values[0]
            
            f_r_varname_pairs.append((row['flux_rate'], row['u_popsize'], reverse_flux_rate, reverse_u_population))
    
    # construct the text of the ratio constraints. These inequaliteies guarantee that the forward/reverse flux ratio is within 'epsilon' of 'ratio'
    constr = ''
    for forward_rate, forward_pop, reverse_rate, reverse_pop in f_r_varname_pairs:
        constr += f'\t\t# anterograde to retrograde net flux ratio: {np.round(ratio * (1 - epsilon), 4)} <= (({forward_rate}*{forward_pop})/({reverse_rate}*{reverse_pop})) <= {np.round(ratio * (1 + epsilon),4)}\n'
        constr += f'\t\t(({forward_rate}*{forward_pop}) - ({reverse_rate}*{reverse_pop})*{np.round(ratio * (1 - epsilon), 4)}),\n'
        constr += f'\t\t(({reverse_rate}*{reverse_pop})*{np.round(ratio * (1 + epsilon),4)} - ({forward_rate}*{forward_pop})),\n'
                      
            
    # make the text for the constraint program
    constr_prog = 'global flux_ratio_constraints\ndef flux_ratio_constraints(x):\n\tarray = [\n'
    constr_prog += constr
    constr_prog += '\t\t]\n\treturn np.array(array)'
    
    print(constr)
    # replace human readable variable names with those that mapp all categories to a single vector
    constr_prog = replace_text(constr_prog, orig_to_x)
    #print(constr_prog)
    
    # execute the text created above as a program, to yield the flux_constraint function
    global flux_ratio_constraints
    flux_ratio_constraints = None
    exec(constr_prog)
    
    return flux_ratio_constraints

# get constraints that guarantee that populations in forwards and backwards node pairs sum to the desired axon node population
def get_pop_ratio_constraints(
        orig_to_x,
        G,
        ratio, 
        epsilon,
        ):
    
    # get populatoins that are forward-reverse pairs
    fr_node_pairs = find_fr_node_pairs(G)
    # get the populations (or the name of the variable) in each of the ndoes
    node_pop_dict = get_node_pop_dict(G)
    
    constr = ''
    # write the terms for each node pair
    for forward, reverse in fr_node_pairs.items():
        forward_pop = node_pop_dict[forward]; reverse_pop = node_pop_dict[reverse]
        constr += f'\t\t# anterograde to retrograde population ratio: {np.round(ratio * (1 - epsilon), 4)} <= (({forward_pop})/({reverse_pop})) <= {np.round(ratio * (1 + epsilon),4)}\n'
        constr += f'\t\t(({forward_pop}) - ({reverse_pop})*{np.round(ratio * (1 - epsilon), 4)}),\n'
        constr += f'\t\t(({reverse_pop})*{np.round(ratio * (1 + epsilon),4)} - ({forward_pop})),\n'
    
    
    # make the text for the constraint program
    constr_prog = 'global pop_pop_ratio_constraints\ndef pop_pop_ratio_constraints(x):\n\tarray = [\n'
    constr_prog += constr
    constr_prog += '\t\t]\n\treturn np.array(array)'
    
    print(constr)
    # replace human readable variable names with those that mapp all categories to a single vector
    constr_prog = replace_text(constr_prog, orig_to_x)
    #print(constr_prog)
    
    # execute the text created above as a program, to yield the poplation ratio constraint function
    global pop_pop_ratio_constraints
    pop_pop_ratio_constraints = None
    exec(constr_prog)

    return pop_pop_ratio_constraints

# objective function is to minimize is the total flux in the system
def get_objective_function(orig_to_x):
    # count the number of rate parameters, the goal is to minimize their sum
    n_flux_param = len([key for key in orig_to_x if key[0] == 'r'])
    
    # make the text for the constraint program
    constr_prog = f'global total_flux\ndef total_flux(x):\n\treturn np.sum(x[0:{n_flux_param}])'
    #print(constr_prog)
    
    # execute the text created above as a program, to yield the flux_constraint function
    global total_flux
    total_flux = None
    exec(constr_prog)
    
    return total_flux
        

# get the lb and ub for each of the flux parameters solved by the minimizer
def get_minimizer_bounds(
        orig_to_x, 
        flux_rate_lb = 0.1, 
        flux_rate_ub = 100,
        pop_lb = 10,
        pop_ub = 1000,
        ):
    
    bounds = []
    
    for key in orig_to_x:
        if key[0] == 'r': bounds.append((flux_rate_lb, flux_rate_ub))
        elif key[0] == 'p': bounds.append((pop_lb, pop_ub))
            
    return bounds

# get the starting values of the flux parameters solved by the minimizer
def get_minimizer_startval(
        orig_to_x, 
        start_flux_rate = 1,
        start_pop_size = 100,
        ):
    
    startval = []
    
    for key in orig_to_x:
        if key[0] == 'r': startval.append(start_flux_rate)
        elif key[0] == 'p': startval.append(start_pop_size)
            
    return startval

def update_flux_map_df(
        G,
        flux_map_df, 
        results, 
        orig_to_x
        ):
    
    # write new population size and flux rate estimates 
    results_flux_rate = {}
    results_pop = {}
    for i, key in enumerate(orig_to_x):
        if key[0] == 'r':
            results_flux_rate[key] = results[i]
        elif key[0] == 'p':
            results_pop[key] = np.round(results[i])

    flux_map_df['u_popsize'] = flux_map_df['u_popsize'].map(results_pop).fillna(flux_map_df['u_popsize'])
    flux_map_df['flux_rate'] = flux_map_df['flux_rate'].map(results_flux_rate).fillna(flux_map_df['flux_rate'])


    # recalulate the net fluxes at each edge
    in_u_popsize = flux_map_df['u_popsize'].tolist()
    in_flux_rate = flux_map_df['flux_rate'].tolist()
    in_net_flux = flux_map_df['net_flux'].tolist()
    new_net_flux = [net_flux if type(net_flux) == float else flux_rate*u_popsize for u_popsize, flux_rate, net_flux in zip(in_u_popsize, in_flux_rate, in_net_flux)]
    flux_map_df['net_flux'] = new_net_flux


    # get anterograde/retrograde ratios if a reverse pair is available
    # find the index of the reverse pair
    reverse_index = []
    pairs_dict = find_fr_edge_pairs(G)
    for i, row in flux_map_df.iterrows():
        if (row['u'], row['v']) in pairs_dict:
            u, v = pairs_dict[(row['u'], row['v'])]
            reverse_index.append(flux_map_df[(flux_map_df['u'] == u) & (flux_map_df['v'] == v)].index[0])
        else:
            reverse_index.append(None)

    # get the ratio
    ant_ret_ratio = []
    for i, row in flux_map_df.iterrows():
        if reverse_index[i] != None:
            ant_ret_ratio.append(np.round(row['net_flux']/new_net_flux[reverse_index[i]], 2))
        else:
            ant_ret_ratio.append(' ')
    
    flux_map_df['ant_ret_ratio'] = ant_ret_ratio
    
    
    return flux_map_df

# update the flux matrix with solved values 
def update_flux_matrix(flux_matrix, flux_map_df):
    for i, row in flux_map_df.iterrows():
        (i, j) = row['flux_mat_i_j']
        flux_matrix[i][j] = row['net_flux']
    
    return flux_matrix

def solve_subgraph_flux(G):

    print('> Flux dataframe with unknowns:')
    flux_map_df = flux_map_df_from_subnetwork(G)
    orig_to_x, x_to_orig = get_variable_name_remap(flux_map_df)
    print(flux_map_df)

    print('\n> Net flux matrix with unknowns:')
    flux_matrix = flux_matrix_from_subnetwork(G, flux_map_df)
    mat_print(flux_matrix)


    print('\n> Mass conservation constraints:')
    mass_conserve_constraint = get_mass_conserve_constraints(orig_to_x, G, flux_matrix)

    print('\n> Population sum constraints:')
    pop_sum_constraints = get_pop_sum_constraints(orig_to_x, G)

    print('\n> Net flux constraints:')
    net_flux_constraints = get_net_flux_constraints(orig_to_x, flux_map_df,)

    print('\n> Anterograde-retrograde flux ratio constraints:')
    flux_ratio_constraints = get_flux_ratio_constraints(orig_to_x, flux_map_df, 2, 0.2)

    print('\n> Anterograde-retrograde population pair size constraints:')
    pop_ratio_constraints = get_pop_ratio_constraints(orig_to_x, G, 2, 0.5)


    mass_cons_cons = {'type': 'eq', 'fun': mass_conserve_constraint}
    pop_sum_cons = {'type': 'eq', 'fun': pop_sum_constraints}
    net_flux_cons = {'type': 'eq', 'fun': net_flux_constraints}
    #flux_ratio_cons = {'type': 'ineq', 'fun': flux_ratio_constraints}
    #pop_ratio_cons = {'type': 'ineq', 'fun': pop_ratio_constraints}
    cons = [
        mass_cons_cons, 
        pop_sum_cons, 
        net_flux_cons, 
        #flux_ratio_cons, 
        #pop_ratio_cons
            ]

    bnds = get_minimizer_bounds(orig_to_x)
    strt = get_minimizer_startval(orig_to_x)
    total_flux = get_objective_function(orig_to_x)


    print('\n> Optimizing flux values...')
    solution = minimize(
        method='SLSQP',
        fun = total_flux, 
        x0 = strt, 
        bounds = bnds, 
        constraints=cons,
        options={'disp':True, 'maxiter':5000},
        #tol = 0.0001,
        )

    results = np.round(solution.x, 4)

    print('\n> Solved flux dataframe:')
    flux_map_df = update_flux_map_df(G, flux_map_df, results, orig_to_x)
    print(flux_map_df)

    print('\n> Solved net flux matrix:')
    flux_matrix = update_flux_matrix(flux_matrix, flux_map_df)
    mat_print(np.round(flux_matrix, 2))

    flux_dict = {(row['u'], row['v']):row['flux_rate'] for i, row in flux_map_df.iterrows()} 

    return flux_dict