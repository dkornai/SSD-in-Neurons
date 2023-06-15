
import networkx as nx
import pandas as pd
import numpy as np; np.set_printoptions(suppress=True)

def dataframes_from_network(G, prnt = True):
    
    # Extract node attributes into a dictionary
    node_attributes = {node: G.nodes[node] for node in G.nodes()}

    # Create a dataframe from the node attributes dictionary
    df = pd.DataFrame.from_dict(node_attributes, orient='index')

    # Add a separate column for the node index
    df.index.name = 'Node'
    df = df.reset_index()


    # Extract edge attributes into a list of dictionaries
    edge_attributes = [{'source': edge[0], 'target': edge[1], 'rate': G.edges[edge]['rate']} for edge in G.edges()]

    # Create a dataframe from the edge attributes list
    df_edges = pd.DataFrame(edge_attributes)

    if prnt == True:
        print('> Nodes:')
        print(df)
        print('\n> Edges:')
        print(df_edges)

    return df, df_edges

# get node names, and component names (node + wt and mt) from the network
def names_from_network(G):
    df, df_edge = dataframes_from_network(G, prnt = False)
    comp = df["Node"].to_list()
    vars = []
    for node in comp:
        vars.append(f'{node}_wt')
        vars.append(f'{node}_mt')
    
    return vars, comp

def empty_react(n_pops):
    return [0]*n_pops

def birth_react(n_pops, birth_i):
    r = empty_react(n_pops)
    r[birth_i] = 1
    return r

def death_react(n_pops, death_i):
    r = empty_react(n_pops)
    r[death_i] = -1
    return r

def transp_react(n_pops, source_i, dest_i):
    r = empty_react(n_pops)
    r[source_i] = -1
    r[dest_i] = 1
    return r


#### GENERATE THE HELPER DATASTRUCTURES FOR THE C MODULE THAT DOES THE GILLESPIE SIMULATION ####
def gillespie_reaction_dict_from_network(G):
    df, df_edges = dataframes_from_network(G, prnt=False)
    
    node_names_list = list(G.nodes())
    
    n_compartments = G.number_of_nodes()
    n_populations = n_compartments*2


    n_reactions = 0
    reactions = []
    reaction_rates = []
    state_index = []

    n_rate_update_b = 0
    birthrate_updates_reaction = [] # indeces of nodes where the birth rates must be updated each iteration
    birthrate_updates_par = [] # parameters (c_b, birth_rate, nss, delta) needed to update birth rates
    birthrate_state_index = []


    ## BIRTH REACTIONS
    for index, row in df.iterrows():
        if row['birth_type'] != 0:
            for index_offset in range(2):
                n_reactions += 1
                
                ## ADD TO REACTIONS
                reactions.append(birth_react(n_populations, index*2+index_offset))
                
                ## SPECIFY RATE
                # birth type 2 is dynamic
                if row['birth_type'] == 2:
                    reaction_rates.append(-1)
                # birth type 1 is constant
                else:
                    reaction_rates.append(row['birth_rate'])
                
                ## SPECIFT INDEX OF NODE
                state_index.append(index*2+index_offset)
                
        ## ADD TO RATE UPDATES
        if row['birth_type'] == 2:
            n_rate_update_b += 1
            birthrate_updates_par.append([row['c_b'], row['birth_rate'], row['nss'], row['delta']])
            birthrate_updates_reaction.append(n_reactions - 2)
            birthrate_state_index.append(index*2)
                
    ## DEATH REACTIONS
    for index, row in df.iterrows():
        for index_offset in range(2):
            n_reactions += 1
            
            ## ADD TO REACTIONS
            reactions.append(death_react(n_populations, index*2+index_offset))

            ## SPECIFY RATE
            reaction_rates.append(row['death_rate'])

            ## SPECIFT INDEX OF NODE
            state_index.append(index*2+index_offset)

    ## TRANSPORT REACTIONS
    for index, row in df_edges.iterrows():
        for index_offset in range(2):
            n_reactions += 1
            
            src_i = node_names_list.index(row['source'])*2 + index_offset
            dst_i = node_names_list.index(row['target'])*2 + index_offset
            
            ## ADD TO REACTIONS
            reactions.append(transp_react(n_populations, src_i, dst_i))

            ## SPECIFY RATE
            reaction_rates.append(row['rate'])

            ## SPECIFT INDEX OF NODE
            state_index.append(src_i)

    # printout of results
    print("> Reactions:")
    print("reaction #\tupdate to system\trate\tstate index")
    for i in range(n_reactions): print(f'{i}\t\t{reactions[i]}\t{reaction_rates[i]}\t{state_index[i]}')

    print("\n> Dynamic birth rates:")
    print("reaction #s\tpar(c_b, mu, nss, delta)\tstate index")
    for i in range(n_rate_update_b): print(f'{birthrate_updates_reaction[i]}, {birthrate_updates_reaction[i]+1}\t\t{birthrate_updates_par[i]}\t\t{birthrate_state_index[i]}')

    # format conversion
    reactions = np.array(reactions, dtype = np.int64)
    reaction_rates = np.array(reaction_rates, dtype = np.float64)
    state_index = np.array(state_index, dtype = np.int64)

    birthrate_updates_par = np.array(birthrate_updates_par, dtype = np.float64)
    birthrate_updates_reaction = np.array(birthrate_updates_reaction, dtype = np.int64)
    birthrate_state_index = np.array(birthrate_state_index, dtype = np.int64)

    # final output               
    gillespie_reactions = {'n_reactions':n_reactions,
                        'reactions':reactions, 
                        'reaction_rates':reaction_rates, 
                        'state_index':state_index}
                                    
    birthrate_update    = {'n_rate_update_birth':n_rate_update_b,
                        'rate_update_birth_par':birthrate_updates_par, 
                        'rate_update_birth_reaction':birthrate_updates_reaction,
                        'birthrate_state_index':birthrate_state_index}

    return {'gillespie': gillespie_reactions, 'update_rate_birth':birthrate_update}

#### GENERATE CODE FOR THE ODE MODEL OF THE NETWORK ####
'''
This is a very very very extremely hacky function that works by collecting the parameters
of the differential equations from the parameters encoded in the network, uses this to make
a text python program, and then evaluates this python program and returns the output. 
'''
def ODE_from_network(G):
    # collect data needed to generate the code
    nodenames, compartments = names_from_network(G)
    df, df_edges = dataframes_from_network(G, prnt=False) 

    # names of the variables (node names + wt and mt)
    vars = []
    for node in compartments:
        vars.append(f'{node}_wt')
        vars.append(f'{node}_mt')

    # for each variable, get the expression for how much it changes in a given time t
    diff_exp = ""    
    for var in vars: 
        node_type = var[-2:]
        node_name = var[:-3]
        var_wt_name = f'{node_name}_wt'
        var_mt_name = f'{node_name}_mt'
        
        # get corresponding node row in row dataframe
        noderow = df.loc[df['Node'] == node_name]
        
        # collect numeric values of specific parameters
        cb = float(noderow['c_b'])
        birthrate = float(noderow['birth_rate'])
        nss = int(noderow['nss'])
        delta = float(noderow['delta'])
        deathrate = float(noderow['death_rate'])

        # generate birth rate
        if   int(noderow["birth_type"]) == 0:
            birth_term = '0'
        elif int(noderow["birth_type"]) == 1:
            birth_term = f"({birthrate})"
        elif int(noderow["birth_type"]) == 2:
            birth_term = f"({birthrate} + {cb}*({nss}-{var_wt_name}-({delta}*{var_mt_name})))"
            
        # generate death rate
        death_term = f"({deathrate})"
        
        # generate terms for outflow
        edges_outrow = df_edges[df_edges['source'] == node_name]
        out_term = f"({round(edges_outrow['rate'].sum(), 8)})"
        
        # generate terms for inflow
        edges_inrow = df_edges[df_edges['target'] == node_name]
        in_term = ''
        for index, row in edges_inrow.iterrows():
            in_term += f"+({str(row['source'])}_{node_type}*{float(row['rate'])})"
        
        diff_exp += f"\t\t({var}*({birth_term}-{death_term}-{out_term})){in_term}, # Δ{var}\Δt\n"

    # set up function
    fulltext = "global ODE_model\ndef ODE_model(t,z):\n"

    # set up tuple of variables
    vars_for_code = str((str(vars)[1:-1]).replace("'",""))
    fulltext += f'\t{vars_for_code} = z'
    
    # set up how each variable changes
    fulltext += f"\n\treturn [\n{diff_exp}\t\t]"
    
    print("> Code for ODE model:\n")
    print(fulltext)

    # execute the code generated by the process
    global ODE_model
    ODE_model = None
    exec(fulltext)

    return ODE_model