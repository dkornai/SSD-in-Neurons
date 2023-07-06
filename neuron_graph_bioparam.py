import networkx as nx
import numpy as np
from neuron_graph_helper import n_nodes_by_type, volume_by_type, total_incoming_edge_length

# infer the nss values for the soma, axon and dendrite by considering the volume, density, and number of nodes in the graph
def calculate_target_nss(
        G:          nx.Graph, 
        bio_param:  dict
        ) ->        dict:
    
    assert type(G) == nx.Graph, 'target nss values can only be calculated for raw undirected neural graphs'

    print('\n> Inferring target node populations per branch type from section volumes.')
    
    n_nodes_per_type = n_nodes_by_type(G)
    volume_per_type = volume_by_type(G)

    for nss_key, density_key, n_nodes, volume in zip(
        ['soma_nss', 'axon_nss', 'dendrite_nss'], 
        ['soma_mito_density', 'axon_mito_density', 'dendrite_mito_density'], 
        n_nodes_per_type, 
        volume_per_type
        ):

        bio_param[nss_key] = int(round((volume*bio_param[density_key])/n_nodes))
        print(f"total {nss_key[:-4]} volume is {volume} µm^3 spread across {n_nodes} node(s). At a density of {bio_param[density_key]} mt/µm^3, this gives a target pop. of {bio_param[nss_key]} mt/node")

    return bio_param

# calculate the required influx and efflux of mt to maintain target population levels at axon and dendrite terminals
def calculate_influx_efflux(
        G:          nx.Graph,
        bio_param:  dict
        ) ->        dict:
    
    assert type(G) == nx.Graph, 'target influx and efflux values can only be calculated for raw undirected neural graphs'
    assert bio_param['axon_nss'] != None and bio_param['dendrite_nss'] != None, 'target influx and efflux values can only be calculated after target nss values have been inferred'

    print('\n> Inferring net influx and efflux at at axon and dendrite terminals.')
    
    for influx_key, efflux_key, nss_key, death_rate_key in zip(
        ['axon_terminal_influx', 'dendrite_terminal_influx'],
        ['axon_terminal_efflux', 'dendrite_terminal_efflux'],
        ['axon_nss', 'dendrite_nss'],
        ['axon_death_rate', 'dendrite_death_rate']
        ):

        bio_param[influx_key] = round(bio_param[nss_key]*bio_param[death_rate_key]*2,2)
        bio_param[efflux_key] = round(bio_param[nss_key]*bio_param[death_rate_key],2)
        print(f"at a death rate of {bio_param[death_rate_key]}, a target {nss_key[:-4]} pop. of {bio_param[nss_key]}, requires an influx of {bio_param[influx_key]} mt and an outflow of {bio_param[efflux_key]} mt")

    return bio_param



# get the death rate for axon or dendrite nodes. For terminal nodes, this is the basal value. For intermediate nodes, it is adjusted down by travel time.
def calculate_death_rate(
            G:          nx.DiGraph, 
            bio_param:  dict, 
            node:       str, 
            data:       dict
            ) ->        float:
    
    if data['nodetype'] == 2: prefix = 'axon'
    elif data['nodetype'] == 3: prefix = 'dendrite'

    # for intermediate nodes, adjust death rate
    if data['terminal'] == False:
        # get the sum of incoming edge lengths
        total_in_edge_len = total_incoming_edge_length(G, node)
        # at the given transport speed, this distance would be covered in:
        total_travel_time = total_in_edge_len/bio_param[f'{prefix}_transp_speed']
        # during this time, the death rate would be:
        death_rate = round(total_travel_time*bio_param[f'{prefix}_death_rate'], 6)
    
    # for terminal nodes, use the basal specific death rate
    death_rate = bio_param[f'{prefix}_death_rate']
    
    return death_rate

# adjust the soma nss downwards if multiple node in the model now represent a single node in the soma
def adjust_soma_nss(
        G:          nx.DiGraph, 
        bio_param:  dict
        ) ->        dict:

    n_soma_nodes = n_nodes_by_type(G)[0]
    if n_soma_nodes != 1:
        print(f"> Adjusting soma nss to {int(round(bio_param['soma_nss']/n_soma_nodes))} in order to spread total population of {bio_param['soma_nss']} across {n_soma_nodes} nodes.")
        bio_param['soma_nss'] = int(round(bio_param['soma_nss']/n_soma_nodes))

    return bio_param


# add the biological parameters required to run simulations to the directed transformed graph
def add_bioparam_attributes(
        G:          nx.DiGraph, 
        bio_param:  dict,
        ) ->        nx.DiGraph:
    
    print('\n>> Adding biological parameters to the network')

    assert type(G) == nx.DiGraph, 'biological parameters can only be added to directed graphs outputted by "neuron_graph_transform", not undirected graphs.'
    for key in bio_param:
        assert bio_param[key] != None, f'biological parameter "{key}" not provided!'

    # adjust soma nss to account for multiple nodes if needed
    bio_param = adjust_soma_nss(G, bio_param)

    # iterate through the nodes
    for node, data in G.nodes(data = True):
        # if the node is in the soma
        if data['nodetype'] == 1:
            nx.set_node_attributes(
                G, 
                {node:{
                    'birth_type':   2,
                    'c_b':          float(bio_param['soma_cb']),
                    'birth_rate':   float(bio_param['soma_br']),
                    'death_rate':   float(bio_param['soma_death_rate']),
                    'nss':          float(bio_param['soma_nss']),
                    'delta':        float(bio_param['delta']),
                    }
                }
            )
        
        # if the node is in an axon
        elif data['nodetype'] == 2:
            nx.set_node_attributes(
                G, 
                {node:{
                    'birth_type':   0, # no births
                    'death_rate':   calculate_death_rate(G, bio_param, node, data),
                    'nss':          float(bio_param['axon_nss']),
                    }
                }
            )
        
        # if the node is in a dendrite
        elif    data['nodetype'] == 3:
            nx.set_node_attributes(
                G, 
                {node:{
                    'birth_type':   0, # no births
                    'death_rate':   calculate_death_rate(G, bio_param, node, data),
                    'nss':          float(bio_param['dendrite_nss']),
                    }
                }
            )

    # iterate through the edges
    for u, v, data in G.edges(data = True):
        edge_type = data['edgetype']
        dest_data = G.nodes(data = True)[v]
        src_data  = G.nodes(data = True)[u]

        # if the edge is in the soma
        if   edge_type == 1:
            nx.set_edge_attributes(
                G, 
                {(u,v):{
                    'rate':         float(bio_param['soma_diffusion']),
                    }
                }
            )

        # if the edge is in the soma
        if   edge_type == 2:
            if dest_data.get('terminal', False) == True:
                nx.set_edge_attributes(
                    G, 
                    {(u,v):{
                        'net_flux':         float(bio_param['axon_terminal_influx']),
                        }
                    }
                )
            elif src_data.get('terminal', False) == True:
                nx.set_edge_attributes(
                    G, 
                    {(u,v):{
                        'net_flux':         float(bio_param['axon_terminal_efflux']),
                        }
                    }
                )

        # if the edge is in the soma
        if   edge_type == 3:
            if dest_data.get('terminal', False) == True:
                nx.set_edge_attributes(
                    G, 
                    {(u,v):{
                        'net_flux':         float(bio_param['dendrite_terminal_influx']),
                        }
                    }
                )
            elif src_data.get('terminal', False) == True:
                nx.set_edge_attributes(
                    G, 
                    {(u,v):{
                        'net_flux':         float(bio_param['dendrite_terminal_efflux']),
                        }
                    }
                )

        # if the edge is a direction reversing edge in the ladder model
        elif edge_type == 5:
            if data['direction'] == 'ar': 
                rate = float(bio_param['switch_rate_ar'])
            elif data['direction'] == 'ra': 
                rate = float(bio_param['switch_rate_ra'])
            # set the rate to the anterior or retrograde rate, depending on the edge direction
            nx.set_edge_attributes(
                G, 
                {(u,v):{
                    'rate':         rate
                    }
                }
            )

    return G