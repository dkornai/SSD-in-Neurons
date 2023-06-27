import networkx as nx
from sim_param_from_network import names_from_network

def sim_helpers_from_network(
        network:        nx.DiGraph, 
        start_pop:      list[int, int],
        ) ->            tuple[list[str], list[str], list[int]]:

    # generate names of variables and node
    vars, node_labels = names_from_network(network)

    # generate starting state of the system variables
    start_state = []
    for i in range(len(network.nodes())):
        start_state.append(start_pop[0]);start_state.append(start_pop[1])
    

    return vars, node_labels, start_state


# generate the circular network corresponding to the soma
def net_gen_hub_ring(
        n_nodes:        int,
        bio_param:      dict,
        ) ->            nx.DiGraph:
    
    # make a circular graph, and transform it to a directed graph
    soma_g = nx.cycle_graph(n_nodes)
    if n_nodes == 1: soma_g.remove_edge(0, 0) # remove self loop if only one node is present
    soma_g = soma_g.to_directed()

    # set node attributes
    nx.set_node_attributes(soma_g, 2,                               "birth_type")
    nx.set_node_attributes(soma_g, float(bio_param['soma_cb']),     "c_b")
    nx.set_node_attributes(soma_g, float(bio_param['soma_br']),     "birth_rate")
    nx.set_node_attributes(soma_g, float(bio_param['death_rate']),  "death_rate")
    nx.set_node_attributes(soma_g, float(bio_param['soma_nss']),    "nss")
    nx.set_node_attributes(soma_g, float(bio_param['delta']),       "delta")
    
    # add diffusion between soma nodes
    nx.set_edge_attributes(soma_g, float(bio_param['soma_diffusion']), "rate")
    
    # relable nodes as to indicate they are part of the soma
    rename_dict = {i:f'S{i}' for i in range(n_nodes)}
    nx.relabel_nodes(soma_g, rename_dict, copy=False)


    return soma_g




# add a chain of nodes to an existing graph
def net_gen_line_chain(
        n_nodes:        int,
        bio_param:      dict,
        line_type:     str,
        attach_graph:   nx.DiGraph,  # graph to which the chain should be attached
        attach_node:    str,         # name of the node to which the chain should be attached
        )->             nx.DiGraph:

    assert n_nodes >= 1, 'chain must at least 1 node'
    assert float(bio_param['gamma_ant']) > 0, "anterograde hopping rate must be > 0"

    # add new nodes and edges    
    for i in range(n_nodes):
    
        prev_node_name = f"{line_type}{i-1}"
        curr_node_name = f"{line_type}{i}"

        attach_graph.add_node(
            curr_node_name, 
            birth_type=0, 
            nss=-1, 
            death_rate = float(bio_param['death_rate'])
            )
        
        # add connections to other nodes
        if i == 0: prev_node_name = attach_node # the precursor of the first node is on the existing graph

        attach_graph.add_edge(prev_node_name, curr_node_name, rate=float(bio_param['gamma_ant']))
        if float(bio_param['gamma_ret']) > 0:
            attach_graph.add_edge(curr_node_name, prev_node_name, rate=float(bio_param['gamma_ret']))


    return attach_graph

# add a ring of nodes to an existing graph
def net_gen_line_ring(
        n_nodes:        int,
        bio_param:      dict,
        line_type:      str,
        attach_graph:   nx.DiGraph,  # graph to which the chain should be attached
        attach_node:    str,         # name of the node to which the chain should be attached
        )->             nx.DiGraph:
    
    assert n_nodes >= 2, 'ring must have 2 or more nodes'
    assert float(bio_param['gamma_ant']) > 0, "hopping rate of ring must be > 0"

    # add new nodes and edges    
    for i in range(n_nodes):
    
        prev_node_name = f"{line_type}{i-1}"
        curr_node_name = f"{line_type}{i}"

        attach_graph.add_node(
            curr_node_name, 
            birth_type=0, 
            nss=-1, 
            death_rate = float(bio_param['death_rate'])
            )
        
        # add connections to other nodes
        if i == 0: prev_node_name = attach_node # the precursor of the first node is on the existing graph

        attach_graph.add_edge(prev_node_name, curr_node_name, rate=float(bio_param['gamma_ant']))
    
    # attach final node to complete the circle
    attach_graph.add_edge(curr_node_name, attach_node, rate=float(bio_param['gamma_ant']))


    return attach_graph

   

# add a powerlaw tree to an existing graph
def net_gen_line_powlawtree(
        n_nodes:        int,
        bio_param:      dict,
        line_type:      str,
        attach_graph:   nx.DiGraph,  # graph to which the chain should be attached
        attach_node:    str,         # name of the node to which the chain should be attached
        )->             nx.DiGraph:
    
    assert n_nodes >= 1, 'powerlaw tree must at least 1 node'

    # extract the anterograde and retrograde hopping rates
    # gamma_ant = float(bio_param['gamma_ant'])
    # gamma_ret = float(bio_param['gamma_ret'])
    assert float(bio_param['gamma_ant']) > 0, "anterograde hopping rate must be > 0"

    # generate starting point of network
    G = nx.random_powerlaw_tree(n_nodes, gamma = 5, tries =100000, seed = 123)
    
    # convert to directed
    H = G.to_directed()
    out_edge = [e for e in H.edges() if e[0]<e[1]]
    in_edge = [e for e in H.edges() if e[0]>e[1]]

    # set attributes for all nodes
    nx.set_node_attributes(H, 0, "birth_type")
    nx.set_node_attributes(H, float(bio_param['death_rate']), "death_rate")
    nx.set_node_attributes(H, -1, "nss")
    

    # set hopping rates for anterograde edges
    nx.set_edge_attributes(H, {e:float(bio_param['gamma_ant']) for e in out_edge}, "rate")
    
    # set hopping rates for retrograde edges (or delete if retrograde rate is 0)
    if float(bio_param['gamma_ret']) == 0:
        for e in in_edge:
            H.remove_edge(e[0], e[1])
    else:
        nx.set_edge_attributes(H, {e:float(bio_param['gamma_ret']) for e in in_edge}, "rate")
    

    # rename nodes for compatibility with other functions
    rename_dict = {i:f'{line_type}{i}' for i in range(n_nodes)}
    nx.relabel_nodes(H, rename_dict, copy=False)
    
    # combine the graphs, and add the connecting edges
    out_graph = nx.DiGraph()
    out_graph.add_edges_from(list(attach_graph.edges(data=True))+list(H.edges(data=True)))
    out_graph.add_nodes_from(list(attach_graph.nodes(data=True))+list(H.nodes(data=True)))
    out_graph.add_edge(list(H.nodes())[0], attach_node, rate=float(bio_param['gamma_ant']))
    if float(bio_param['gamma_ret']) > 0: out_graph.add_edge(attach_node, list(H.nodes())[0], rate=float(bio_param['gamma_ret']))


    return out_graph