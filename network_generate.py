import networkx as nx
from sim_param_from_network import names_from_network

# generate a chain of nodes
def network_gen_chain(
        n_nodes     :int, 
        start_pop   :list[int, int], 
        bio_param   :dict
        ):

    assert n_nodes > 1, 'chain must have 2+ nodes'

    # extract the anterograde and retrograde hopping rates
    hr_ant = float(bio_param['gamma_ant'])
    hr_ret = float(bio_param['gamma_ret'])
    assert hr_ant > 0, "anterograde hopping rate must be > 0"

    
    G = nx.DiGraph()
    # add main starer node
    G.add_node('N0', birth_type=2, c_b = float(bio_param['soma_cb']), birth_rate = float(bio_param['soma_br']), nss=float(bio_param['soma_nss']), delta = float(bio_param['delta']), death_rate = float(bio_param['turnover']),)

    # add remaining nodes:
    for i in range(1, n_nodes):
        prev_node_name = f"N{i-1}"
        curr_node_name = f"N{i}"

        G.add_node(curr_node_name, birth_type=0, nss=-1, death_rate = float(bio_param['turnover']),)
        G.add_edge(prev_node_name, curr_node_name, rate=hr_ant)
        if hr_ret > 0:
            G.add_edge(curr_node_name, prev_node_name, rate=hr_ret)

    # extract variable names, and compartment (node) names
    vars, comp = names_from_network(G)

    # set a given starting state
    start_state = []
    for i in range(n_nodes):
        start_state.append(start_pop[0]);start_state.append(start_pop[1])

    return G, vars, comp, start_state

# generate a ring of nodes
def network_gen_ring(
        n_nodes     :int, 
        start_pop   :list[int, int], 
        bio_param   :dict
        ):
    
    assert n_nodes > 2, 'ring must have 3+ nodes'

    # extract the anterograde and retrograde hopping rates
    hr_ant = float(bio_param['gamma_ant'])
    hr_ret = float(bio_param['gamma_ret'])
    assert hr_ant > 0, "anterograde hopping rate must be > 0"

    G = nx.DiGraph()
    # add main starer node
    G.add_node('N0', birth_type=2, c_b = float(bio_param['soma_cb']), birth_rate = float(bio_param['soma_br']), nss=float(bio_param['soma_nss']), delta = float(bio_param['delta']), death_rate = float(bio_param['turnover']),)

    # add remaining nodes:
    for i in range(1, n_nodes):
        prev_node_name = f"N{i-1}"
        curr_node_name = f"N{i}"

        G.add_node(curr_node_name, birth_type=0, nss=-1, death_rate = float(bio_param['turnover']),)
        G.add_edge(prev_node_name, curr_node_name, rate=hr_ant)
        if hr_ret > 0:
            G.add_edge(curr_node_name, prev_node_name, rate=hr_ret)

    # add edges connecting last node to N0
    G.add_edge(f"N{n_nodes-1}", 'N0', rate=hr_ant)
    if hr_ret > 0:
        G.add_edge('N0', f"N{n_nodes-1}", rate=hr_ret)


    # extract variable names, and compartment (node) names
    vars, comp = names_from_network(G)

    # set a given starting state
    start_state = []
    for i in range(n_nodes):
        start_state.append(start_pop[0]);start_state.append(start_pop[1])

    return G, vars, comp, start_state     

# generate a small tree
def network_gen_powlaw_tree(
        n_nodes     :int, 
        start_pop   :list[int, int], 
        bio_param   :dict
        ):
    
    assert n_nodes > 1, 'powerlaw tree must have 2+ nodes'

    # extract the anterograde and retrograde hopping rates
    hr_ant = float(bio_param['gamma_ant'])
    hr_ret = float(bio_param['gamma_ret'])
    assert hr_ant > 0, "anterograde hopping rate must be > 0"

    # generate starting point of network
    G = nx.random_powerlaw_tree(n_nodes, gamma = 5, tries =100000, seed = 123)
    
    # convert to directed
    H = G.to_directed()
    out_edge = [e for e in H.edges() if e[0]<e[1]]
    in_edge = [e for e in H.edges() if e[0]>e[1]]

    # set attributes for all nodes
    nx.set_node_attributes(H, 0, "birth_type")
    nx.set_node_attributes(H, float(bio_param['turnover']), "death_rate")
    nx.set_node_attributes(H, -1, "nss")

    # change attributes for node 0
    nx.set_node_attributes(H, {0:2}, "birth_type")
    nx.set_node_attributes(H, {0:float(bio_param['soma_cb'])}, "c_b")
    nx.set_node_attributes(H, {0:float(bio_param['soma_br'])}, "birth_rate")
    nx.set_node_attributes(H, {0:float(bio_param['soma_nss'])}, "nss")
    nx.set_node_attributes(H, {0:float(bio_param['delta'])}, "delta")
    
    # set hopping rates
    nx.set_edge_attributes(H, {e:hr_ant for e in out_edge}, "rate")
    if hr_ret == 0:
        for e in in_edge:
            H.remove_edge(e[0], e[1])

    else:
        nx.set_edge_attributes(H, {e:hr_ret for e in in_edge}, "rate")
    
    # rename nodes for compatibility with other functions
    rename_dict = {i:f'N{i}' for i in range(n_nodes)}
    nx.relabel_nodes(H, rename_dict, copy=False)
    

    # extract variable names, and compartment (node) names
    vars, comp = names_from_network(H)

    # set a given starting state
    start_state = []
    for i in range(n_nodes):
        start_state.append(start_pop[0]);start_state.append(start_pop[1])

    return H, vars, comp, start_state