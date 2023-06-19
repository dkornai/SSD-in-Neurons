import networkx as nx
from sim_param_from_network import names_from_network

def linear_network(n_nodes, bio_param, start_pop):

    G = nx.DiGraph()

    # add main starer node
    G.add_node('N0', birth_type=2, c_b = float(bio_param['c_b']), birth_rate = float(bio_param['mu_a']), nss=float(bio_param['nss']), delta = float(bio_param['delta']), death_rate = float(bio_param['mu']),)

    # add remaining nodes:
    for i in range(1, n_nodes):
        prev_node_name = f"N{i-1}"
        curr_node_name = f"N{i}"

        G.add_node(curr_node_name, birth_type=0, nss=-1, death_rate = float(bio_param['mu']),)
        G.add_edge(prev_node_name, curr_node_name, rate=float(bio_param['gamma_ant']))
        G.add_edge(curr_node_name, prev_node_name, rate=float(bio_param['gamma_ret']))

    vars, comp = names_from_network(G)

    # set a given starting state
    start_state = []
    for i in range(n_nodes):
        start_state.append(start_pop[0]);start_state.append(start_pop[1])

    return G, vars, comp, start_state