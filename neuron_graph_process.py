from copy import deepcopy

from neuron_graph_bioparam import calculate_target_nss, calculate_influx_efflux, add_bioparam_attributes
from neuron_graph_transform import neuron_graph_transform
from neuron_graph_flow_solve import solve_neuron_fluxes
from plot_module import plot_neuron_graph_subset

def neuron_graph_process(
        input_graph, 
        input_bio_param, 
        transform_type,
        n_soma_nodes, 
        soma_g_type, 
        prnt, 
        plot
        ):
    
    bio_param = deepcopy(input_bio_param)

    # calculate the target nss values, based on the mitochondrial densities, volumes, and node counts of each respective section
    bio_param = calculate_target_nss(input_graph, bio_param)
    # calculate the target net influx and efflux at axon and soma terminals
    bio_param = calculate_influx_efflux(input_graph, bio_param)

    # transform the graph 
    G, G_subgraphs = neuron_graph_transform(
        input_graph, 
        transform_type = transform_type, 
        n_soma_nodes   = n_soma_nodes, 
        soma_g_type    = soma_g_type
        )

    # add biological parameters
    G = add_bioparam_attributes(G, bio_param)

    # print('\n> Edges in the transformed graph:')
    # for u, v, data in G.edges(data = True): print(f'{u}-{v}: {data}')
    # print('\n> Nodes in the transformed graph:')
    # for node, data in G.nodes(data = True): print(f'{node}: {data}')

    if plot: plot_neuron_graph_subset(G)

    # calculate and process the fluxes
    G, bio_param = solve_neuron_fluxes(G, G_subgraphs, bio_param, prnt)

    return G