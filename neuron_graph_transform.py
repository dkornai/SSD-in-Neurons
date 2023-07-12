'''
TAKE AN UNDIRECTED NEURON GRAPH, WITH MARKED SOMA, DENDRITES, AND AXONS, AND TRANSFORM IT INTO A DIRECTED NETWORK, SUCH THAT:
1) THE SOMA IS REPLACED WITH A RING WITH BIDIRECTIONAL TRANSPORT
2) AXONS AND DENDRITES ARE REPLACED WITH LADDERIZED, CIRCULARIZED, OR BIDIERTIONALIZED VERSIONS
'''

from copy import deepcopy
import networkx as nx
import numpy as np
from network_generate import net_gen_hub_ring, net_gen_hub_complete
from neuron_graph_helper import infer_graph_type, get_edges_in_each_branch, get_nodes_in_each_branch

nodetype_dict = {1:'soma', 2:'axon',3:'dendrite'}
nodetype_shortdict = {1:'S', 2:'A',3:'D'}

# isolate a tree subgraph of an undirected graph, and return a directed subtree with edges away from the root node
def make_subgraph_tree_directed(
        G:              nx.Graph, 
        subgraph_nodes: list[str], 
        root:           str
        ) ->            nx.DiGraph:

    # find leaf nodes in the source graph
    leaf_nodes = [x for x in G.nodes() if (G.degree(x)==1 and x != root)]

    D = nx.DiGraph()

    # Add nodes and their attributes to the directed graph
    D.add_nodes_from((n, d) for n, d in G.nodes(data=True) if n in subgraph_nodes)

    # mark leaf nodes
    # if the subgraph has only one node, it is terminal
    if nx.number_of_nodes(D) == 1:
        nx.set_node_attributes(D, True,'terminal')
    # otherwise the leaves are terminal
    else:    
        for node in D.nodes():
            if node in leaf_nodes:
                nx.set_node_attributes(D, {node:True},'terminal')
            else:
                nx.set_node_attributes(D, {node:False},'terminal')

    # Add edges and their attributes to the directed graph using BFS
    for u, v, d in G.edges(data=True):
        if u in subgraph_nodes and v in subgraph_nodes:
            if nx.has_path(G, root, v):
                D.add_edge(u, v, **d)
            else:
                D.add_edge(v, u, **d)

    return D

# ladderize a directed graph
def ladderize(
        G:          nx.DiGraph,
        root_node:  str,
        ) ->        tuple[nx.DiGraph, list[str]]:
    
    # infer graph type
    gtype_i, gtype_c = infer_graph_type(G)

    L = nx.DiGraph()

    # if the graph has only one node
    if len(G.nodes) == 1:
        # add a single terminal node, and 
        node_data = G.nodes[root_node]
        L.add_node(f"{gtype_c}{root_node}T", **node_data)
        # set the nodes to connect accordingly
        nodes_to_connect = [f"{gtype_c}{root_node}T"]
    
    else:
        for node, data in G.nodes(data=True):
            node_is_terminal = data.get('terminal', False)
            # For each node, add a forward and a reverse node to the ladder graph, unless it's terminal
            if not node_is_terminal:
                L.add_node(f"{gtype_c}{node}F", **data)
                L.add_node(f"{gtype_c}{node}R", **data)
            else:
                L.add_node(f"{gtype_c}{node}T", **data)

        for u, v, data in G.edges(data=True):
            v_is_terminal = G.nodes[v].get('terminal')
            # add the edges
            if v_is_terminal: # if the node is terminal
                L.add_edge(f"{gtype_c}{u}F", f"{gtype_c}{v}T", **data, direction = 'forward')
                L.add_edge(f"{gtype_c}{v}T", f"{gtype_c}{u}R", **data, direction = 'reverse')
            elif not v_is_terminal:
                L.add_edge(f"{gtype_c}{u}F", f"{gtype_c}{v}F", **data, direction = 'forward')
                L.add_edge(f"{gtype_c}{v}R", f"{gtype_c}{u}R", **data, direction = 'reverse')
    
        for node in G.nodes():
            is_terminal = G.nodes[node].get('terminal', False)
            # For each node, add an edge between the forward and the reverse node in the ladder graph unless the node is terminal
            if not is_terminal:
                L.add_edge(f"{gtype_c}{node}F", f"{gtype_c}{node}R", edgetype = 5, direction = 'ar')
                L.add_edge(f"{gtype_c}{node}R", f"{gtype_c}{node}F", edgetype = 5, direction = 'ra')


        # set the nodes where the subgraph should be rejoined with the main graph
        nodes_to_connect = [f'{gtype_c}{root_node}F', f'{gtype_c}{root_node}R']

    # Now, L is the ladder graph of G
    return L, nodes_to_connect 


# circularize a directed graph
def circularize(
        G:          nx.DiGraph, 
        root_node:  str,
        ) ->        tuple[nx.DiGraph, list[str]]:
    
    # Function to get all root-to-leaf paths
    def get_paths(G, root_node):
        paths = []
        for leaf in (n for n, d in G.out_degree() if d==0):
            paths.extend(nx.all_simple_paths(G, root_node, leaf))
        return paths

    # infer graph type
    gtype_i, gtype_c = infer_graph_type(G)

    # Create an empty directed graph for the circular graph
    C = nx.DiGraph()
    
    # if the graph has only one node
    if len(G.nodes) == 1:
        # add a single terminal node, and 
        node_data = G.nodes[root_node]
        C.add_node(f"{gtype_c}{root_node}T", **node_data)
        # set the nodes to connect accordingly
        nodes_to_connect = [f"{gtype_c}{root_node}T"]
    
    # if the graph has more than one node, traverse the paths from the root 
    else:
        # Add root nodes with their attributes
        node_data = G.nodes[root_node]
        C.add_node(f"{gtype_c}{root_node}F", **node_data)
        C.add_node(f"{gtype_c}{root_node}R", **node_data)

        # Get all root-to-leaf paths
        paths = get_paths(G, root_node)
        
        # For each path
        for path in paths:
            for i in range(len(path)-1):
                edge_data = G.edges[path[i], path[i+1]]
                node_data = G.nodes[path[i+1]]
                
                node_is_terminal = node_data['terminal']
                # If it's not the terminal node, add forward and reverse strands
                if not node_is_terminal:
                    C.add_node(f"{gtype_c}{path[i+1]}F", **node_data)
                    C.add_node(f"{gtype_c}{path[i+1]}R", **node_data)
                    C.add_edge(f"{gtype_c}{path[i]}F", f"{gtype_c}{path[i+1]}F", **edge_data, direction = 'forward')
                    C.add_edge(f"{gtype_c}{path[i+1]}R", f"{gtype_c}{path[i]}R", **edge_data, direction = 'reverse')
                elif node_is_terminal:
                    # For terminal nodes, add a single node and connect forward and reverse strands
                    C.add_node(f"{gtype_c}{path[i+1]}T", **node_data)
                    C.add_edge(f"{gtype_c}{path[i]}F", f"{gtype_c}{path[i+1]}T", **edge_data, direction = 'forward')
                    C.add_edge(f"{gtype_c}{path[i+1]}T", f"{gtype_c}{path[i]}R", **edge_data, direction = 'reverse')

        # set the nodes where the subgraph should be rejoined with the main graph
        nodes_to_connect = [f'{gtype_c}{root_node}F', f'{gtype_c}{root_node}R']

    # Now, C is the circularized graph of G
    return C, nodes_to_connect

# bidirectionalize a directed graph
def bidirectionalize(
        G:      nx.DiGraph
        ) ->    tuple[nx.DiGraph, list[str]]:
    
    # infer graph type
    gtype_i, gtype_c = infer_graph_type(G)
    
    # Create an empty directed graph for the bidirectional graph
    B = nx.DiGraph()

    # Iterate over the nodes in the original graph
    for node, data in G.nodes(data=True):
        B.add_node(f'{gtype_c}{node}B', **data)

    # For each edge, add an edge and its reverse to the bidirectional graph
    for u, v, data in G.edges(data=True):
        B.add_edge(f'{gtype_c}{u}B', f'{gtype_c}{v}B', **data, direction = 'forward')
        B.add_edge(f'{gtype_c}{v}B', f'{gtype_c}{u}B', **data, direction = 'reverse')

    # set the nodes to connect to the first node in the graph
    nodes_to_connect = [f"{list(B.nodes())[0]}"]

    # Now, B is the bidirectional graph of G
    return B, nodes_to_connect


# ladderize, circularize, or bidirectionalize a subgraph (axon or dendrite) from the full neural graph
def subgraph_transform(
        full_graph:             nx.Graph, 
        subgraph_nodes:         list[str], 
        subgraph_root:          str,
        transform_type:         str
        ) ->                    tuple[nx.DiGraph, list]:
    
    # make subgraph directed away from root node
    dir_graph = make_subgraph_tree_directed(full_graph, subgraph_nodes, subgraph_root)

        # ladderize graph
    if   transform_type == 'ladder':
        out_graph, nodes_to_connect = ladderize(dir_graph, subgraph_root)

        # circularize graph
    elif transform_type == 'circle':
        out_graph, nodes_to_connect = circularize(dir_graph, subgraph_root)
        
        # bidirectionalise graph
    elif transform_type == 'bidirect':
        out_graph, nodes_to_connect = bidirectionalize(dir_graph)

    return out_graph, nodes_to_connect

# merge a transformed subgraph back into to the full graph
def subgraph_remerge(
        full_graph:             nx.DiGraph, 
        transformed_subgraph:   nx.DiGraph,
        soma_attach_node:       str, 
        subgraph_attach_nodes:  list[str],
        original_edge_len:      float,
        ) ->                    nx.DiGraph:
    
    # find the type (axon or dendrite) of the subgraph
    gtype_i, gtype_c = infer_graph_type(transformed_subgraph)
    
    # merge full graph and subgraph
    full_graph = nx.union(full_graph, transformed_subgraph)
    

    # ladderized and circularized graphs have their forward and reverse sides connected to the target soma node
    if   len(subgraph_attach_nodes) == 2:
        full_graph.add_edge(soma_attach_node, subgraph_attach_nodes[0],
                            edgetype = gtype_i, direction = 'forward', len = original_edge_len)
        full_graph.add_edge(subgraph_attach_nodes[1], soma_attach_node,
                            edgetype = gtype_i, direction = 'reverse', len = original_edge_len)
        
    # bidirectionailzed graphs connect their root to the target soma node
    elif len(subgraph_attach_nodes) == 1:
        full_graph.add_edge(soma_attach_node, subgraph_attach_nodes[0],
                            edgetype = gtype_i, direction = 'forward', len = original_edge_len)
        full_graph.add_edge(subgraph_attach_nodes[0], soma_attach_node,
                            edgetype = gtype_i, direction = 'reverse', len = original_edge_len)
        
    return full_graph




# isolate each arbor (dendrite or axon) in a basic graph
def find_subgraphs(G, root_node = '1'):
    
    subgraphs = []

    # get edges and nodes in each subgraph
    edges_in_each_subgraph = get_edges_in_each_branch(G, root_node=root_node)
    nodes_in_each_subgraph = get_nodes_in_each_branch(G, root_node=root_node)
    for edges, nodes in zip(edges_in_each_subgraph, nodes_in_each_subgraph):
        subgraphs.append({
            'type':G.nodes()[nodes[1]]['nodetype'],
            'type_name':nodetype_dict[G.nodes()[nodes[1]]['nodetype']],
            'total_volume':round(np.sum([data['volume'] for u, v, data in G.edges(data = True) if (u,v) in edges]), 2),
            'n_nodes':len(nodes[1:]),
            'branch_start_node':nodes[1],
            'branch_start_edge_len':G.edges()[edges[0]]['len'],
            'branch_nodes':nodes[1:], 
            'edges':edges
            })
    
    print(f'Found {len(subgraphs)} branches:')
    for subgraph in subgraphs:
        print(f"{subgraph['type_name']} of {subgraph['n_nodes']} nodes, total volume is {subgraph['total_volume']} um^3")
    
    return subgraphs



# take an undirected graph of a neuron, and transform it into a directed network
def neuron_graph_transform(
        input_G:                nx.Graph, 
        transform_type:         str, 
        n_soma_nodes:           int,
        soma_g_type:            str,
        ) ->                    tuple[nx.DiGraph, list[list[str]]]:

    assert transform_type in ['circle','ladder','bidirect'], 'transform_type must be "circle", "ladder", or "bidirect"'
    assert n_soma_nodes > 0, 'must have at least 1 node in the soma'
    assert soma_g_type in ['circle','complete'], 'soma is represented as a circle graph ("circle") or complete graph ("complete")'

    G = deepcopy(input_G) # deepcopy to avoid modifying the source graph
    print(f"\n>> Transforming input graph:\n> The undirected input graph has {len(list(G.nodes()))} nodes, and {len(list(G.edges()))} edges.", end = ' ')

    # find the axon and dendrite branches of the full graph
    subgraphs = find_subgraphs(G, root_node='1')
    n_subgraphs = len(subgraphs)

    # separate into subgraphs by removing all edges relating to the soma
    G.remove_edges_from(list(G.edges('1')))

    # make soma using the ring generator
    if soma_g_type == 'circle':
        neuron_g = net_gen_hub_ring(n_nodes=n_soma_nodes)
    elif soma_g_type == 'complete':
        neuron_g = net_gen_hub_complete(n_nodes=n_soma_nodes)
    
    # prepare to place each arbor such that they are maximally spread across the ring
    soma_nodes = list(neuron_g.nodes())
    if n_soma_nodes > 1:
        soma_attach_nodes = [soma_nodes[int(round(i * ((n_soma_nodes-1) / (n_subgraphs - 1))))] for i in range(n_subgraphs)]
    else:
        soma_attach_nodes = [soma_nodes[0] for i in range(n_subgraphs)]

    # start collecting the nodes in each subgraph of the output graph
    output_subgraphs = []

    # complete the transformation of the original graph by:
    for subgraph, soma_nodes_to_connect in zip(subgraphs, soma_attach_nodes):
        # transforming the subgraphs (dendrites and axons) according to the transform type
        transformed_subgraph, branch_nodes_to_connect = subgraph_transform(
            full_graph = G, 
            subgraph_nodes = subgraph['branch_nodes'], 
            subgraph_root = subgraph['branch_start_node'], 
            transform_type = transform_type
            )
        # and attaching them to the specified node on the soma ring
        neuron_g = subgraph_remerge(
            full_graph = neuron_g, 
            transformed_subgraph = transformed_subgraph, 
            soma_attach_node = soma_nodes_to_connect, 
            subgraph_attach_nodes = branch_nodes_to_connect,
            original_edge_len = subgraph['branch_start_edge_len']
            )
        
        # connect the nodes forming a subgraph in the outputted graph
        transformed_subgraph_nodes = list(transformed_subgraph.nodes())
        transformed_subgraph_nodes.append(soma_nodes_to_connect)
        output_subgraphs.append(transformed_subgraph_nodes)

    print(f"> the directed output graph has {len(list(neuron_g.nodes()))} nodes, and {len(list(neuron_g.edges()))} edges")

    return neuron_g, output_subgraphs