'''
TAKE AN UNDIRECTED NEURON GRAPH, WITH MARKED SOMA, DENDRITES, AND AXONS, AND TRANSFORM IT INTO A DIRECTED NETWORK, SUCH THAT:
1) THE SOMA IS REPLACED WITH A RING WITH BIDIRECTIONAL TRANSPORT
2) AXONS AND DENDRITES ARE REPLACED WITH LADDERIZED, CIRCULARIZED, OR BIDIERTIONALIZED VERSIONS
'''

from copy import deepcopy
import networkx as nx
from network_generate import net_gen_hub_ring

nodetype_dict = {1:'soma', 2:'axon',3:'dendrite'}


# isolate a tree subgraph of an undirected graph, and return a directed subtree with edges away from the root node
def make_subgraph_tree_directed(
        G:              nx.Graph, 
        subgraph_nodes: list[str], 
        root:           str
        ) ->            nx.DiGraph:

    D = nx.DiGraph()

    # Add nodes and their attributes to the directed graph
    D.add_nodes_from((n, d) for n, d in G.nodes(data=True) if n in subgraph_nodes)

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
        G:      nx.DiGraph
        ) ->    nx.DiGraph:
    
    L = nx.DiGraph()

    for node, data in G.nodes(data=True):
        # For each node, add a forward and a reverse node to the ladder graph
        L.add_node(f"{node}F", **data)
        L.add_node(f"{node}R", **data)

    for u, v, data in G.edges(data=True):
        # For each edge, add an edge from the forward node of the source to the forward node of the target
        # and from the reverse node of the source to the reverse node of the target in the ladder graph
        L.add_edge(f"{u}F", f"{v}F", **data)
        L.add_edge(f"{v}R", f"{u}R", **data)

    for node in G.nodes():
        # For each node, add an edge between the forward and the reverse node in the ladder graph
        L.add_edge(f"{node}F", f"{node}R")
        L.add_edge(f"{node}R", f"{node}F")
    
    # Now, L is the ladder graph of G
    return L    

# circularize a directed graph
def circularize(
        G:          nx.DiGraph, 
        root_node:  str,
        ) ->        nx.DiGraph:
    
    # Function to get all root-to-leaf paths
    def get_paths(G, root_node):
        paths = []
        for leaf in (n for n, d in G.out_degree() if d==0):
            paths.extend(nx.all_simple_paths(G, root_node, leaf))
        return paths

    # Create an empty directed graph for the circular graph
    C = nx.DiGraph()
    
    # add in the root nodes of each subgraph
    node_data = G.nodes[root_node]
    C.add_node(f"{root_node}F", **node_data)
    C.add_node(f"{root_node}R", **node_data)
    
    # if the graph has only one node, create a forward and a reverse node and connect them
    if len(G.nodes) == 1:
        C.add_edge(f"{root_node}F", f"{root_node}R")
    
    # if the graph has more than one node, traverse the paths from the root 
    else:
        # Get all root-to-leaf paths
        paths = get_paths(G, root_node)
        
        # For each path
        for path in paths:
            for i in range(len(path)-1):
                edge_data = G.edges[path[i], path[i+1]]
                node_data = G.nodes[path[i+1]]
                
                # Add forward and reverse strands
                C.add_node(f"{path[i+1]}F", **node_data)
                C.add_edge(f"{path[i]}F", f"{path[i+1]}F", **edge_data)
                
                C.add_node(f"{path[i+1]}R", **node_data)
                C.add_edge(f"{path[i+1]}R", f"{path[i]}R", **edge_data)
                

            # Connect forward and reverse strands at the leaf nodes
            C.add_edge(f"{path[-1]}F", f"{path[-1]}R")
    
    # Now, C is the circularized graph of G
    return C

# bidirectionalize a directed graph
def bidirectionalize(
        G:      nx.DiGraph
        ) ->    nx.DiGraph:
    
    # Create an empty directed graph for the bidirectional graph
    B = nx.DiGraph()

    # Iterate over the nodes in the original graph
    for node, data in G.nodes(data=True):
        B.add_node(node, **data)

    # Iterate over the edges in the original graph
    for u, v, data in G.edges(data=True):
        # For each edge, add an edge and its reverse to the bidirectional graph
        B.add_edge(u, v, **data)
        B.add_edge(v, u, **data)

    # Now, B is the bidirectional graph of G
    return B


# ladderize, circularize, or bidirectionalize a subgraph (axon or dendrite) from the full neural graph
def subgraph_transform(
        full_graph:             nx.Graph, 
        subgraph_nodes:         list[str], 
        subgraph_root:          str,
        transform_type:         str
        ) ->                    tuple[nx.DiGraph, list]:
    
    # make subgraph directed away from root node
    dir_graph = make_subgraph_tree_directed(full_graph, subgraph_nodes, subgraph_root)
    
    #pos = subset_layout(dir_graph)
    #nx.draw_networkx(dir_graph, pos = pos, connectionstyle='arc3,rad=0.1')
    #plt.show()
    
    # ladderize graph
    if   transform_type == 'ladder':
       
        out_graph = ladderize(dir_graph)
        nodes_to_connect = [f'{subgraph_root}F', f'{subgraph_root}R']
        
        # calculate new positions for visualizing
        #newpos = {f'{key}F':pos[key] for key in pos}
        #for key in pos: newpos[f'{key}R'] = (pos[key][0], pos[key][1]+2000)
    
    # circularize graph
    elif transform_type == 'circle':
        # circularize graph
        out_graph = circularize(dir_graph, subgraph_root)
        nodes_to_connect = [f'{subgraph_root}F', f'{subgraph_root}R']
        
        # calculate new positions for visualizing
        #newpos = {f'{key}F':pos[key] for key in pos}
        #for key in pos: newpos[f'{key}R'] = (pos[key][0], pos[key][1]+2000)
    
    # bidirectionalise graph
    elif transform_type == 'bidirect':
       
        out_graph = bidirectionalize(dir_graph)
        nodes_to_connect = [subgraph_root]
        
        # calculate new positions for visualizing
        #newpos = pos    
    
    #nx.draw_networkx(out_graph, pos = newpos, connectionstyle='arc3,rad=0.1')
    #plt.show()
    
    return out_graph, nodes_to_connect

# merge a transformed subgraph back into to the full graph
def subgraph_remerge(
        full_graph:             nx.DiGraph, 
        transformed_subgraph:   nx.DiGraph,
        soma_attach_node:       str, 
        subgraph_attach_nodes:  list[str]
        ) ->                    nx.DiGraph:
    
    full_graph = nx.union(full_graph, transformed_subgraph)
    
    # ladderized and circularized graphs have their forward and reverse sides connected to the target soma node
    if   len(subgraph_attach_nodes) == 2:
        full_graph.add_edge(soma_attach_node, subgraph_attach_nodes[0])
        full_graph.add_edge(subgraph_attach_nodes[1], soma_attach_node)
    # bidirectionailzed graphs connect their root to the target soma node
    elif len(subgraph_attach_nodes) == 1:
        full_graph.add_edge(soma_attach_node, subgraph_attach_nodes[0])
        full_graph.add_edge(subgraph_attach_nodes[0], soma_attach_node)
        
    return full_graph

# take an undirected graph of a neuron, and transform it into a directed network
def neuron_graph_transform(
        input_G:                nx.Graph, 
        transform_type:         str, 
        n_soma_nodes:           int,
        ) ->                    nx.DiGraph:

    assert transform_type in ['circle','ladder','bidirect'], 'transform_type must be "circle", "ladder", or "bidirect"'
    
    G = deepcopy(input_G) # deepcopy to avoid modifying the source graph
    print(f"> the undirected input graph has {len(list(G.nodes()))} nodes, and {len(list(G.edges()))} edges")


    # isolate subgraph roots (first node of each axon and dendrite)
    subgraph_roots = list(nx.descendants_at_distance(G, '1', 1))
    subgraph_roots.sort(key=lambda x: int(x)) # sort for match of order with subgraph list

    # separate into subgraphs by removing all edges relating to the soma
    G.remove_edges_from(list(G.edges('1')))
    subgraphs = list(nx.connected_components(G)); subgraphs = [list(subgraph) for subgraph in subgraphs]
    subgraphs.sort(key=lambda x: int(x[0])); subgraphs = subgraphs[1:] # sort to match order of subgraph root list
    
    n_subgraphs = len(subgraphs)
    assert n_subgraphs <= n_soma_nodes, f'n_soma_nodes (currently {n_soma_nodes}) must be >= the number of axons and dendrites (currently {n_subgraphs})'
    
    # check the number and types of subgraphs
    subgraph_types = [nodetype_dict[G.nodes()[subgraph[0]]['nodetype']] for subgraph in subgraphs]
    print(f'> identified {n_subgraphs} subgraph(s):')
    print(f"{str([f'{subgraph_types[i]} with {len(subgraphs[i])} nodes' for i in range(n_subgraphs)])[1:-1]}")


    # make soma using the ring generator
    neuron_g = net_gen_hub_ring(n_nodes=n_soma_nodes)
    soma_nodes = list(neuron_g.nodes())
    
    # prepare to place each arbor such that they are maximally spread across the ring
    soma_attach_node_indeces = [round(i * ((n_soma_nodes-1) / (n_subgraphs - 1))) for i in range(n_subgraphs)]


    # transform the subgraphs (dendrites and axons) according to the transform type, and attach them to the specified node on the soma ring
    for i in range(n_subgraphs):
        
        transformed_subgraph, nodes_to_connect = subgraph_transform(G, subgraphs[i], subgraph_roots[i], transform_type)
        
        soma_attach_node = soma_nodes[soma_attach_node_indeces[i]]
        
        neuron_g = subgraph_remerge(neuron_g, transformed_subgraph, soma_attach_node, nodes_to_connect)


    print(f"> the output directed graph has {len(list(neuron_g.nodes()))} nodes, and {len(list(neuron_g.edges()))} edges")


    return neuron_g