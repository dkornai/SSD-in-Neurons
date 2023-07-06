from copy import deepcopy
from collections import Counter
import networkx as nx
import numpy as np

nodetype_dict = {1:'soma', 2:'axon',3:'dendrite'}
nodetype_shortdict = {1:'S', 2:'A',3:'D'}





# get the edges in each branch (axon or dendrite)
def get_edges_in_each_branch(graph, root_node = '1'):
    # List to store branches
    branches = []
    # Depth-first search starting from the root
    for edge in nx.dfs_edges(graph, source=root_node):
        if not branches or edge[0] == root_node:
            # Start a new branch
            branches.append([edge])
        else:
            # Continue the current branch
            branches[-1].append(edge)
    return branches

# get the nodes in each branch (axon or dendrite)
def get_nodes_in_each_branch(graph, root_node = '1'):
    edges_in_each_branch = get_edges_in_each_branch(graph, root_node)
    nodes_in_each_branch = []
    for branch in edges_in_each_branch:
        nodes_in_each_branch.append(sorted(list(set([item for sublist in branch for item in sublist])), key = int))
    
    return nodes_in_each_branch

# distance between two points
def edge_length(input_df, node_id, parent_id):
    node_row = input_df.loc[input_df['comp_id'] == node_id]
    parent_row = input_df.loc[input_df['comp_id'] == parent_id]
    
    node_xyz = np.array([node_row['x'], node_row['y'], node_row['z']]).flatten()
    parent_xyz = np.array([parent_row['x'], parent_row['y'], parent_row['z']]).flatten()
    
    # 3d euclidean distance
    return np.round(np.linalg.norm(node_xyz-parent_xyz), 4)

# distance from a given node to the root
def distance_to_root(G, target_node, root_node = '1'):
    return nx.dijkstra_path_length(G, source=root_node, target=target_node, weight='len')

# approximate the volume of an edge using a cylinder (produces identical results to neuromorpho)
def edge_volume(input_df, node_id, parent_id):
    node_row = input_df.loc[input_df['comp_id'] == node_id]
    parent_row = input_df.loc[input_df['comp_id'] == parent_id]
    
    node_xyz = np.array([node_row['x'], node_row['y'], node_row['z']]).flatten()
    parent_xyz = np.array([parent_row['x'], parent_row['y'], parent_row['z']]).flatten()
    
    # volume of cylinder
    h = np.linalg.norm(node_xyz-parent_xyz)
    r = float(node_row['radius'])
    return np.round(np.pi*r*r*h, 4)

# infer the type of a given subgraph
def infer_graph_type(G):
    gtype_i = int(list(G.nodes(data = True))[0][1]['nodetype'])
    return gtype_i, nodetype_shortdict[gtype_i]

# volume of the soma, axon, and dendrite 
def volume_by_type(G, prnt = False):
    edge_volumes_and_types = ([(e[2]['volume'], int(e[2]['edgetype'])) for e in G.edges(data=True)])
    
    soma_volume = (4/3)*np.power(list(G.nodes(data = True))[0][1]['radius'], 3)*np.pi
    axon_volume = np.sum([element[0] for element in edge_volumes_and_types if element[1] == 2])
    dendrite_volume = np.sum([element[0] for element in edge_volumes_and_types if element[1] == 3])
    
    absolute_volumes = np.array([soma_volume, axon_volume, dendrite_volume]).round(2)
    relative_volumes = np.round(absolute_volumes/np.sum(absolute_volumes), 2)

    if prnt:
        print('absolute volumes (s, a, d)', absolute_volumes)
        print('relative volumes (s, a, d)', relative_volumes)
    
    return absolute_volumes

# number of nodes in the soma, axon, and dendrite
def n_nodes_by_type(G):
    node_types = ([int(node[1]['nodetype']) for node in G.nodes(data=True)])
    return [Counter(node_types)[1], Counter(node_types)[2], Counter(node_types)[3]]

# read a dataframe generated from a .swc file, and generate a graph corresponding to the neuron being described
def nxgraph_from_swc_df(swc_df):
    G = nx.Graph()
    
    # add nodes
    for index, row in swc_df.iterrows():
        G.add_node(
            str(int(row['comp_id'])), 
            nodetype = int(row['type']), 
            radius = float(row['radius']),
            xy = (float(row['x']), float(row['y']))
        )
    
    # add edges
    for index, row in swc_df.iterrows():
        if index > 0:
            G.add_edge(
                str(int(row['parent_id'])), str(int(row['comp_id'])), 
                volume = edge_volume(swc_df, int(row['comp_id']), int(row['parent_id'])),
                len = edge_length(swc_df, int(row['comp_id']), int(row['parent_id'])),
                edgetype = int(row['type'])      
            )

    return G

# remove intermediate (non-branch, non-leaf, non-root) nodes from a full graph 
def remove_transition_nodes(G, swc_df):
    out_df = deepcopy(swc_df)

    # find leaf, root, and branch nodes
    all_nodes = list(G.nodes())
    leaf_nodes = [x for x in G.nodes() if G.degree(x)==1]
    root_node = [all_nodes[0]]; leaf_nodes = [node for node in leaf_nodes if node not in root_node]
    branch_nodes = [x for x in G.nodes() if len(nx.descendants_at_distance(G, x, 1)) > 2]
    
    # find remaining nodes, which are transitions.
    transition_nodes = list(set(G.nodes()) - set(leaf_nodes) - set(root_node) - set(branch_nodes))
    
    # go through the graph, removing the transition nodes
    for tran_node_id in transition_nodes:
        
        tran_node_row = out_df.loc[out_df['comp_id'] == int(tran_node_id)]
        prnt_node_id = int(tran_node_row['parent_id'])
        
        chld_node_row = out_df.loc[out_df['parent_id'] == int(tran_node_id)]

        # rewrite parent of child node to parent of transition node
        out_df.loc[chld_node_row.index, 'parent_id'] = prnt_node_id

        # delete the row corresponding to the node
        out_df = out_df.drop(tran_node_row.index)
            
    
    return out_df

# get the total length of incoming edges to a node
def total_incoming_edge_length(G, node):
    incoming_edges = list(G.in_edges(node, data=True))
    return sum([incoming_edges[i][2].get('len',0) for i in range(len(incoming_edges))])