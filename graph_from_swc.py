import numpy as np
import pandas as pd
import networkx as nx

from copy import deepcopy

# distance between two points
def edge_length(input_df, node_id, parent_id):
    node_row = input_df.loc[input_df['comp_id'] == node_id]
    parent_row = input_df.loc[input_df['comp_id'] == parent_id]
    
    node_xyz = np.array([node_row['x'], node_row['y'], node_row['z']]).flatten()
    parent_xyz = np.array([parent_row['x'], parent_row['y'], parent_row['z']]).flatten()
    
    # 3d euclidean distance
    return np.round(np.linalg.norm(node_xyz-parent_xyz), 4)

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
    
def volume_by_type(G):
    edge_volumes_and_types = ([(e[2]['volume'], int(e[2]['edgetype'])) for e in G.edges(data=True)])
    
    soma_volume = (4/3)*np.power(list(G.nodes(data = True))[0][1]['radius'], 3)*np.pi
    axon_volume = np.sum([element[0] for element in edge_volumes_and_types if element[1] == 2])
    dendrite_volume = np.sum([element[0] for element in edge_volumes_and_types if element[1] != 2])
    
    absolute_volumes = np.array([soma_volume, axon_volume, dendrite_volume]).round(2)
    relative_volumes = np.round(absolute_volumes/np.sum(absolute_volumes), 2)

    print('absolute volumes (s, a, d)', absolute_volumes)
    print('relative volumes (s, a, d)', relative_volumes)

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
                str(int(row['comp_id'])), str(int(row['parent_id'])), 
                volume = edge_volume(swc_df, int(row['comp_id']), int(row['parent_id'])),
                len = edge_length(swc_df, int(row['comp_id']), int(row['parent_id'])),
                edgetype = int(row['type'])      
            )
    
    return G

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