import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from networkx.drawing.nx_agraph import to_agraph 
'''
Plot the results of a numerical solution to an ODE.
'''
def plot_ode_results(
        results,        # variable values over time (number of wildtype and mutant in each compartment)
        time_points,    # time points where system is sampled
        delta,          # mutant deficiency, used in calculating effective population sizes
        vars,           # name of the variables being tracked (compartment name + wt/mt)
        comp,            # name of the compartments (e.g. soma, axon, etc.)
        prnt = True,
        ):
    PLOT_THRESHOLD = 16

    n_vars = len(vars)
    n_comp = len(comp)
    
    # plot wildtype and mutant counts in each compartment over time
    fig, axs = plt.subplots(2, 2, figsize=(15, 8))
    for i in range(n_vars):
        if n_vars <= PLOT_THRESHOLD:
            axs[0, 0].plot(time_points, results[i], label = vars[i], alpha = 0.7)
        else:
            axs[0, 0].plot(time_points, results[i], alpha = 0.7)

    if n_vars <= PLOT_THRESHOLD: axs[0, 0].legend()
    axs[0, 0].set_title('ODE: wt and mt counts in each compartment over time')

    # plot heteroplasmy levels in each compartment over time
    for i in range(n_comp):
        het = results[(i*2)+1]/(results[(i*2)+1]+results[i*2])
        if n_comp <= PLOT_THRESHOLD/2:
            axs[1, 0].plot(time_points, het, label = f'{comp[i]} het', alpha = 0.7)
        else:
            axs[1, 0].plot(time_points, het, alpha = 0.7)

    if n_comp <= PLOT_THRESHOLD/2: axs[1, 0].legend()
    axs[1, 0].set_ylim([0, 1])
    axs[1, 0].set_title('ODE: heteroplasmy in each compartment over time')

    # plot effective population sizes over time
    min_eps = 100000; max_eps = 0
    for i in range(n_comp):
        eps = results[(i*2)+1]*delta + results[i*2]
        if min(eps) < min_eps: min_eps = min(eps)
        if max(eps) > max_eps: max_eps = max(eps)
        
        if n_comp <= PLOT_THRESHOLD/2:
            axs[0, 1].plot(time_points, eps, label = f'{comp[i]} eff. pop. size', alpha = 0.7)
        else:
            axs[0, 1].plot(time_points, eps, alpha = 0.7)

    axs[0, 1].set_ylim([round(min_eps-5,0), round(max_eps+5,0)])
    if n_comp <= PLOT_THRESHOLD/2: axs[0, 1].legend()
    axs[0, 1].set_title('ODE: effective population size in each compartment over time')

    # plot total population size
    total_pop = np.sum(results, axis = 0)
    axs[1,1].plot(time_points, total_pop, label = f'total pop. size') 
    axs[1,1].legend()
    axs[1,1].set_title('ODE: total population over time')
    axs[1,1].set_ylim([round(min(total_pop)-5,0), round(max(total_pop)+5,0)])
    plt.show()

    if prnt:
        # print parameter values in the final time point
        print("> Final counts of mt and wt in each compartment:")
        print([f'{vars[i]}  {round(results[i,-1], 4)}' for i in range(n_vars)])

        print("\n> Final effective population sizes in each compartment:")
        print([f'{comp[i]}  {round(results[(i*2)+1,-1]*delta + results[i*2,-1], 4)}' for i in range(n_comp) ])


'''
Plot the mean values across many replicate gillespie simulations of the system
'''
def plot_sde_results(
        replicate_results,  # variable values over time (number of wildtype and mutant in each compartment)
        time_points,        # time points where system is sampled
        delta,              # mutant deficiency, used in calculating effective population sizes
        vars,               # name of the variables being tracked (compartment name + wt/mt)
        comp,                # name of the compartments (e.g. soma, axon, etc.)
        prnt = True,
        ):
    
    PLOT_THRESHOLD = 16

    n_vars = len(vars)
    n_comp = len(comp)
    n_samples = replicate_results.shape[0]
    
    # separate out wt and mt counts
    wt_counts = replicate_results[:,np.arange(0, n_vars, 2),:]
    mt_counts = replicate_results[:,np.arange(1, n_vars, 2),:]
    total_wt = np.sum(wt_counts, axis = 1)
    total_mt = np.sum(mt_counts, axis = 1)


    ## plot wildtype and mutant counts in each compartment over time
    mean_per_var_counts = []
    fig, axs = plt.subplots(2, 2, figsize=(15, 8))
    for i in range(n_vars):
        counts = np.nanmean(replicate_results[:,i,:], axis = 0)
        mean_per_var_counts.append(counts)
        if n_vars <= PLOT_THRESHOLD:
            axs[0, 0].plot(time_points, counts, label = vars[i], alpha = 0.7)
        else:
            axs[0, 0].plot(time_points, counts, alpha = 0.7)
    
    if n_vars <= PLOT_THRESHOLD:axs[0, 0].legend()
    axs[0, 0].set_title('SDE: mean wt and mt counts in each compartment over time')


    ## plot effective population sizes over time
    min_eps = 100000; max_eps = 0
    mean_per_comp_eps = []
    for i in range(n_comp):
        eps = np.nanmean(mt_counts[:,i,:]*delta + wt_counts[:,i,:], axis = 0)
        mean_per_comp_eps.append(eps)
        if min(eps) < min_eps: min_eps = min(eps)
        if max(eps) > max_eps: max_eps = max(eps)

        if n_comp <= PLOT_THRESHOLD/2:
            axs[1, 0].plot(time_points, eps, label = f'{comp[i]} eff. pop. size', alpha = 0.7)
        else:
            axs[1, 0].plot(time_points, eps, alpha = 0.7)

    axs[1, 0].set_ylim([round(min_eps-5,0), round(max_eps+5,0)])
    if n_comp <= PLOT_THRESHOLD/2: axs[1, 0].legend()
    axs[1, 0].set_title('SDE: mean effective population size in each compartment over time')


    ## plot heteroplasmy levels over time
    total_het = total_mt/(total_mt+total_wt)
    total_het_mean = np.nanmean(total_het, axis = 0)
    total_het_sem = np.nanstd(total_het, axis = 0)/np.sqrt(n_samples)
    total_het_lb = total_het_mean - total_het_sem
    total_het_ub = total_het_mean + total_het_sem

    # Flatten the results and create a corresponding time array
    #times = np.repeat(time_points, total_het.shape[0])
    # flat_results = total_het.flatten(order = 'F')

    # # Calculate and plot 2D histogram (heatmap)
    # heatmap, xedges, yedges = np.histogram2d(times, flat_results, bins=[20, 30])
    # cmap = plt.cm.binary  
    # plt.imshow(heatmap.T, origin='lower', aspect='auto', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap)

    axs[0, 1].plot(time_points, total_het_mean, label = f'mean heteroplasmy') 
    axs[0, 1].fill_between(time_points, total_het_lb, total_het_ub, color='blue', alpha=0.2)

    axs[0, 1].set_ylim([0, 1])
    if n_comp <= PLOT_THRESHOLD/2: axs[0, 1].legend()
    axs[0, 1].set_title('SDE: mean heteroplasmy over time')


    ## plot total population size
    total_pop = np.sum(replicate_results, axis = 1)
    total_pop_mean = np.nanmean(total_pop, axis = 0)
    total_pop_sem = np.nanstd(total_pop, axis=0)/np.sqrt(n_samples)
    total_pop_lb = total_pop_mean-total_pop_sem
    total_pop_ub = total_pop_mean+total_pop_sem
    

    # Flatten the results and create a corresponding time array
    #times = np.repeat(time_points, total_pop.shape[0])
    # flat_results = total_pop.flatten(order = 'F')

    # # Calculate and plot 2D histogram (heatmap)
    # heatmap, xedges, yedges = np.histogram2d(times, flat_results, bins=[20, 30])
    # cmap = plt.cm.binary  
    # plt.imshow(heatmap.T, origin='lower', aspect='auto', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap)

    axs[1, 1].plot(time_points, np.mean(total_pop, axis = 0), label = f'mean total pop. size') 
    axs[1, 1].fill_between(time_points, total_pop_lb, total_pop_ub, color='blue', alpha=0.2)

    axs[1, 1].legend()
    axs[1, 1].set_title('SDE: total population over time')
    plt.show()

    if prnt:
        # print parameter values in the final time point
        print("> Final mean counts of mt and wt in each compartment:")
        print([f'{vars[i]}  {round(mean_per_var_counts[i][-1], 4)}' for i in range(n_vars)])

        print("\n> Final mean effective population sizes in each compartment:")
        print([f'{comp[i]}  {round(mean_per_comp_eps[i][-1], 4)}' for i in range(n_comp)])

    print("\n> Change in mean heteroplasmy: ")
    # get the total sums across every compartment (keeps replicates and time seperate)
    het_start = total_het_mean[0]
    print(f"start: {round(het_start, 4)} +-{round(total_het_sem[0],4)}" )
    het_final = total_het_mean[-1]
    print(f"final: {round(het_final, 4)} +-{round(total_het_sem[-1],4)}" )
    
    print("delta:", round(het_final-het_start, 4))



# plotting the network 
plot_col_dict = {0:'lightblue', 1:'lightgreen', 2:'orange'}
def plot_simulator_graph(G):
    # Assign colors based on 'birth_type' attribute
    node_colors = [plot_col_dict[G.nodes[node]["birth_type"]] for node in G.nodes()]

    pos = subset_layout(G)
    
    # get edge widths based on rates, and scale for display
    edge_widths = np.array([G.edges[edge]['rate'] for edge in G.edges()])
    edge_widths = np.interp(edge_widths, (edge_widths.min(), edge_widths.max()), (0.5, 3))
    edge_widths = list(edge_widths)

    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed

    # Draw nodes with sizes and colors based on attributes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.8)

    # Draw edges with adjusted connectionstyle
    nx.draw_networkx_edges(G, pos, alpha=1, width=edge_widths, arrowstyle='->', connectionstyle='arc3,rad=0.1')

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', font_color='r')

    plt.axis('off')  # Hide the axis
    plt.show()       # Display the graph





# node colors, xy coordinates, and radii from real data
swc_color_dict = {1:'red', 2:'blue', 3:'limegreen', 4:'limegreen'}
def draw_helpers(
        G:              nx.Graph
        ):

    node_colors = [swc_color_dict[data['nodetype']] for node, data in G.nodes(data=True)]
        
    xy_coordinates = {node:data['xy'] for node, data in G.nodes(data=True)}

    node_radius = np.array([data['radius'] for node, data in G.nodes(data=True)])
    n_axon_nodes = node_colors.count('red') # get number of axons (red nodes)
    if  n_axon_nodes == 1:
        node_radius = node_radius*(5/np.min(node_radius)) # scale for display such that max radius is 5 if only 1 axonic node is present
    else:
        node_radius = node_radius*(2/np.min(node_radius)) # scale for display such that max radius is 2 if only 1 axonic node is present

    
    return node_colors, xy_coordinates, node_radius



# plot a network derived from the swm file corresponding to a real neuron
def plot_neuron_graph_realxy(
        G:              nx.Graph,
        ):
    
    node_colors, xy_coordinates, node_radius = draw_helpers(G)

    assert type(G) == nx.Graph, 'real x-y coordinate plotting only viable for undirected raw network'
    
   # drawing using real xy coordinates from swm file
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw(G, pos=xy_coordinates, node_size=node_radius, ax=ax, node_color = node_colors)
    
    plt.show()


def subset_layout(G):
    # drawing using layout that shows structure
    A = to_agraph(G) # Convert the networkx graph to a pygraphviz graph
    
    # get the xy coordinates from the neato layout engine
    A.graph_attr.update(model='subset')
    A.layout(prog='neato')
    positions = {}
    for node in A.nodes():
        x, y = map(float, node.attr['pos'].split(','))
        positions[node] = (x, y)

    return positions

    # plot a network derived from the swm file corresponding to a real neuron
def plot_neuron_graph_subset(
        G:              nx.Graph,
        ):
    
    node_colors, xy_coordinates, node_radius = draw_helpers(G)

    # drawing using layout that shows structure
    A = to_agraph(G) # Convert the networkx graph to a pygraphviz graph

    # get the xy coordinates from the neato layout engine
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw(G, pos=subset_layout(G), node_size=node_radius, ax=ax, node_color = node_colors)
    
    plt.show()