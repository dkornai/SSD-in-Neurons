import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

'''
Plot the results of a numerical solution to an ODE.
'''
def plot_ODE(
        results,        # variable values over time (number of wildtype and mutant in each compartment)
        time_points,    # time points where system is sampled
        delta,          # mutant deficiency, used in calculating effective population sizes
        vars,           # name of the variables being tracked (compartment name + wt/mt)
        comp            # name of the compartments (e.g. soma, axon, etc.)
        ):
    
    n_vars = len(vars)
    n_comp = len(comp)
    
    # plot wildtype and mutant counts in each compartment over time
    plt.subplots(figsize=(10, 5))
    for i in range(n_vars):
        plt.plot(time_points, results[i], label = vars[i], alpha = 0.7)
    plt.legend()
    plt.title('wt and mt counts in each compartment over time')

    # plot heteroplasmy levels in each compartment over time
    plt.subplots(figsize=(10, 5))
    for i in range(n_comp):
        het = results[(i*2)+1]/(results[(i*2)+1]+results[i*2])
        plt.plot(time_points, het, label = f'{comp[i]} het', alpha = 0.7)
    plt.ylim([0, 1])
    plt.legend()
    plt.title('heteroplasmy in each compartment over time')

    # plot effective population sizes over time
    plt.subplots(figsize=(10, 5))
    min_eps = 100000; max_eps = 0
    for i in range(n_comp):
        eps = results[(i*2)+1]*delta + results[i*2]
        if min(eps) < min_eps: min_eps = min(eps)
        if max(eps) > max_eps: max_eps = max(eps)

        plt.plot(time_points, eps, label = f'{comp[i]} eff. pop. size', alpha = 0.7)
    plt.ylim([round(min_eps-5,0), round(max_eps+5,0)])
    plt.legend()
    plt.title('effective population size in each compartment over time')

    # print parameter values in the final time point
    print("> Final counts of mt and wt in each compartment:")
    for i in range(n_vars):
        print(f'{vars[i]}\t{round(results[i,-1], 4)}\t')

    print("\n> Final effective population sizes in each compartment:")
    for i in range(n_comp):
        eps = results[(i*2)+1,-1]*delta + results[i*2,-1]
        print(f'{comp[i]}\t{round(eps, 4)}\t')


'''
Plot the mean values across many replicate gillespie simulations of the system
'''
def plot_gillespie(
        replicate_results,  # variable values over time (number of wildtype and mutant in each compartment)
        time_points,    # time points where system is sampled
        delta,          # mutant deficiency, used in calculating effective population sizes
        vars,           # name of the variables being tracked (compartment name + wt/mt)
        comp            # name of the compartments (e.g. soma, axon, etc.)
        ):
    
    n_vars = len(vars)
    n_comp = len(comp)
    
    # separate out wt and mt counts
    wt_counts = replicate_results[:,np.arange(0, n_vars, 2),:]
    mt_counts = replicate_results[:,np.arange(1, n_vars, 2),:]

    # plot wildtype and mutant counts in each compartment over time
    mean_per_var_counts = []
    plt.subplots(figsize=(10, 5))
    for i in range(n_vars):
        counts = np.nanmean(replicate_results[:,i,:], axis = 0)
        mean_per_var_counts.append(counts)

        plt.plot(time_points, counts, label = vars[i], alpha = 0.7)
    plt.legend()
    plt.title('mean wt and mt counts in each compartment over time')

    # plot heteroplasmy levels in each compartment over time
    mean_per_comp_het = []
    plt.subplots(figsize=(10, 5))
    for i in range(n_comp):
        # calculate mean heteroplasmy as a mean of ratios
        het = np.nanmean(mt_counts[:,i,:]/(mt_counts[:,i,:]+wt_counts[:,i,:]), axis = 0)
        mean_per_comp_het.append(het)

        plt.plot(time_points, het, label = f'{comp[i]} het', alpha = 0.7)
    plt.ylim([0, 1])
    plt.legend()
    plt.title('mean heteroplasmy in each compartment over time')

    # plot effective population sizes over time
    min_eps = 100000; max_eps = 0
    mean_per_comp_eps = []
    plt.subplots(figsize=(10, 5))
    for i in range(n_comp):
        eps = np.nanmean(mt_counts[:,i,:]*delta + wt_counts[:,i,:], axis = 0)
        mean_per_comp_eps.append(eps)
        if min(eps) < min_eps: min_eps = min(eps)
        if max(eps) > max_eps: max_eps = max(eps)

        plt.plot(time_points, eps, label = f'{comp[i]} eff. pop. size', alpha = 0.7)
    plt.ylim([round(min_eps-5,0), round(max_eps+5,0)])
    plt.legend()
    plt.title('mean effective population size in each compartment over time')

    # print parameter values in the final time point
    print("> Final mean counts of mt and wt in each compartment:")
    for i in range(n_vars):
        print(f'{vars[i]}\t{round(mean_per_var_counts[i][-1], 4)}\t')

    print("\n> Final mean heteroplasmy in each compartment:")
    for i in range(n_comp):
        print(f'{comp[i]}\t{round(mean_per_comp_het[i][-1], 4)}\t')

    print("\n> Final mean effective population sizes in each compartment:")
    for i in range(n_comp):
        print(f'{comp[i]}\t{round(mean_per_comp_eps[i][-1], 4)}\t')

    print("\n> Change in mean heteroplasmy: ")
    # get the total sums across every compartment (keeps replicates and time seperate)
    total_wt = np.sum(wt_counts, axis = 1)
    total_mt = np.sum(mt_counts, axis = 1)

    het_start = np.average(total_mt[:,0]/(total_mt[:,0]+total_wt[:,0]), 
                           axis = 0, 
                           weights=total_mt[:,0]*delta + total_wt[:,0])
    print("start:", round(het_start, 4))
    het_final = np.average(total_mt[:,-1]/(total_mt[:,-1]+total_wt[:,-1]), 
                           axis = 0, 
                           weights=total_mt[:,-1]*delta + total_wt[:,-1])
    print("final:", round(het_final, 4))
    print("delta:", round(het_final-het_start, 4))


def plot_network(G):
    # Assign colors based on 'birth_type' attribute
    node_colors = ['lightblue' if G.nodes[node]['birth_type'] == 0 else 'lightgreen' if G.nodes[node]['birth_type'] == 1 else 'orange' for node in G.nodes()]

    pos = nx.spring_layout(G)
    edge_widths = [G.edges[edge]['rate']*50 for edge in G.edges()]

    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed

    # Draw nodes with sizes and colors based on attributes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.8)

    # Draw edges with adjusted connectionstyle
    nx.draw_networkx_edges(G, pos, alpha=1, width=edge_widths, arrowstyle='->', connectionstyle='arc3,rad=0.1')

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    plt.axis('off')  # Hide the axis
    plt.show()  # Display the graph