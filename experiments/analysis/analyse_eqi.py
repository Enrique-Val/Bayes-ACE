import json
import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import pybnesian as pb

from bayesace import path_likelihood_length
from bayesace.models.lingam_cat import LingamClassifier
from experiments.analysis.utils import plot_dolan_more
from experiments.utils import friedman_posthoc

import scikit_posthocs as sp
from matplotlib import colors as mcolors
# Experiments:

# Seggregate by RUCC, class label and both and analyse:
#      - Which EQI is changed the most
#      - Which variables (of each domain) are changed the most
#      - Compare with Wachter algorithm


def get_palette(algorithms):
    base_colors = {
        "BayesACE": "#0088cc", #"blue",
        "FACE": "#cc6300",
        "Wachter": "orange",
    }
    color_palette = {
        "BayesACE": "#ffb907",  # Golden Yellow
        "FACE": "#2799db",  # Medium Blue
        "Wachter": "#865ebf"  # Purple
    }
    return color_palette
    palette = {}
    for method in algorithms:
        for base_method in base_colors.keys():
            if base_method in method:
                palette[method] = base_colors[base_method]
    return palette

def get_var_palette():
    # Create a palette for each domain
    palette = {}
    palette["Air"] = "#5da2e2"
    palette["Water"] = "#48c9a2"
    palette["Land"] = "#E89C23"
    palette["Built"] = "#AC60CC"
    palette["Sociod"] = "#F05343"
    palette["Sociodemographic"] = "#F05343"
    return palette

eqi_color = "#BDC3C7"


def label_with_medians(data, ax):
    # Label with the median
    medians = data.median().to_numpy()
    vertical_offset = medians * 0.05  # offset from median for display
    # Round median to 2 decimal places
    medians = medians.round(2)
    '''for xtick in ax.get_xticks():
        ax.text(xtick, medians[xtick] + vertical_offset[xtick],
                f'{medians[xtick]:.2f}', color='black', ha='center')'''
    return medians

if __name__ == "__main__":
    domains_capital = ["Air", "Water", "Land", "Built", "Sociod"]
    root_dir = "../results/exp_eqi_final/"
    data_dir = os.path.join(root_dir, "data_processed")
    model_dir = os.path.join(root_dir, "models")
    results_dir = os.path.join(root_dir, "results")
    plots_dir = os.path.join(root_dir, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    algorithms_dir = os.path.join(root_dir, "algorithms")
    # Load the metadata, scaler, data train and test
    metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv"), index_col=0)
    scaler = pickle.load(open(os.path.join(model_dir, "scaler.pkl"), "rb"))
    scaler_class = pickle.load(open(os.path.join(model_dir, "scaler_class.pkl"), "rb"))
    df_train = pd.read_csv(os.path.join(data_dir, "data_train.csv"), index_col=0)
    df_test = pd.read_csv(os.path.join(data_dir, "data_test.csv"), index_col=0)
    data = pd.concat([df_train, df_test], axis=0)
    var_types = pickle.load(open(os.path.join(data_dir, "var_types.pkl"), "rb"))
    # Reorder to go as follows: air, water, land, built, sociod
    var_types = {key: var_types[key] for key in ["Air", "Water", "Land", "Built", "Sociodemographic"]}

    # Load the Bayesian network
    with open(os.path.join(model_dir, "lingam.pkl"), "rb") as f:
        bn : LingamClassifier = pickle.load(f)
        bn.class_var_name = "EQI"


    vertices_list = [0, 1]
    penalty_list = [5]#[3, 5, 10, 15]

    # Import the algorithms, results, differences and distances
    algorithms = {}
    results_bank = {}
    differences = {}
    distances = {}

    alg_names = ["face"] + ["wachter"] + ["bayesace_" + str(i) for i in vertices_list]
    new_alg_names = ["FACE"] + ["Wachter"] + [f"BayesACE {i} vertices" for i in vertices_list]
    generic_algorithms = ["FACE", "Wachter", "BayesACE", "BayesACE - FACE"]

    for penalty in penalty_list:
        penalty_dir = os.path.join(plots_dir, "penalty_" + str(penalty))
        if not os.path.exists(penalty_dir):
            os.makedirs(penalty_dir)
        algorithms[penalty] = {}
        results_bank[penalty] = {}
        differences[penalty] = {}
        distances[penalty] = {}

        alg_names_penalty = [f"{alg_name}_{penalty}" for alg_name in alg_names]
        bayesace_names_penalty = alg_names_penalty[2:]
        bayesace_distances = {}
        for alg_name in bayesace_names_penalty:
            bayesace_distances[alg_name] = \
            pd.read_csv(os.path.join(results_dir, "distances_" + alg_name + ".csv"), index_col=0).to_numpy().T[0]
        bayesace_distances = pd.DataFrame(bayesace_distances)

        # Check which algorithm is better. Count which column is the smallest for each row and select that one
        counter = {alg_name: 0 for alg_name in bayesace_names_penalty}
        for i in range(bayesace_distances.shape[0]):
            row = bayesace_distances.iloc[i]
            min_alg = row.idxmin()
            counter[min_alg] += 1

        # Select the one with the bigger counter
        best_alg = max(counter, key=counter.get)
        best_alg = "bayesace_1_"+ str(penalty)
        #best_alg = "bayesace_1_"+ str(penalty)
        selected_algs = alg_names_penalty[:2] + [best_alg]

        name_map = {}

        for new_alg_name, alg_name in zip(new_alg_names, alg_names_penalty):
            if not alg_name in selected_algs:
                continue
            name_map[alg_name] = new_alg_name
            alg_path = os.path.join(algorithms_dir, alg_name + ".pkl")
            # algorithms[penalty][new_alg_name] = pickle.load(open(alg_path, "rb"))
            differences[penalty][new_alg_name] = pd.read_csv(os.path.join(results_dir, "diff_" + alg_name + ".csv"),
                                                             index_col=0)
            distances[penalty][new_alg_name] = pd.read_csv(
                os.path.join(results_dir, "distances_" + alg_name + ".csv"), index_col=0)
            results_bank_dir = os.path.join(results_dir, alg_name)
            results_bank[penalty][new_alg_name] = {}
            for i in range(len(os.listdir(results_bank_dir))):
                file_i = str(i) + ".pkl"
                instance_path = os.path.join(results_bank_dir, file_i)
                results_bank[penalty][new_alg_name][i] = pickle.load(open(instance_path, "rb"))
        # Add as a new key the difference between the best BayesACE and FACE
        differences[penalty]["BayesACE - FACE"] = (differences[penalty][name_map[best_alg]] - differences[penalty][
            name_map["face_" + str(penalty)]]).abs()
        distances[penalty]["BayesACE - FACE"] = (distances[penalty][name_map[best_alg]] - distances[penalty][
            name_map["face_" + str(penalty)]]).abs()
        eqis = differences[penalty]["Wachter"].columns[-5:]

    # --- STEP 1: PARSE ADJACENCY MATRIX ---
    # iterating through the matrix (Assuming Index=Source/Parent, Columns=Target/Child)
    # stack() converts it to a Series with a MultiIndex (Parent, Child)
    adj_matrix = bn.lingam_model.adjacency_matrix_.T
    node_names = bn.columns + [bn.get_class_var_name()]
    sp_eqi_vars = node_names[-6:]
    print(sp_eqi_vars)


    def get_display_name(name):
        """
        Replicates your original logic:
        If node contains "EQI" (but isn't just "EQI"), format it nicely.
        """
        if "EQI" in name and "EQI" != name:
            return "EQI " + name.split("_")[0].capitalize()
        return name


    # --- 2. PARSE NUMPY ADJACENCY MATRIX ---

    edge_list = []
    edge_weights = []

    # Get indices where edges exist (non-zero weight)
    # rows = parents, cols = children
    rows, cols = np.nonzero(adj_matrix)

    for r, c in zip(rows, cols):
        weight = adj_matrix[r, c]

        # Optional: Filter out negligible weights if needed
        if abs(weight) < 1e-8:
            continue

        # Map index to original name
        raw_parent = node_names[r]
        raw_child = node_names[c]

        # Apply the display formatting
        u = get_display_name(raw_parent)
        v = get_display_name(raw_child)

        edge_list.append((u, v))
        edge_weights.append(weight)

    # --- STEP 2: BUILD GRAPH & COLORS ---
    graph = nx.DiGraph()
    edge_colors = []

    # Replicating your original color logic
    for edge, weight in zip(edge_list, edge_weights):
        u, v = edge

        # Logic: Green for positive, Red for negative.
        # Reversed if specific "EQI" conditions are met.
        is_eqi_involved = ("EQI" in u and "EQI" in v) or ("EQI" not in u and "EQI" not in v)

        if is_eqi_involved:
            if weight > 0:
                edge_colors.append("green")
            else:
                edge_colors.append("red")
        else:
            # One is EQI, one is not -> Reverse colors
            if weight > 0:
                edge_colors.append("red")
            else:
                edge_colors.append("green")

    # Add edges to graph
    # Note: we use absolute weight for the attribute, but keep track of sign via color
    abs_weights = [abs(w) for w in edge_weights]

    for edge, weight, color, raw_weight in zip(edge_list, abs_weights, edge_colors, edge_weights):
        graph.add_edges_from([edge], weight=weight, color=color, raw_weight=raw_weight)

    # --- STEP 3: PLOTTING (EQI SUBGRAPH) ---
    eqi_nodes = [node for node in graph.nodes() if "EQI" in node]
    if eqi_nodes:  # Check to ensure we actually have EQI nodes
        graph_eqi = graph.subgraph(eqi_nodes)

        # Layout
        try:
            pos = nx.drawing.nx_agraph.graphviz_layout(graph_eqi, prog="dot")
        except:
            pos = nx.spring_layout(graph_eqi)  # Fallback if graphviz is missing

        edge_weights_dict = nx.get_edge_attributes(graph_eqi, "weight")
        edge_weights_sub = np.array(list(edge_weights_dict.values())) * 6
        edge_colors_sub = list(nx.get_edge_attributes(graph_eqi, "color").values())

        # Node Colors
        node_color_mapping = []
        for node in graph_eqi.nodes():
            if node == "EQI":
                node_color_mapping.append(eqi_color)  # Ensure 'eqi_color' is defined in your env
            else:
                # Safe access to palette
                try:
                    key = node.split(" ")[1]
                    node_color_mapping.append(get_var_palette()[key])
                except (IndexError, KeyError):
                    node_color_mapping.append("grey")  # Fallback color

        fig = plt.figure(figsize=(10, 5))
        ax = fig.gca()
        nx.draw(graph_eqi, pos, with_labels=True, width=edge_weights_sub,
                edge_color=edge_colors_sub, node_size=2500,
                font_size=14, font_weight="bold", node_color=node_color_mapping)

        if not os.path.exists(os.path.join(plots_dir, "graphs")):
            os.makedirs(os.path.join(plots_dir, "graphs"))
        ax.margins(0.1)
        fig.savefig(os.path.join(plots_dir, "graphs", "eqi_graph.pdf"))

        # Label weights
        color_mapping = {"green": "#006400", "red": "#8B0000"}

        for (u, v), weight in edge_weights_dict.items():
            if u not in pos or v not in pos: continue
            x, y = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
            edge_c = nx.get_edge_attributes(graph_eqi, "color")[(u, v)]
            plt.text(x, y, str(round(weight, 2)), fontsize=8,
                     color=color_mapping.get(edge_c, "black"),
                     fontweight="bold", ha='center', va='center')

        if not os.path.exists(os.path.join(plots_dir, "graphs_weight")):
            os.makedirs(os.path.join(plots_dir, "graphs_weight"))
        plt.savefig(os.path.join(plots_dir, "graphs_weight", "eqi_graph_weights.pdf"))
        plt.show()
        plt.close()

    # --- STEP 4: DOMAIN PLOTS ---
    # (This logic remains largely similar, just ensuring robust dictionary access)

    for domain_raw_name in sp_eqi_vars[:-1]:  # Exclude the last one if it's not a domain
        domain = get_display_name(domain_raw_name)[4:]
        domain_vars = var_types.get(domain if domain != "Sociod" else "Sociodemographic", [])

        # Handle domain aliases

        color_dom = get_var_palette()[domain]

        # Filter nodes that exist in the new graph
        domain_nodes = [node for node in graph.nodes() if node in domain_vars]
        eqi_node = "EQI " + domain

        # Proceed only if nodes exist
        if not domain_nodes:
            continue

        domain_graph = graph.subgraph(domain_nodes + [eqi_node])
        collapse = "ate"

        # ... [Insert your existing Logic for 'Total', 'Louvain', 'Importance' here] ...
        # The graph object 'domain_graph' is now fully compatible with your existing code below.

        if collapse == "Importance":
            # Note: This part requires 'df_train' to be available in your environment
            # and 'eqis' list.
            df_train_dom = df_train[domain_nodes].copy()

            # Find real EQI name (reverse lookup or fuzzy match)
            eqi_real_name = next((e for e in eqis if domain.lower() in e), None)

            if eqi_real_name:
                df_train_dom[eqi_node] = df_train[eqi_real_name]
                corr_matrix = df_train_dom.corr()

                if eqi_node in corr_matrix:
                    eqi_corr = corr_matrix[eqi_node].abs().sort_values(ascending=False)
                    # Slice top correlated (excluding self)
                    top_nodes = list(eqi_corr.index[1:9])
                    # Ensure these nodes are actually in our subgraph
                    valid_top_nodes = [n for n in top_nodes if n in domain_graph.nodes()]

                    domain_graph = graph.subgraph(valid_top_nodes + [eqi_node])

                    # Plotting...
                    try:
                        pos = nx.drawing.nx_agraph.graphviz_layout(domain_graph, prog="dot")
                    except:
                        pos = nx.spring_layout(domain_graph)

                    edge_weights_dict = nx.get_edge_attributes(domain_graph, "weight")
                    edge_weights_sub = np.array(list(edge_weights_dict.values())) * 3
                    edge_colors_sub = list(nx.get_edge_attributes(domain_graph, "color").values())

                    fig = plt.figure(figsize=(10, 5))
                    ax = fig.gca()
                    nx.draw(domain_graph, pos, with_labels=True, width=edge_weights_sub,
                            edge_color=edge_colors_sub, node_size=1000,
                            font_size=14, font_weight="bold", node_color=color_dom, ax=ax)
                    ax.margins(0.2)
                    fig.savefig(os.path.join(plots_dir, "graphs", domain + "_graph_importance.pdf"))
                    plt.show()
                    plt.close()
        elif collapse == "Total" :
            pos = nx.drawing.nx_agraph.graphviz_layout(domain_graph, prog="dot")
            #pos = nx.spring_layout(domain_graph, pos=pos, k =10/np.sqrt(len(domain_nodes)), iterations = 1000)
            edge_weights_dict = nx.get_edge_attributes(domain_graph, "weight")
            edge_weights_sub = np.array(list(edge_weights_dict.values()))*1.5
            edge_colors_sub = list(nx.get_edge_attributes(domain_graph, "color").values())
            fig = plt.figure(figsize=(12, 6))
            ax = fig.gca()
            nx.draw(domain_graph, pos, with_labels=True, width = edge_weights_sub, edge_color = edge_colors_sub, node_size = 300,
                    font_size=14, font_weight="bold", node_color = color_dom, ax=ax)
            ax.margins(0.15)
            fig.savefig(os.path.join(plots_dir, "graphs", domain+"_graph.pdf"))
            plt.show()
            # Label the edges with the weights
            for (u, v), weight in edge_weights_dict.items():
                x, y = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
                edge_color = nx.get_edge_attributes(domain_graph, "color")[(u, v)]
                edge_color = color_mapping[edge_color]
                fig.text(x, y, str(round(weight,2)), fontsize=8, color=edge_color, fontweight="bold", ha='center', va='center')
            fig.savefig(os.path.join(plots_dir, "graphs_weight", domain+"_graph_weights.pdf"))
            plt.close()
        elif collapse == "Louvain":
            resolution = 2
            if domain == "Built":
                resolution = 7
            elif domain == "Sociod" :
                resolution = 7
            domain_graph_no_eqi = graph.subgraph(domain_nodes)

            # Step 1: Apply Louvain to find communities using networkx
            partition = nx.community.louvain_communities(domain_graph_no_eqi.to_undirected(), resolution = resolution)

            # Convert partition to a dictionary and find representative nodes
            partition_dict = {}
            community_representatives = {}

            for i, community in enumerate(partition):
                sorted_nodes = sorted(community)  # Sort nodes in ascending order
                representative = sorted_nodes[0]  # Pick the top node as the name
                community_representatives[i] = representative

                for node in community:
                    partition_dict[node] = i

            # Ensure EQI node is its own separate community
            partition_dict[eqi_node] = -1  # Assign a unique community ID

            # Step 2: Collapse nodes into supernodes based on community labels
            super_graph = nx.DiGraph()
            node_to_super = {}

            for node, comm in partition_dict.items():
                if node == eqi_node:
                    supernode = eqi_node
                else:
                    supernode = community_representatives[comm]  # Use top node as name
                node_to_super[node] = supernode
                super_graph.add_node(supernode)

            # Step 3: Add edges between supernodes (aggregating weights)
            for u, v, data in domain_graph.edges(data=True):
                super_u = node_to_super[u]
                super_v = node_to_super[v]

                weight = data.get("weight", 1.0)
                color = data.get("color")
                if color == "red" :
                    weight = weight*-1


                if super_u != super_v:  # Only keep inter-community edges
                    if not super_graph.has_edge(super_u, super_v) and not super_graph.has_edge(super_v, super_u):
                        super_graph.add_edge(super_u, super_v, weight=weight)
                    elif super_graph.has_edge(super_u, super_v) :
                        super_graph[super_u][super_v]["weight"] += weight
                    elif super_graph.has_edge(super_v, super_u) :
                        super_graph[super_v][super_u]["weight"] += weight
                    else :
                        raise ValueError("This should not happen")

            # Step 4: Compute edge colors based on weight averages
            edge_colors = []
            edge_weights = []
            for u, v, data in super_graph.edges(data=True):
                weight = data["weight"]
                edge_weights.append(weight * 1.5)
                edge_colors.append(color_mapping["green"] if weight > 0 else color_mapping["red"])

            # Step 5: Compute layout and plot
            pos = nx.drawing.nx_agraph.graphviz_layout(super_graph, prog="dot")
            #pos = nx.spring_layout(super_graph)

            fig = plt.figure(figsize=(12, 6))
            ax = fig.gca()
            nx.draw(super_graph, pos, with_labels=True, width=edge_weights, edge_color=edge_colors,
                    node_size=500, font_size=14, font_weight="bold", node_color=color_dom, ax = ax)

            ax.margins(0.15)
            #fig.savefig(os.path.join(plots_dir, "graphs", domain + "_collapsed_graph.pdf"))
            plt.show()

        elif collapse == "ate" :
            # Graphs based on the average total effect
            test_cont = data.drop(columns=["EQI"])
            test_cont[test_cont.columns[:-1]] = scaler.transform(test_cont[test_cont.columns[:-1]])
            test_cont[test_cont.columns[-1]] = scaler_class.transform(test_cont[[test_cont.columns[-1]]])
            # Compute the average total effect on test data
            ate_dict = {}
            for node in domain_nodes:
                ate_dict[node] = bn.lingam_model.estimate_total_effect(test_cont, node_names.index(node), node_names.index(domain_raw_name))
            # Retain only the top 8 most important nodes according to absolute value of the ATE
            n_top_vars = 6
            ate_dict = {k: v for k, v in sorted(ate_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:n_top_vars]}
            domain_graph_ate = graph.subgraph(list(ate_dict.keys()) + [eqi_node]).copy()
            # Drop edges whose weight is less than 0.15 in absolute value
            eqi_node_pasted = "EQI " + domain
            edges_to_remove = [(u, v) for u, v, data in domain_graph_ate.edges(data=True) if abs(data["weight"]) < 0.2 and not (u == eqi_node_pasted or v == eqi_node_pasted)]
            domain_graph_ate.remove_edges_from(edges_to_remove)
            pos = nx.drawing.nx_agraph.graphviz_layout(domain_graph_ate, prog="dot")
            edge_weights_dict = nx.get_edge_attributes(domain_graph_ate, "weight")
            edge_weights_sub = np.array(list(edge_weights_dict.values())) * 5
            edge_colors_sub = list(nx.get_edge_attributes(domain_graph_ate, "color").values())
            fig = plt.figure(figsize=(8*0.95, 6*0.95))
            ax = fig.gca()
            nx.draw(domain_graph_ate, pos, with_labels=True, width=edge_weights_sub,
                    edge_color=edge_colors_sub, node_size=1200,
                    font_size=14, font_weight="bold", node_color=color_dom, ax=ax)
            ax.margins(0.2)
            fig.savefig(os.path.join(plots_dir, "graphs", domain + "_graph_ate.pdf"))
            plt.show()
            plt.close()





    generic_exp = False
    if generic_exp:
        # I need a experiment that, for each penalty, shows the sparsity, distance and diff_l2
        for i,penalty in enumerate(penalty_list):
            fig, ax = plt.subplots(3, 1, figsize=(8/2, 3*3))
            fig_cdd, ax_cdd = plt.subplots(3,1, figsize=(10,6))
            penalty_dir = os.path.join(plots_dir, "penalty_" + str(penalty))
            sparsity = pd.DataFrame(index=differences[penalty]["Wachter"].index[3:], columns=["FACE", "Wachter", "BayesACE"])
            diff_l2 = sparsity.copy()
            distance_ll =sparsity.copy()
            for gen_alg,alg_name in zip(["FACE", "Wachter", "BayesACE"],differences[penalty].keys()):
                diff = differences[penalty][alg_name]
                diff = diff.iloc[3:]
                sparsity[gen_alg] = (diff.abs() > 0.25).sum(axis=1) / len(diff.columns)
                diff_unscaled = diff.copy()
                diff_unscaled[:] = scaler.inverse_transform(diff)
                diff_l2[gen_alg] = (diff_unscaled**2).sum(axis=1).to_numpy()
                # For computing the log-likelihood aware dist, we need to resort to the data_bank
                # Compute sum of differences over the neural net
                distance_ll[gen_alg] = distances[penalty][alg_name].iloc[3:].to_numpy().T[0]
            # Reorder diff_l2, sparsity and distance_ll to be the ["BayesACE", "FACE", "Wachter"]
            diff_l2 = diff_l2[["BayesACE", "FACE", "Wachter"]]
            sparsity = sparsity[["BayesACE", "FACE", "Wachter"]]
            distance_ll = distance_ll[["BayesACE", "FACE", "Wachter"]]
            palette = get_palette(["FACE", "Wachter", "BayesACE"])
            fbh_sparsity = friedman_posthoc(sparsity)
            fbh_diff = friedman_posthoc(diff_l2)
            fbh_distance = friedman_posthoc(distance_ll)
            sp.critical_difference_diagram(fbh_distance["summary_ranks"], fbh_distance["p_adjusted"], ax=ax_cdd[0],
                                           color_palette = palette)
            ax_cdd[0].set_title("Density-aware distance")
            sp.critical_difference_diagram(fbh_diff["summary_ranks"], fbh_diff["p_adjusted"], ax=ax_cdd[1],
                                           color_palette = palette)
            ax_cdd[1].set_title("Euclidean distance")
            sp.critical_difference_diagram(fbh_sparsity["summary_ranks"], fbh_sparsity["p_adjusted"], ax=ax_cdd[2],
                                           color_palette = palette)
            ax_cdd[2].set_title("Sparsity")
            distance_ll = distance_ll.subtract(distance_ll.mean(axis=1), axis=0).divide(distance_ll.std(axis=1), axis=0)
            #sns.boxplot(data=distance_ll, ax=ax[0], palette = palette, showfliers=False)
            plot_dolan_more(distance_ll, ax[0], palette = palette)
            ax[0].spines[['right', 'top']].set_visible(False)
            ax[0].set_title("Density-aware distance")
            # Label with the median
            label_with_medians(distance_ll, ax[0])
            diff_l2 = diff_l2.subtract(diff_l2.mean(axis=1), axis=0).divide(diff_l2.std(axis=1), axis=0)
            sns.boxplot(data=diff_l2, ax=ax[1], palette = palette, showfliers=False)
            ax[1].set_title("Euclidean distance")
            ax[1].spines[['right', 'top']].set_visible(False)
            # Label with the median
            label_with_medians(diff_l2, ax[1])
            sparsity = sparsity.subtract(sparsity.mean(axis=1), axis=0).divide(sparsity.std(axis=1), axis=0)
            sns.boxplot(data=sparsity, ax=ax[2], palette = palette, showfliers=False)
            ax[2].set_title("Sparsity")
            ax[2].spines[['right', 'top']].set_visible(False)
            # Label with the median
            label_with_medians(sparsity, ax[2])
            fig.suptitle("Penalty " + str(penalty))
            fig.savefig(os.path.join(penalty_dir, "generic_exp.pdf"))
            plt.close()
            fig_cdd.suptitle("Penalty " + str(penalty))
            fig_cdd.savefig(os.path.join(penalty_dir, "generic_exp_cdd.pdf"))

        # Do an experiment for each penalty
        for alg_i,alg_gen in zip(range(4), generic_algorithms):
            if not alg_gen == "BayesACE":
                pass
            fig, ax = plt.subplots(2, 2, figsize=(14, 4))
            fig_cdd, ax_cdd = plt.subplots(2, 2, figsize=(10, 4))
            for i, penalty in enumerate(penalty_list):
                penalty_dir = os.path.join(plots_dir, "penalty_" + str(penalty), alg_gen)
                if not os.path.exists(penalty_dir):
                    os.makedirs(penalty_dir)
                new_alg_name = list(differences[penalty].keys())[alg_i]
                # Get the differences
                diff = differences[penalty][new_alg_name]
                # Ignore the first 3 instances
                diff = diff.iloc[3:]
                # Unscaled differences
                diff_unscaled = diff.copy()
                diff_unscaled[:] = scaler.inverse_transform(diff_unscaled)
                diff_eqi = diff[diff.columns[-5:]]
                diff_eqi = diff_eqi.rename(columns={col: col.split("_")[0].capitalize() for col in diff_eqi.columns})*-1
                diff_eqi = diff_eqi[domains_capital]
                diff_eqi = diff_eqi.rename(columns={col : col if col != "Sociod" else "Sociodemographic" for col in diff_eqi.columns})
                # Plot in box plot the five last differences
                sns.boxplot(data=diff_eqi, ax=ax[i//2, i%2], palette=get_var_palette(), showfliers = False)
                ax[i//2, i%2].set_title("Penalty " + str(penalty))
                #ax[i // 2, i % 2].set_ylim(-3, 3)
                # Label with the median
                label_with_medians(diff_eqi, ax[i//2, i%2])

                # Label with the number of instances
                ax[i//2, i%2].text(0.5, 0.95, "n = " + str(len(diff_eqi)), horizontalalignment='center',
                                      verticalalignment='center')


                # Obtain also the CDD for the EQIs
                fbh = friedman_posthoc(diff_eqi)
                sp.critical_difference_diagram(fbh["summary_ranks"], fbh["p_adjusted"],
                                                  ax=ax_cdd[i // 2, i % 2], color_palette=get_var_palette())
                ax_cdd[i // 2, i % 2].set_title("Penalty " + str(penalty))



                # Obtain boxplot for the 5 variables that change the most for each domain
                fig_vars, ax_vars = plt.subplots(3, 2, figsize=(12, 10))
                fig_nx, ax_nx = plt.subplots(3,2, figsize=(16,10))
                fig_joint, ax_joint = plt.subplots(5,2, figsize=(12,13))
                domains = list(var_types.keys())
                for j,domain in enumerate(domains):
                    domain_vars = var_types[domain]
                    domain_print = domain
                    if domain == "Sociodemographic":
                        domain = "Sociod"
                    color_dom = get_var_palette()[domain]
                    diff_domain = diff[[col for col in diff.columns if col in domain_vars]]
                    selected_vars = diff_domain.median().abs().sort_values(ascending=False).head(5).index
                    #print(selected_vars)
                    diff_domain_unscaled = diff_unscaled[selected_vars]
                    diff_domain = diff_domain[selected_vars]
                    flierprops = dict(marker='o', color='black', markersize=5)
                    sns.boxplot(data=diff_domain, ax=ax_vars[j//2, j%2], flierprops = flierprops,
                                color = color_dom)
                    ax_vars[j//2, j%2].set_title("Domain " + domain_print)
                    ax_vars[j // 2, j % 2].spines[['right', 'top']].set_visible(False)
                    # Tilt the x axis
                    for tick in ax_vars[j//2, j%2].get_xticklabels():
                        tick.set_rotation(20)
                        tick.set_fontsize(14)
                    #label_with_medians(diff_domain_unscaled, ax_vars[j//2, j%2])

                    # Plot also the subgraph of the domain
                    domain_nodes = list(selected_vars) + ["EQI " + domain]
                    domain_graph = graph.subgraph(domain_nodes)
                    pos = nx.drawing.nx_agraph.graphviz_layout(domain_graph, prog="dot")
                    edge_weights_sub = np.array(list(nx.get_edge_attributes(domain_graph, "weight").values()))*3
                    edge_colors_sub = list(nx.get_edge_attributes(domain_graph, "color").values())
                    # Set also a title for the graph
                    nx.draw(domain_graph, pos, with_labels=True, width = edge_weights_sub, edge_color = edge_colors_sub, node_size=1600,
                            font_size=14, font_weight="bold", ax=ax_nx[j//2, j%2], node_color = color_dom)
                    ax_nx[j // 2, j % 2].margins(0.2)

                    # Repeat for the joint plot
                    sns.boxplot(data=diff_domain, ax=ax_joint[j, 0], showfliers=False, color=color_dom)
                    ax_joint[j, 0].set_title("Domain " + domain_print)
                    ax_joint[j, 0].spines[['right', 'top']].set_visible(False)
                    # Tilt the x axis
                    for tick in ax_joint[j, 0].get_xticklabels():
                        tick.set_rotation(18)
                        tick.set_fontsize(11)

                    # Plot also the subgraph of the domain
                    nx.draw(domain_graph, pos, with_labels=True, width=edge_weights_sub, edge_color=edge_colors_sub,
                            node_size=800,
                            font_size=14, font_weight="bold", ax=ax_joint[j, 1], node_color=color_dom)
                    # Add margins
                    ax_joint[j, 1].margins(0.2)

                fig_vars.tight_layout()
                fig_vars.suptitle("Algorithm " + str(new_alg_name) + " with penalty " + str(penalty))
                fig_vars.savefig(os.path.join(penalty_dir, new_alg_name+"_variables.pdf"))
                fig_vars.clf()
                fig_nx.tight_layout()
                fig_nx.suptitle("Algorithm " + str(new_alg_name) + " with penalty " + str(penalty))
                fig_nx.savefig(os.path.join(penalty_dir, new_alg_name+"_graphvars.pdf"))
                fig_nx.clf()
                fig_joint.tight_layout()
                fig_joint.suptitle("Algorithm " + str(new_alg_name) + " with penalty " + str(penalty))
                fig_joint.savefig(os.path.join(penalty_dir, new_alg_name+"_joint.pdf"))
                plt.close()
            fig.suptitle("Algorithm " + alg_gen)
            fig.savefig(os.path.join(plots_dir, "eqi_" + alg_gen + ".pdf"))
            fig_cdd.suptitle("Algorithm " + alg_gen)
            fig_cdd.savefig(os.path.join(plots_dir, "eqi_" + alg_gen + "_cdd.pdf"))
            plt.close()

        # Repeat experiments, but segregatev by RUCC. The RUCC is indicated in the metadata
        for penalty in penalty_list:
            penalty_dir = os.path.join(plots_dir, "penalty_" + str(penalty))
            for new_alg_name, gen_alg in zip(differences[penalty].keys(), generic_algorithms):
                fig, ax = plt.subplots(2, 2, figsize=(10, 10))
                fig_line_rucc = plt.figure(figsize=(6,4*0.8))
                medians_df = pd.DataFrame(index=range(1,4+1), columns=[col.split("_")[0].capitalize() for col in sp_eqi_vars[:-1]])
                for i in range(4):
                    rucc = i+1
                    # Get the differences
                    diff = differences[penalty][new_alg_name]
                    # Ignore the first 3 instances
                    diff = diff.iloc[3:]
                    # Leave in the diff only the instances with the same RUCC
                    # First, create a copy of the metadata only with the instances in diff
                    metadata_diff = metadata.loc[diff.index]
                    # Indexes of the metadata_diff with the RUCC = rucc
                    indexes_rucc = metadata_diff[metadata_diff["cat_RUCC"] == rucc].index
                    diff = diff.loc[indexes_rucc]
                    # Plot in box plot the five last differences
                    diff_eqi = diff[diff.columns[-5:]]
                    #Drop nas
                    diff_eqi = diff_eqi.dropna()
                    #print("RUCC " + str(rucc) + " for algorithm " + new_alg_name + " with penalty " + str(penalty) + " has " + str(len(diff_eqi)) + " instances")
                    #print(diff_eqi)
                    diff_eqi = diff_eqi.rename(columns={col: col.split("_")[0].capitalize() for col in diff_eqi.columns})*-1
                    #print(diff_eqi)
                    diff_eqi = diff_eqi[domains_capital]
                    #print(diff_eqi)
                    #raise ValueError("Stop here")
                    sns.boxplot(data=diff_eqi, ax=ax[i//2, i%2], palette=get_var_palette())
                    ax[i // 2, i % 2].spines[['right', 'top']].set_visible(False)
                    medians = label_with_medians(diff_eqi, ax[i//2, i%2])
                    # Label with the number of instances
                    ax[i // 2, i % 2].text(0.5, 0.95, "n = " + str(len(diff_eqi)), horizontalalignment='center',
                                           verticalalignment='center')
                    medians_df.loc[rucc] = medians
                    ax[i//2, i%2].set_title(f"Category RUCC {rucc}")
                    # Fix axis between -3 and 3
                    ax[i//2, i%2].set_ylim(-3, 3)
                fig.suptitle(f"Differences for {new_alg_name}, penalty {penalty}")
                fig.savefig(os.path.join(penalty_dir, gen_alg, new_alg_name+"_rucc.pdf"))
                # Create line plot of the medians
                ax_line = fig_line_rucc.gca()
                for domain_name in domains_capital:
                    subset = medians_df[domain_name]
                    color_dom = get_var_palette()[domain_name]
                    sns.lineplot(data=subset, label=domain_name, marker="o", color = color_dom, ax=ax_line)
                # Label with the number of instances
                ax_line.text(0.5, 0.95, "n = " + str(len(diff_eqi)), horizontalalignment='center',
                                       verticalalignment='center')
                ax_line.set_title(f"Counterfactual proposals")
                ax_line.set_xlabel("RUCC")
                ax_line.set_ylabel("Median EQI change")
                fig_line_rucc.savefig(os.path.join(penalty_dir, gen_alg, new_alg_name+"_rucc_line.pdf"))
                plt.close()
                fig, ax = plt.subplots(3, 2, figsize=(10, 10))
                fig_line_class = plt.figure(figsize=(6,4*0.8))
                medians_df = pd.DataFrame(index=range(2,7), columns=[col.split("_")[0].capitalize() for col in sp_eqi_vars[:-1]])
                for i in range(5):
                    class_label = i+2
                    # Get the differences
                    diff = differences[penalty][new_alg_name]
                    # Ignore the first 3 instances
                    diff = diff.iloc[3:]
                    # Leave in the diff only the instances with the same class label
                    # First, create a copy of the metadata only with the instances in diff
                    data_copy = data.loc[diff.index]
                    # Indexes of the metadata_diff with the class label = class_label
                    indexes_class = data_copy[data_copy["EQI"] == class_label].index
                    diff = diff.loc[indexes_class]
                    # Plot in box plot the five last differences
                    diff_eqi = diff[diff.columns[-5:]]
                    diff_eqi = diff_eqi.rename(columns={col: col.split("_")[0].capitalize() for col in diff_eqi.columns})
                    diff_eqi = diff_eqi.dropna()*-1
                    if len(diff_eqi) == 0:
                        continue
                    diff_eqi = diff_eqi[domains_capital]
                    sns.boxplot(data=diff_eqi, ax=ax[i // 2, i % 2], palette=get_var_palette())
                    medians = label_with_medians(diff_eqi, ax[i // 2, i % 2])
                    ax[i // 2, i % 2].set_title(f"Class {class_label}")
                    ax[i // 2, i % 2].spines[['right', 'top']].set_visible(False)
                    ax[i // 2, i % 2].set_ylim(-3, 3)
                    print(medians_df.loc[class_label])
                    print(medians)
                    medians_df.loc[class_label] = medians
                fig.suptitle(f"Differences for {new_alg_name}, penalty {penalty}")
                fig.savefig(os.path.join(penalty_dir, gen_alg, new_alg_name + "_class.pdf"))
                ax_line = fig_line_class.gca()
                for domain_name in domains_capital:
                    subset = medians_df[domain_name]
                    color_dom = get_var_palette()[domain_name]
                    sns.lineplot(data=subset, label=domain_name, marker="o", color = color_dom, ax = ax_line)
                # Label with the number of instances
                ax_line.text(0.5, 0.95, "n = " + str(len(diff_eqi)), horizontalalignment='center',
                                       verticalalignment='center')
                ax_line.set_title(f"Counterfactual proposals")
                ax_line.set_xlabel("EQI category")
                ax_line.set_ylabel("Median EQI change")
                fig_line_class.savefig(os.path.join(penalty_dir, gen_alg, new_alg_name + "_class_line.pdf"))
                plt.close()
    #raise ValueError("Stop here")
    # Analyse 3 first instances, separately
    all_alg_diffs = []
    for i in range(3):
        index_i = differences[penalty]["FACE"].index[i]
        county_name = metadata.loc[index_i]["County_Name"]
        print("Analysis county", county_name)
        diff_res = pd.DataFrame(index = penalty_list, columns=["FACE", "Wachter", "BayesACE"])
        spars_res = diff_res.copy()
        pll_res = diff_res.copy()
        alg_diffs = pd.DataFrame(index=eqis, columns = [alg + " Pen-" + str(penalty) for alg in ["FACE", "Wachter", "BayesACE"] for penalty in penalty_list])
        for penalty in penalty_list:
            print("\tPenalty", penalty)
            all_eqi_diffs = pd.DataFrame(index = ["BayesACE", "FACE", "Wachter"], columns = eqis)
            for gen_alg_name,new_alg_name in zip(["FACE", "Wachter", "BayesACE", "BayesACE-FACE"],differences[penalty].keys()):
                if new_alg_name == "BayesACE - FACE":
                    continue
                print("\t\tAlgorithm", new_alg_name)
                diff = differences[penalty][new_alg_name]
                diff_unscaled = diff.copy()
                diff_unscaled = diff_unscaled.iloc[:3]
                diff_unscaled[:] = scaler.inverse_transform(diff_unscaled)
                diff_i = diff.loc[index_i]
                # Only the last 5 EQIs
                diff_i = diff_i[-5:]*-1
                all_eqi_diffs.loc[gen_alg_name] = diff_i
                print("\t\t\tDifferences")
                alg_diffs[gen_alg_name + " Pen-"+str(penalty)] = diff_i.to_numpy().round(2)
                print("Sum of differences", (diff_i**2).sum())
                diff_res.loc[penalty, gen_alg_name] = round((diff_i**2).sum(),2)
                print("Sparsity", (diff_i.abs() > 0.25).sum() / len(diff_i))
                spars_res.loc[penalty, gen_alg_name] = (diff_i.abs() > 0.25).sum() / len(diff_i)
                # Compute sum of differences over the neural net
                results_bank_i = results_bank[penalty][new_alg_name][i]
                pll = path_likelihood_length(path=results_bank_i.path.reset_index(drop=True), density_estimator=bn, penalty=penalty)
                print("Path likelihood length", pll**(1/penalty))
                pll_res.loc[penalty, gen_alg_name] = round(pll**(1/penalty),2)
                # For each domain, get and plot the subgraphs with the 5 most changed vars per domain
                fig_nx, ax_nx = plt.subplots(3, 2, figsize=(10, 10))
                domains = list(var_types.keys())
                for j,domain in enumerate(domains) :
                    domain_vars = var_types[domain]
                    diff_domain = diff[domain_vars]
                    selected_vars = diff_domain.loc[index_i].sort_values(ascending=False, key=abs).head(10).index
                    if domain == "Sociodemographic":
                        domain = "Sociod"
                    color_dom = get_var_palette()[domain]
                    domain_nodes = list(selected_vars) + ["EQI " + domain]
                    domain_graph = graph.subgraph(domain_nodes)
                    pos = nx.drawing.nx_agraph.graphviz_layout(domain_graph, prog="dot")
                    edge_weights_sub = np.array(list(nx.get_edge_attributes(domain_graph, "weight").values())) * 1.5
                    edge_colors_sub = list(nx.get_edge_attributes(domain_graph, "color").values())
                    # Set also a title for the graph
                    nx.draw(domain_graph, pos, with_labels=True, width=edge_weights_sub, edge_color=edge_colors_sub, node_size=300,
                            font_size=14, font_weight="bold", ax=ax_nx[j // 2, j % 2], node_color = color_dom)
                    # Save also changes in those variables, unscaled
                    diff_unscaled_i = diff_unscaled.loc[index_i]
                    diff_unscaled_i = diff_unscaled_i[selected_vars]
                    diff_unscaled_i = diff_unscaled_i.round(2)
                    # Save it to a file
                    if not os.path.exists(os.path.join(plots_dir, "practical", "vars", "penalty_"+str(penalty), gen_alg_name)):
                        os.makedirs(os.path.join(plots_dir, "practical","vars", "penalty_"+str(penalty), gen_alg_name))
                    diff_unscaled_i.to_csv(os.path.join(plots_dir, "practical", "vars", "penalty_"+str(penalty), gen_alg_name, "county_"+county_name+"_domain_"+domain+".csv"))

                fig_nx.tight_layout()
                fig_nx.suptitle("Algorithm " + str(new_alg_name) + " with penalty " + str(penalty) + " for county " + county_name)
                #fig_nx.show()
                #fig_nx.clf()

            # Bar plot
            # Trim EQI names
            all_eqi_diffs = all_eqi_diffs.rename(columns={col: col.split("_")[0].capitalize() for col in all_eqi_diffs.columns})
            # Reorder EQIs
            all_eqi_diffs = all_eqi_diffs[domains_capital]
            # Set to 0 values between -0.25 and 0.25
            all_eqi_diffs = all_eqi_diffs.where(all_eqi_diffs.abs() > 0.26, 0.1)
            fig_bar = plt.figure(figsize=(8, 2.9))
            ax_bar = fig_bar.gca()
            bar_width = 0.25
            x = np.arange(all_eqi_diffs.shape[1])
            bars = []
            for i, (label, row) in enumerate(all_eqi_diffs.iterrows()):
                bar_group = ax_bar.bar(x + (i - 1) * bar_width, row, width=bar_width, color=get_palette(None)[label], label=label)
                bars.append(bar_group)

            # Add value labels
            for bar_group in bars:
                for bar in bar_group:
                    height = bar.get_height()
                    print_height = height
                    if abs(print_height) < 0.11:
                        print_height = 0
                    # Choose offset based on positive or negative
                    offset = 0.03 if height >= 0 else -0.03
                    va = 'bottom' if height >= 0 else 'top'
                    ax_bar.text(bar.get_x() + bar.get_width() / 2, height + offset,
                            f'{print_height:.1f}', ha='center', fontsize=7, va=va)

            # Set x-axis ticks to the center of each group
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels([i.capitalize() for i in all_eqi_diffs.columns])

            # Aesthetics
            ax_bar.set_xlabel('Domain-specific EQI')
            ax_bar.set_ylabel('Actionable\n recourse')
            ax_bar.set_title('Differences for ' + county_name)
            ax_bar.legend(fontsize = 8)
            ax_bar.axhline(0, color='black', linewidth=1)

            # Increase y-lims by 0.1
            ylim = ax_bar.get_ylim()
            if ylim[0] < 0:
                ax_bar.set_ylim(ylim[0] - 0.25, ylim[1] + 0.25)
            else :
                ax_bar.set_ylim(ylim[0], ylim[1] + 0.25)
            fig_bar.tight_layout()
            fig_bar.show()
            fig_bar.savefig(os.path.join(plots_dir, "practical", "bar_" + county_name + ".pdf"))


            # Save the results
            if not os.path.exists(os.path.join(plots_dir, "practical")):
                os.makedirs(os.path.join(plots_dir, "practical"))
            diff_res.to_csv(os.path.join(plots_dir, "practical", "county_"+county_name+"_diff.csv"))
            spars_res.to_csv(os.path.join(plots_dir,"practical", "county_"+county_name+"_spars.csv"))
            pll_res.to_csv(os.path.join(plots_dir,"practical", "county_"+county_name+"_pll.csv"))
            alg_diffs.to_csv(os.path.join(plots_dir,"practical", "county_"+county_name+"_eqis_diff.csv"))

        all_alg_diffs.append(alg_diffs)

    # Define the explicit order for Algorithms
    algo_order = ["BayesACE","FACE", "Wachter"]

    # Reorder all_alg_diffs
    new_alg_diffs = [all_alg_diffs[1], all_alg_diffs[2], all_alg_diffs[0]]
    all_alg_diffs = new_alg_diffs

    # Iterate through each penalty
    for penalty in penalty_list:
        print(f"Creating Transposed Summary Heatmap for Penalty: {penalty}")

        county_frames = []
        county_names = ["New York county", "Los Angeles county", "Conejos county"]  # Default names in case metadata lookup fails

        # Iterate through the 3 counties (stored in all_alg_diffs)
        for i in range(3):
            # 1. Get Data & Name
            df_county = all_alg_diffs[i].copy()

            # 2. Filter Columns for current Penalty
            # Current cols are like "FACE Pen-0.1", "Wachter Pen-0.1"
            cols_to_keep = [c for c in df_county.columns if f"Pen-{str(penalty)}" in c]
            subset = df_county[cols_to_keep].copy()

            # 3. Clean Column Names (Remove " Pen-0.1")
            subset.columns = [c.split(" Pen-")[0] for c in subset.columns]

            # 4. Enforce Algorithm Order (Columns)
            # Check which of the standard algos are present and reorder
            present_algos = [alg for alg in algo_order if alg in subset.columns]
            subset = subset[present_algos]

            # Threshold small absolute values (-0.25, 0.25) to 0.0 for better visualization
            #subset = subset.where(subset.abs() > 0.26, 0)

            # 5. Enforce EQI Order (Rows) & Capitalize
            # First, ensure index is capitalized to match domains_capital
            subset.index = [idx.split("_")[0].capitalize() for idx in subset.index]
            # Reindex to force the specific order from the barplot
            subset = subset.reindex(domains_capital)

            county_frames.append(subset)

        # 6. Concatenate Side-by-Side
        # Result: Rows=EQIs, Cols=MultiIndex(County, Algorithm)
        df_viz = pd.concat(county_frames, axis=1, keys=county_names)

        # --- Plotting ---
        plt.figure(figsize=(12*0.8, 5*0.8))  # Wider figure to fit 3 counties side-by-side
        ax = plt.gca()

        # Custom Palette: Red (Negative) <-> White (0) <-> Blue (Positive)
        cmap = sns.diverging_palette(10, 240, as_cmap=True, s=90, l=45, sep=1)

        # Plot Heatmap
        sns.heatmap(df_viz,
                    annot=True, fmt=".2f",
                    cmap=cmap, center=0,
                    cbar_kws={'label': 'Actionable recourse', 'shrink': 0.8},
                    linewidths=1, linecolor='white',
                    ax=ax)

        # --- Aesthetics ---
        #ax.set_title(f'Impact on EQIs (Penalty {penalty})', fontsize=16, pad=20)
        ax.set_ylabel('EQI Domain', fontsize=12)
        ax.set_xlabel('')

        # Clean up X-axis labels
        # Default is "(County, Algo)". We want just "Algo" on bottom, and "County" on top
        # 1. Set bottom labels to just the Algorithm names
        ax.set_xticklabels([label.get_text().split('-')[1] for label in ax.get_xticklabels()], rotation=0)

        # 2. Add County Labels at the top
        # We place text centered over each group of 3 columns
        num_algos = len(algo_order)
        for idx, name in enumerate(county_names):
            # Calculate center position: (start_index + end_index) / 2
            center_pos = (idx * num_algos) + (num_algos / 2)
            ax.text(center_pos, -0.05, name,
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold', color='#333333')

            # Add thick vertical line to separate counties (except after the last one)
            if idx < len(county_names):
                ax.axvline((idx + 1) * num_algos, color='black', linewidth=3)

        plt.tight_layout()

        # Save
        if not os.path.exists(os.path.join(plots_dir, "practical")):
            os.makedirs(os.path.join(plots_dir, "practical"))
        save_path = os.path.join(plots_dir, "practical", f"heatmap_transposed_penalty_{penalty}.pdf")
        plt.savefig(save_path, bbox_inches='tight')
        plt.tight_layout()
        plt.show()
        plt.close()