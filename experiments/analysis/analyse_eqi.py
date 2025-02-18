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
from experiments.utils import friedman_posthoc

# Experiments:

# Seggregate by RUCC, class label and both and analyse:
#      - Which EQI is changed the most
#      - Which variables (of each domain) are changed the most
#      - Compare with Wachter algorithm


if __name__ == "__main__":
    root_dir = "../results/exp_eqi/"
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
    df_train = pd.read_csv(os.path.join(data_dir, "data_train.csv"), index_col=0)
    df_test = pd.read_csv(os.path.join(data_dir, "data_test.csv"), index_col=0)
    data = pd.concat([df_train, df_test], axis=0)
    var_types = pickle.load(open(os.path.join(data_dir, "var_types.pkl"), "rb"))

    # Load the Bayesian network
    with open(os.path.join(model_dir, "bn_restricted_lim_arcs.pkl"), "rb") as f:
        bn = pickle.load(f)

    betas = {}
    var_edges = {}
    for node in bn.bayesian_network.nodes():
        cpd = bn.bayesian_network.cpd(node)
        if "EQI" in node and "EQI" != node:
            node_name = node.split("_")[0]
            betas[node_name] = {}
            if len(cpd.evidence()) > 0:
                print(cpd)
                parents = cpd.evidence()
                parents.remove("EQI")
                parents = ["EQI"] + parents
                parents = [parent.split("_")[0] for parent in parents]
                coefs_df = pd.DataFrame(columns=parents, index=range(7))
                for i in range(7):
                    cpd_i = cpd.conditional_factor(pb.Assignment({"EQI" : str(i)}))
                    coefs_df.loc[i] = cpd_i.beta
                # Compute differential of the first column
                coefs_df["EQI"] = coefs_df["EQI"].diff()
                betas[node_name] = coefs_df.mean(axis=0).to_dict()
        elif "EQI" != node:
            for i,parent in enumerate(cpd.evidence()):
                var_edges[(parent, node)] = cpd.beta[i+1]
                print(parent, "->", node, cpd.beta[i+1])

    print(betas)
    edge_list = []
    edge_weights = []
    for i in betas.keys():
        for j in betas[i].keys():
            edge_list.append((j,i))
            edge_weights.append(betas[i][j])

    graph = nx.DiGraph()
    # Colour in green positive arcs, in red negative arcs. Strenght of the colour depends on the absolute value of the coefficient
    edge_colors = []
    for weight in edge_weights:
        if weight > 0:
            edge_colors.append("green")
        else:
            edge_colors.append("red")
    edge_weights = [abs(weight)*6 for weight in edge_weights]
    for edge, weight, color in zip(edge_list, edge_weights, edge_colors):
        graph.add_edges_from([edge], weight=weight, color=color)
    pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog="dot")
    edge_weights = list(nx.get_edge_attributes(graph, "weight").values())
    edge_colors = list(nx.get_edge_attributes(graph, "color").values())
    nx.draw(graph, pos, with_labels=True, width = edge_weights, edge_color = edge_colors, node_size=1000,
            font_size=8, font_weight="bold")
    plt.show()


    # Print CPDs that include the characters "EQI"
    for node in bn.bayesian_network.nodes() :
        if "EQI" in node :
            print(bn.bayesian_network.cpd(node))

    # Load the normalizing flow model
    with open(os.path.join(model_dir, "nf.pkl"), "rb") as f:
        nf = pickle.load(f)

    vertices_list = [0, 1, 2]
    penalty_list = [1,5,10,15]

    # Import the algorithms, results, differences and distances
    algorithms = {}
    results_bank = {}
    differences = {}
    distances = {}

    alg_names = ["face"] + ["wachter"] + ["bayesace_"+str(i) for i in vertices_list]
    new_alg_names = ["FACE"] + ["Wachter"] + [f"BayesACE {i} vertices" for i in vertices_list]

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
            bayesace_distances[alg_name] = pd.read_csv(os.path.join(results_dir, "distances_"+alg_name+".csv"), index_col=0).to_numpy()[0]
        bayesace_distances = pd.DataFrame(bayesace_distances)

        # Check using Friedman test which algorithm is the best
        f_bh_result = friedman_posthoc(bayesace_distances)

        # Select the one with the lowest ranking
        best_alg = f_bh_result["summary_ranks"].idxmin()
        selected_algs = alg_names_penalty[:2] + [best_alg]

        name_map = {}

        for new_alg_name,alg_name in zip(new_alg_names,alg_names_penalty):
            if not alg_name in selected_algs:
                continue
            name_map[alg_name] = new_alg_name
            alg_path = os.path.join(algorithms_dir, alg_name + ".pkl")
            #algorithms[penalty][new_alg_name] = pickle.load(open(alg_path, "rb"))
            differences[penalty][new_alg_name] = pd.read_csv(os.path.join(results_dir, "diff_"+alg_name+".csv"), index_col=0)
            distances[penalty][new_alg_name] = pd.read_csv(os.path.join(results_dir, "distances_"+alg_name+".csv"), index_col=0)
            results_bank_dir = os.path.join(results_dir, alg_name)
            results_bank[penalty][new_alg_name] = {}
            for i in range(len(os.listdir(results_bank_dir))):
                file_i = str(i) + ".pkl"
                instance_path = os.path.join(results_bank_dir, file_i)
                results_bank[penalty][new_alg_name][i] = pickle.load(open(instance_path, "rb"))
        # Add as a new key the difference between the best BayesACE and FACE
        differences[penalty]["BayesACE - FACE"] = (differences[penalty][name_map[best_alg]] - differences[penalty][name_map["face_"+str(penalty)]]).abs()
        distances[penalty]["BayesACE - FACE"] = (distances[penalty][name_map[best_alg]] - distances[penalty][name_map["face_"+str(penalty)]]).abs()

    generic_exp = False
    if generic_exp:
        # Do an experiment for each penalty
        for alg_i in range(3):
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            for i, penalty in enumerate(penalty_list):
                penalty_dir = os.path.join(plots_dir, "penalty_" + str(penalty))
                new_alg_name = list(differences[penalty].keys())[alg_i]
                # Get the differences
                diff = differences[penalty][new_alg_name]
                # Ignore the first 3 instances
                diff = diff.iloc[3:]
                diff_eqi = diff[diff.columns[-5:]]
                diff_eqi = diff_eqi.rename(columns={col: col.split("_")[0] for col in diff_eqi.columns})
                # Plot in box plot the five last differences
                sns.boxplot(data=diff_eqi, ax=ax[i//2, i%2])
                ax[i//2, i%2].set_title("Penalty " + str(penalty))
                ax[i // 2, i % 2].set_ylim(-3, 3)

                # Obtain boxplot for the 5 variables that change the most for each domain
                fig_vars, ax_vars = plt.subplots(3, 2, figsize=(10, 10))
                domains = list(var_types.keys())
                domains.remove("ID")
                for j,domain in enumerate(domains):
                    domain_vars = var_types[domain]
                    diff_domain = diff[[col for col in diff.columns if col in domain_vars]]
                    selected_vars = diff_domain.mean().sort_values(ascending=False).head(5).index
                    #print(selected_vars)
                    diff_domain = diff_domain[selected_vars]
                    sns.boxplot(data=diff_domain, ax=ax_vars[j//2, j%2])
                    ax_vars[j//2, j%2].set_title("Domain " + domain)
                    # Tilt the x axis
                    for tick in ax_vars[j//2, j%2].get_xticklabels():
                        tick.set_rotation(30)
                fig_vars.tight_layout()
                fig_vars.suptitle("Algorithm " + str(new_alg_name) + " with penalty " + str(penalty))
                fig_vars.savefig(os.path.join(penalty_dir, new_alg_name+"_variables.pdf"))
                fig_vars.clf()
            fig.suptitle("Algorithm " + str(alg_i))
            fig.savefig(os.path.join(plots_dir, "eqi.pdf"))
            plt.close()

        # Repeat experiments, but segregatev by RUCC. The RUCC is indicated in the metadata
        for penalty in penalty_list:
            penalty_dir = os.path.join(plots_dir, "penalty_" + str(penalty))
            for new_alg_name in differences[penalty].keys():
                fig, ax = plt.subplots(2, 2, figsize=(10, 10))
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
                    diff_eqi = diff_eqi.rename(columns={col: col.split("_")[0] for col in diff_eqi.columns})
                    sns.boxplot(data=diff_eqi, ax=ax[i//2, i%2])
                    ax[i//2, i%2].set_title(f"Category RUCC {rucc}")
                    # Fix axis between -3 and 3
                    ax[i//2, i%2].set_ylim(-3, 3)
                fig.suptitle(f"Differences for {new_alg_name}, penalty {penalty}")
                fig.savefig(os.path.join(penalty_dir, new_alg_name+"_rucc.pdf"))
                plt.close()
                fig, ax = plt.subplots(3, 2, figsize=(10, 10))
                for i in range(5):
                    class_label = i
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
                    diff_eqi = diff_eqi.rename(columns={col: col.split("_")[0] for col in diff_eqi.columns})
                    sns.boxplot(data=diff_eqi, ax=ax[i // 2, i % 2])
                    ax[i // 2, i % 2].set_title(f"Class {class_label}")
                    ax[i // 2, i % 2].set_ylim(-3, 3)
                fig.suptitle(f"Differences for {new_alg_name}, penalty {penalty}")
                fig.savefig(os.path.join(penalty_dir, new_alg_name + "_class.pdf"))
                plt.close()


    # Analyse 3 first instances, separately
    for i in range(3):
        index_i = differences[1]["FACE"].index[i]
        county_name = metadata.loc[index_i]["County_Name"]
        print("Analysis county", county_name)
        for penalty in [1,5,10,15]:
            print("\tPenalty", penalty)
            for new_alg_name in differences[penalty].keys():
                if new_alg_name == "BayesACE - FACE":
                    continue
                print("\t\tAlgorithm", new_alg_name)
                diff = differences[penalty][new_alg_name]
                diff_i = diff.loc[index_i]
                # Only the last 5 EQIs
                diff_i = diff_i[-5:]
                print("\t\t\tDifferences")
                print(diff_i)
                print("Sum of differences", diff_i.abs().sum())
                # Compute sum of differences over the neural net
                results_bank_i = results_bank[penalty][new_alg_name][i]
                pll = path_likelihood_length(path=results_bank_i.path.reset_index(drop=True), density_estimator=nf, penalty=penalty)
                print("Path likelihood length", pll**(1/penalty))
        print()
        print()



