import os
import time
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
import re

from experiments.utils import friedman_posthoc

from experiments.experiment2 import FACE_BASELINE, FACE_KDE, FACE_EPS, WACHTER, BAYESACE

# Path to dataset root
root_dir = "../results/exp_2/"

# Wilcoxon test alternative hypothesis
wx_alt = ["two-sided", "greater", "less"]

# Regex to match filenames like distances_data44123_pen1.csv
file_pattern = re.compile(r"distances_data(\d+)_penalty(\d+)\.csv")

# New names dictionary
new_names = {FACE_BASELINE: "FACE ground-truth",
             FACE_KDE: "FACE KDE",
             FACE_EPS: "FACE ε",
             WACHTER: "Wachter"}

# This part is hard coded, might be changed in the future
for vertices in [0, 1, 2, 3]:
    new_names[BAYESACE + "_" + "clg" + "_v" + str(vertices)] = "BayesACE" + " " + str(vertices) + " vertices"

for vertices in [0, 1, 2, 3]:
    new_names[BAYESACE + "_" + "nf" + "_v" + str(vertices)] = "DAACE" + " " + str(vertices) + " vertices"

for vertices in [0, 1, 2, 3]:
    new_names[BAYESACE + "_" + "gt" + "_v" + str(vertices)] = "DAACE (ground-truth)" + " " + str(vertices) + " vertices"


# Get all the values for penalty, dataset_id, models and n_vertex
def list_subdir(dir, exclude=[]):
    return_list = []
    for element in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, element)) and element not in exclude:
            return_list.append(element)
    return return_list


def get_values(root_dir):
    # Get datasets ids
    dataset_ids = list_subdir(root_dir, exclude=["plots"])

    # Get the list of penalties
    dataset_path = os.path.join(root_dir, dataset_ids[0])
    penalties = list_subdir(dataset_path)

    # Get the list of metrics
    penalty_path = os.path.join(dataset_path, penalties[0])
    metrics = list_subdir(penalty_path, exclude=["algorithms", "counterfactual", "paths"])

    # Get the list of algorithms
    algorithm_path = os.path.join(penalty_path, "algorithms")
    algorithms = list_subdir(algorithm_path)
    # Remove the .pkl
    algorithms = [algorithm.split(".")[0] for algorithm in algorithms]

    # Get the list of likelihood and post_prob thresholds
    metric_path = os.path.join(penalty_path, metrics[0])
    likelihoods = []
    post_probs = []
    file_pattern_mod = re.compile(r"likelihood(-?[\d.]+)_pp(-?[\d.]+)\.csv")
    for file in os.listdir(metric_path):
        match = file_pattern_mod.match(file)
        if match:
            # Extract dataset_id and penalty from filename
            likelihood = match.group(1)
            post_prob = match.group(2)
            likelihoods.append(likelihood)
            post_probs.append(post_prob)
    likelihoods = list(set(likelihoods))
    post_probs = list(set(post_probs))

    # Order the likelihoods and post_probs
    likelihoods = sorted(likelihoods, key=lambda x: float(x))
    post_probs = sorted(post_probs, key=lambda x: float(x))

    return algorithms, metrics,{"dataset_ids": dataset_ids, "penalties": penalties, "likelihoods": likelihoods,
            "post_probs": post_probs}


# Function to load and organize data from the directory
def load_data(root_dir, metrics, values_dict):
    data_dict = {}
    for metric in metrics:
        data_dict[metric] = {}
        for dataset_id, penalty, likelihood, post_prob in product(*values_dict.values()):
            # Get the path to the file
            file_name = "likelihood" + str(likelihood) + "_pp" + str(post_prob) + ".csv"
            file_path = os.path.join(root_dir, dataset_id, penalty, metric, file_name)
            if os.path.exists(file_path):
                df = pd.read_csv(open(file_path), index_col=0)
                #df = df[df.columns[:7]]
                # If real_logl or real_pp, invert the sign
                if "real" in metric:
                    df = -df

                # When dealing with time, drop the rows with nans and infs
                if "time" in metric:
                    df = df.dropna()
                    df = df.replace([np.inf, -np.inf], np.nan)
                    df = df.dropna()

                # Subsitute nans for inf (a good path was not found, hence infinite distance)
                df = df.fillna(np.inf)
                data_dict[metric][(dataset_id, penalty, likelihood, post_prob)] = df
    return data_dict


def perform_bh_param(data_dict, values_dict, metric, segregate=None):
    data_dict_new = aggregate_data(data_dict, values_dict, metric, segregate)
    data_dict_new_dist = aggregate_data(data_dict, values_dict, "distance", segregate)

    data_dict_new = remove_redundant(data_dict_new, data_dict_new_dist)

    results = {}
    for combination_aggregate in data_dict_new.keys():
        # Check which BayesACE, DAACE and DAACE gt are the best performing via a Friedman test
        results[combination_aggregate] = friedman_posthoc(data_dict_new[combination_aggregate].dropna())
    return results

def get_redundant(df) :
    redundant_models = []
    for model in ["clg", "nf", "gt"]:
        models = [BAYESACE + "_" + model + "_v" + str(vertices) for vertices in range(4)]
        df_model = df[models]
        f_bh_result = friedman_posthoc(df_model.dropna())["summary_ranks"]
        # Get all the models except the best one
        best_idx = f_bh_result.idxmin()
        worst_models = [model for model in models if model != best_idx]
        redundant_models.extend(worst_models)
    return redundant_models

def remove_redundant(data_dict_new, data_dict_new_dist):
    data_dict_no_redundant = {}
    for comb in data_dict_new.keys() :
        redundant_models = get_redundant(data_dict_new_dist[comb])
        data_dict_no_redundant[comb] = data_dict_new[comb].drop(columns=redundant_models)
    return data_dict_no_redundant

def aggregate_data(data_dict, values_dict, metric, segregate=None):
    data_dict_new = {}
    if segregate is None:
        data_dict_new["total"] = pd.concat(
            [data_dict[metric][comb] for comb in
             product(*values_dict.values())]).reset_index(drop=True)
    elif len(segregate) == len(values_dict.keys()):
        for comb in product(*[values_dict[key] for key in segregate]):
            data_dict_new[comb] = data_dict[metric][comb].copy()
    else :
        # Aggregate is the difference between the list of values_dict keys and the seggregate list. But maintain the order
        aggregate = [key for key in values_dict.keys() if key not in segregate]
        print(aggregate)
        print(segregate)
        for combination_segregate in product(*[values_dict[key] for key in segregate]):
            tmp_list = []
            for combination_aggregate in product(*[values_dict[key] for key in aggregate]):
                # We cannot sum combination and combination2, we have to reorder so it complies with the order (dataset_id, penalty, likelihood, post_prob)
                reconstructed_combination = []
                for key in values_dict.keys():
                    if key in aggregate:
                        reconstructed_combination.append(combination_aggregate[aggregate.index(key)])
                    else:
                        reconstructed_combination.append(combination_segregate[segregate.index(key)])
                reconstructed_combination = tuple(reconstructed_combination)
                tmp_list.append(data_dict[metric][reconstructed_combination])
            data_dict_new[combination_segregate] = pd.concat(tmp_list).reset_index(drop=True)
    return data_dict_new


# Function to run Friedman test and Nemenyi posthoc for each dataset/model/penalty
def perform_bh_by_all(data_dict, values_dict, metric):
    results = {}
    for dataset_id, likelihood, post_prob in product(values_dict["dataset_ids"], values_dict["likelihoods"],
                                                     values_dict["post_probs"]):
        results[(dataset_id, likelihood, post_prob)] = friedman_posthoc(
            data_dict[metric][(dataset_id, likelihood, post_prob)].dropna())


def perform_bh_by_thresholds(data_dict, values_dict, metric):
    # First, we group by the data by penalty and model
    data_dict_new = {}
    for likelihood, post_prob in product(values_dict["likelihoods"], values_dict["post_probs"]):
        data_dict_new[likelihood, post_prob] = pd.concat(
            [data_dict[metric][(dataset_id, likelihood, post_prob)] for dataset_id in
             values_dict["dataset_ids"]]).reset_index(drop=True)

    results = {}
    for likelihood, post_prob in data_dict_new.keys():
        results[(likelihood, post_prob)] = friedman_posthoc(data_dict_new[(likelihood, post_prob)].dropna())
    return results


def perform_bh_by_dataset(data_dict, values_dict, metric):
    data_dict_new = {}
    for dataset_id in values_dict["dataset_ids"]:
        data_dict_new[dataset_id] = pd.concat(
            [data_dict[metric][(dataset_id, likelihood, post_prob)] for likelihood, post_prob in
             product(values_dict["likelihoods"], values_dict["post_probs"])]).reset_index(drop=True)

    results = {}
    for dataset_id in data_dict_new.keys():
        results[dataset_id] = friedman_posthoc(data_dict_new[dataset_id].dropna())
    return results


def perform_bh(data_dict, values_dict, metric):
    data_dict_new = pd.concat(
        [data_dict[metric][(dataset_id, penalty, likelihood, post_prob)] for dataset_id, penalty, likelihood, post_prob in
         product(values_dict["dataset_ids"], values_dict["penalties"], values_dict["likelihoods"], values_dict["post_probs"])]).reset_index(
        drop=True)
    results = friedman_posthoc(data_dict_new.dropna())
    return results


# Run the main function when the script is executed
if __name__ == "__main__":
    # Create subfolder plots if it does not exist in the root dir
    if not os.path.exists(os.path.join(root_dir, "plots")):
        os.makedirs(os.path.join(root_dir, "plots"))

    # Get the values for dataset_id, model, penalty and n_vertex
    algorithms, metrics, values_dict = get_values(root_dir)

    print(metrics)

    # Create a subfolder for each metric
    for metric in metrics:
        if not os.path.exists(os.path.join(root_dir, "plots", metric)):
            os.makedirs(os.path.join(root_dir, "plots", metric))

    # Load the data
    data_dict = load_data(root_dir, metrics, values_dict)

    '''# Assign color to each method
    palette = {}
    for i, method in enumerate(values_dict["algorithms"]):
        if "BayesACE" in method:
            palette[method] = "blue"
        elif "FACE" in method or "Wachter" in method:
            palette[method] = "orange"
        elif "DAACE" in method:
            palette[method] = "purple"'''

    heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True,
                    'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}

    # Subplot, 2x4 for each metric
    fig, axs = plt.subplots(4, 2, figsize=(12, 12))

    '''data_dict_new = pd.concat(
        [data_dict["counterfactual"][(dataset_id, likelihood, post_prob)] for dataset_id, likelihood, post_prob in
         product(values_dict["dataset_ids"], values_dict["likelihoods"], values_dict["post_probs"])]).reset_index(
        drop=True)
    print(data_dict_new.isin([np.inf]).sum())'''

    '''for i, metric in enumerate(metrics):
        # Perform BH test for the metric distances globally
        friedman_bh_results = perform_bh(data_dict, values_dict, metric)
        sp.critical_difference_diagram(friedman_bh_results["summary_ranks"], friedman_bh_results["p_adjusted"],
                                       ax=axs[i // 2, i % 2],
                                       label_fmt_left="{label}", label_fmt_right="{label}",)
                                       #color_palette=palette)
        axs[i // 2, i % 2].set_title(f"Metric: {metric}")
        fig.tight_layout()
        # plt.title(f"Metric: {metric}")
        # plt.savefig(os.path.join(root_dir, "plots", f"bh_{metric}.pdf"), bbox_inches='tight')
        # plt.show()
        # plt.clf()
        # sp.sign_plot(friedman_bh_results["p_adjusted"], **heatmap_args)
        # plt.savefig(os.path.join(root_dir, "plots", f"heatmap_{metric}.pdf"), bbox_inches='tight')
        # plt.show()
        # plt.clf()
    fig.show()
    fig.savefig(os.path.join(root_dir, "plots", f"bh_all.pdf"), bbox_inches='tight')'''

    for i, metric in enumerate(metrics):
        # Perform BH test for the metric distances globally
        friedman_bh_results = perform_bh_param(data_dict, values_dict, metric, segregate=["dataset_ids","penalties"])[("44089","1")]
        sp.critical_difference_diagram(friedman_bh_results["summary_ranks"], friedman_bh_results["p_adjusted"],
                                       ax=axs[i // 2, i % 2],
                                       label_fmt_left="{label}", label_fmt_right="{label}",)
                                       #color_palette=palette)
        axs[i // 2, i % 2].set_title(f"Metric: {metric}")
        fig.tight_layout()
        # plt.title(f"Metric: {metric}")
        # plt.savefig(os.path.join(root_dir, "plots", f"bh_{metric}.pdf"), bbox_inches='tight')
        # plt.show()
        # plt.clf()
        # sp.sign_plot(friedman_bh_results["p_adjusted"], **heatmap_args)
        # plt.savefig(os.path.join(root_dir, "plots", f"heatmap_{metric}.pdf"), bbox_inches='tight')
        # plt.show()
        # plt.clf()
    fig.show()
    fig.savefig(os.path.join(root_dir, "plots", f"bh_all.pdf"), bbox_inches='tight')

    data_dict_new = {}
    for likelihood, post_prob in product(values_dict["likelihoods"], values_dict["post_probs"]):
        data_dict_new[likelihood, post_prob] = pd.concat(
            [data_dict["counterfactual"][(dataset_id, likelihood, post_prob)] for dataset_id in
             values_dict["dataset_ids"]]).reset_index(drop=True)

    for likelihood, post_prob in data_dict_new.keys():
        print("Likelihood: ", likelihood, "Post_prob: ", post_prob, "Inf: ",
              data_dict_new[likelihood, post_prob].isin([np.inf]).sum())

    for metric in metrics:
        '''
        # Perform BH test for the metric distances segregated by dataset_id, likelihood and post_prob
        friedman_bh_results = perform_bh_by_thresholds(data_dict, values_dict, "distances")
        for i in friedman_bh_results.keys():
            sp.critical_difference_diagram(friedman_bh_results[i]["summary_ranks"], friedman_bh_results[i]["p_adjusted"], label_fmt_left="{label} vertices", label_fmt_right="{label} vertices")
            plt.title(f"Model: {i[0]}, Penalty: {i[1]}")
            plt.show()
        
        # Perform BH test for the metric distances segregated by dataset_id
        friedman_bh_results = perform_bh_by_dataset(data_dict, values_dict, "distances")
        for i in friedman_bh_results.keys():
            sp.critical_difference_diagram(friedman_bh_results[i]["summary_ranks"], friedman_bh_results[i]["p_adjusted"], label_fmt_left="{label} vertices", label_fmt_right="{label} vertices")
            plt.title(f"Dataset: {i}")
            plt.show()'''

        # Perform BH test for the metric distances segregated by thresholds
        fig, axs = plt.subplots(4, 4, figsize=(20, 10))
        fig.suptitle(f"Metric: {metric}", fontsize=30)
        for ax, col in zip(axs[0], values_dict["post_probs"]):
            ax.set_title("PP " + col, rotation=0, fontsize=30)

        for ax, row in zip(axs[:, 0], values_dict["likelihoods"]):
            ax.set_ylabel("LL " + row, rotation=45, fontsize=30)
        friedman_bh_results = perform_bh_by_thresholds(data_dict, values_dict, metric)
        for i, key in enumerate(friedman_bh_results.keys()):
            # time.sleep(1)
            sp.critical_difference_diagram(friedman_bh_results[key]["summary_ranks"],
                                           friedman_bh_results[key]["p_adjusted"],
                                           ax=axs[i // 4, i % 4],
                                           label_fmt_left="{label}", label_fmt_right="{label}", color_palette=palette)
            axs[i // 4, i % 4].set_xlabel(f"Likelihood: {key[0]}, Post_prob: {key[1]}")
            fig.tight_layout()
            # plt.title(f"Metric: {metric}, Likelihood: {key[0]}, Post_prob: {key[1]}")
            # plt.savefig(os.path.join(root_dir, "plots", metric, f"bh_{metric}_ll{key[0]}_pp{key[1]}.pdf"), bbox_inches='tight')
            # plt.show()
            # plt.clf()
            # sp.sign_plot(friedman_bh_results[key]["p_adjusted"], **heatmap_args)
            # plt.savefig(os.path.join(root_dir, "plots", metric, f"heatmap_{metric}_ll{key[0]}_pp{key[1]}.pdf"), bbox_inches='tight')
            # plt.show()
            # plt.clf()
        fig.show()
        fig.savefig(os.path.join(root_dir, "plots", f"bh_all_{metric}.pdf"), bbox_inches='tight')
