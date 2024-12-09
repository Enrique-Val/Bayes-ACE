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
new_names = {FACE_BASELINE : "FACE ground-truth",
             FACE_KDE : "FACE KDE",
             FACE_EPS : "FACE ε",
             WACHTER : "Wachter"}

# This part is hard coded, might be changed in the future
new_names[BAYESACE+"_gt_v0"] = "DAACE ground-truth"
new_names[BAYESACE+"_nf_v0"] = "DAACE"
new_names[BAYESACE+"_clg_v0"] = "BayesACE 0-vertex"
new_names[BAYESACE+"_clg_v1"] = "BayesACE 1-vertex"



# Get all the values for penalty, dataset_id, models and n_vertex
def get_values(root_dir):
    dataset_ids = []
    metrics = []
    likelihoods = []
    post_probs = []

    # Get datasets ids
    for dataset_id in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset_id)
        if os.path.isdir(dataset_path) and dataset_id != "plots":
            dataset_ids.append(dataset_id)

    # Get the list of metrics
    dataset_path = os.path.join(root_dir, dataset_ids[0])
    for metric in os.listdir(dataset_path):
        metric_path = os.path.join(dataset_path, metric)
        if os.path.isdir(metric_path):
            metrics.append(metric)

    # Get the list of likelihood and post_prob thresholds
    metric_path = os.path.join(dataset_path, metrics[0])
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

    # Get the list of algorithms opening any file
    any_file = os.listdir(metric_path)[0]
    file_path = os.path.join(metric_path, any_file)
    df = pd.read_csv(file_path, index_col=0)
    algorithms = list(df.columns)
    algorithms.remove("bayesace_gt_v1")
    algorithms.remove("bayesace_nf_v1")
    # Rename the columns
    algorithms = [new_names[algorithm] for algorithm in algorithms]
    return {"dataset_ids": dataset_ids, "metrics": metrics, "likelihoods": likelihoods, "post_probs": post_probs, "algorithms": algorithms}



# Function to load and organize data from the directory
def load_data(root_dir, values_dict):
    data_dict = {}
    for metric in values_dict["metrics"]:
        data_dict[metric] = {}
        for dataset_id, likelihood, post_prob in product(values_dict["dataset_ids"], values_dict["likelihoods"], values_dict["post_probs"]):
            # Get the path to the file
            file_name = "likelihood"+str(likelihood)+"_pp"+str(post_prob)+".csv"
            file_path = os.path.join(root_dir, dataset_id, metric, file_name)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0)
                df = df.drop(columns=[BAYESACE+"_gt_v1",BAYESACE+"_nf_v1"])
                # Rename the columns
                df = df.rename(columns=new_names)
                # If real_logl or real_pp, invert the sign
                if "real" in metric:
                    df = -df
                # Subsitute nans for inf (a good path was not found, hence infinite distance)
                df = df.fillna(np.inf)
                data_dict[metric][(dataset_id, likelihood, post_prob)] = df
    return data_dict


# Function to run Friedman test and Nemenyi posthoc for each dataset/model/penalty
def perform_bh_by_all(data_dict, values_dict, metric):
    results = {}
    for dataset_id, likelihood, post_prob in product(values_dict["dataset_ids"], values_dict["likelihoods"], values_dict["post_probs"]):
        results[(dataset_id, likelihood, post_prob)] = friedman_posthoc(data_dict[metric][(dataset_id, likelihood, post_prob)].dropna())


def perform_bh_by_thresholds(data_dict, values_dict, metric):
    # First, we group by the data by penalty and model
    data_dict_new = {}
    for likelihood, post_prob in product(values_dict["likelihoods"], values_dict["post_probs"]):
        data_dict_new[likelihood, post_prob] = pd.concat([data_dict[metric][(dataset_id, likelihood, post_prob)] for dataset_id in values_dict["dataset_ids"]]).reset_index(drop=True)

    results = {}
    for likelihood, post_prob in data_dict_new.keys():
        results[(likelihood, post_prob)] = friedman_posthoc(data_dict_new[(likelihood, post_prob)].dropna())
    return results

def perform_bh_by_dataset(data_dict, values_dict, metric):
    data_dict_new = {}
    for dataset_id in values_dict["dataset_ids"]:
        data_dict_new[dataset_id] = pd.concat([data_dict[metric][(dataset_id, likelihood, post_prob)] for likelihood, post_prob in product(values_dict["likelihoods"], values_dict["post_probs"])]).reset_index(drop=True)

    results = {}
    for dataset_id in data_dict_new.keys():
        results[dataset_id] = friedman_posthoc(data_dict_new[dataset_id].dropna())
    return results


def perform_bh(data_dict, values_dict,metric):
    data_dict_new = pd.concat([data_dict[metric][(dataset_id, likelihood, post_prob)] for dataset_id, likelihood, post_prob in product(values_dict["dataset_ids"], values_dict["likelihoods"], values_dict["post_probs"])]).reset_index(drop=True)
    results = friedman_posthoc(data_dict_new.dropna())
    return results


# Run the main function when the script is executed
if __name__ == "__main__":
    # Create subfolder plots if it does not exist in the root dir
    if not os.path.exists(os.path.join(root_dir, "plots")):
        os.makedirs(os.path.join(root_dir, "plots"))

    # Get the values for dataset_id, model, penalty and n_vertex
    values_dict = get_values(root_dir)
    metrics = list(values_dict["metrics"])

    # Remove paths and counterfactuals from the metrics
    metrics.remove("paths")
    metrics.remove("counterfactual")
    print(metrics)

    # Create a subfolder for each metric
    for metric in metrics:
        if not os.path.exists(os.path.join(root_dir, "plots", metric)):
            os.makedirs(os.path.join(root_dir, "plots", metric))

    # Load the data
    data_dict = load_data(root_dir, values_dict)

    # Assign color to each method
    palette = {}
    for i, method in enumerate(values_dict["algorithms"]):
        if "BayesACE" in method:
            palette[method] = "blue"
        elif "FACE" in method or "Wachter" in method:
            palette[method] = "orange"
        elif "DAACE" in method:
            palette[method] = "purple"

    print(data_dict.keys())

    heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}

    # Subplot, 2x4 for each metric
    fig, axs = plt.subplots(4, 2, figsize=(12, 12))

    data_dict_new = pd.concat([data_dict["counterfactual"][(dataset_id, likelihood, post_prob)] for dataset_id, likelihood, post_prob in product(values_dict["dataset_ids"], values_dict["likelihoods"], values_dict["post_probs"])]).reset_index(drop=True)
    print(data_dict_new.isin([np.inf]).sum())

    for i,metric in enumerate(metrics):
        # Perform BH test for the metric distances globally
        friedman_bh_results = perform_bh(data_dict, values_dict, metric)
        sp.critical_difference_diagram(friedman_bh_results["summary_ranks"], friedman_bh_results["p_adjusted"],
                                       ax=axs[i//2, i%2],
                                       label_fmt_left="{label}", label_fmt_right="{label}",
                                       color_palette=palette)
        axs[i//2, i%2].set_title(f"Metric: {metric}")
        fig.tight_layout()
        #plt.title(f"Metric: {metric}")
        #plt.savefig(os.path.join(root_dir, "plots", f"bh_{metric}.pdf"), bbox_inches='tight')
        #plt.show()
        #plt.clf()
        #sp.sign_plot(friedman_bh_results["p_adjusted"], **heatmap_args)
        #plt.savefig(os.path.join(root_dir, "plots", f"heatmap_{metric}.pdf"), bbox_inches='tight')
        #plt.show()
        #plt.clf()
    fig.show()
    fig.savefig(os.path.join(root_dir, "plots", f"bh_all.pdf"), bbox_inches='tight')

    data_dict_new = {}
    for likelihood, post_prob in product(values_dict["likelihoods"], values_dict["post_probs"]):
        data_dict_new[likelihood, post_prob] = pd.concat(
            [data_dict["counterfactual"][(dataset_id, likelihood, post_prob)] for dataset_id in
             values_dict["dataset_ids"]]).reset_index(drop=True)

    for likelihood, post_prob in data_dict_new.keys():
        print("Likelihood: ", likelihood, "Post_prob: ", post_prob, "Inf: ", data_dict_new[likelihood, post_prob].isin([np.inf]).sum())

    for metric in metrics :
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
            ax.set_title("PP "+col, rotation=0, fontsize=30)

        for ax, row in zip(axs[:, 0], values_dict["likelihoods"]):
            ax.set_ylabel("LL "+row, rotation=45, fontsize=30)
        friedman_bh_results = perform_bh_by_thresholds(data_dict, values_dict, metric)
        for i,key in enumerate(friedman_bh_results.keys()):
            #time.sleep(1)
            sp.critical_difference_diagram(friedman_bh_results[key]["summary_ranks"], friedman_bh_results[key]["p_adjusted"],
                                           ax=axs[i//4, i%4],
                                           label_fmt_left="{label}", label_fmt_right="{label}", color_palette=palette)
            axs[i//4, i%4].set_xlabel(f"Likelihood: {key[0]}, Post_prob: {key[1]}")
            fig.tight_layout()
            #plt.title(f"Metric: {metric}, Likelihood: {key[0]}, Post_prob: {key[1]}")
            #plt.savefig(os.path.join(root_dir, "plots", metric, f"bh_{metric}_ll{key[0]}_pp{key[1]}.pdf"), bbox_inches='tight')
            #plt.show()
            #plt.clf()
            #sp.sign_plot(friedman_bh_results[key]["p_adjusted"], **heatmap_args)
            #plt.savefig(os.path.join(root_dir, "plots", metric, f"heatmap_{metric}_ll{key[0]}_pp{key[1]}.pdf"), bbox_inches='tight')
            #plt.show()
            #plt.clf()
        fig.show()
        fig.savefig(os.path.join(root_dir, "plots", f"bh_all_{metric}.pdf"), bbox_inches='tight')


