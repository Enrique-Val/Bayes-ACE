import os
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
import re

from experiments.utils import bh_test

# Path to dataset root
root_dir = "../results/exp_1/"

# Wilcoxon test alternative hypothesis
wx_alt = ["two-sided", "greater", "less"]

# Regex to match filenames like distances_data44123_pen1.csv
file_pattern = re.compile(r"distances_data(\d+)_penalty(\d+)\.csv")

# Get all the values for penalty, dataset_id, models and n_vertex
def get_values(root_dir):
    dataset_ids = []
    metrics = []
    likelihoods = []
    post_probs = []

    # Get datasets ids
    for dataset_id in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset_id)
        if os.path.isdir(dataset_path):
            dataset_ids.append(dataset_id)

    # Get the list of metrics
    dataset_path = os.path.join(root_dir, dataset_ids[0])
    for metric in os.listdir(dataset_path):  # "clg" and "nf" subfolders (models)
        metric_path = os.path.join(dataset_path, metric)
        if os.path.isdir(metric_path):
            metrics.append(metric)

    # Get the list of likelihood and post_prob thresholds
    metric_path = os.path.join(dataset_path, metrics[0])
    file_pattern_mod = re.compile(r"likelihood(\d+)_pp(\d+)\.csv")
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

    # Get the list of algorithms opening any file
    any_file = os.listdir(metric_path)[0]
    file_path = os.path.join(metric_path, any_file)
    df = pd.read_csv(file_path, index_col=0)
    algorithms = list(df.columns)
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
                data_dict[metric][(dataset_id, likelihood, post_prob)] = df
    return data_dict


# Function to run Friedman test and Nemenyi posthoc for each dataset/model/penalty
def perform_bh_by_all(data_dict, values_dict, metric):
    results = {}
    for dataset_id, likelihood, post_prob in product(values_dict["dataset_ids"], values_dict["likelihoods"], values_dict["post_probs"]):
        results[(dataset_id, likelihood, post_prob)] = bh_test(data_dict[metric][(dataset_id, likelihood, post_prob)].dropna())


def perform_bh_by_thresholds(data_dict, values_dict, metric):
    # First, we group by the data by penalty and model
    data_dict_new = {}
    for likelihood, post_prob in product(values_dict["likelihoods"], values_dict["post_probs"]):
        data_dict_new[likelihood, post_prob] = pd.concat([data_dict[metric][(dataset_id, likelihood, post_prob)] for dataset_id in values_dict["dataset_ids"]])

    results = {}
    for likelihood, post_prob in data_dict_new.keys():
        results[(likelihood, post_prob)] = bh_test(data_dict_new[(likelihood, post_prob)].dropna())
    return results

def perform_bh_by_dataset(data_dict, values_dict, metric):
    data_dict_new = {}
    for dataset_id in values_dict["dataset_ids"]:
        data_dict_new[dataset_id] = pd.concat([data_dict[metric][(dataset_id, likelihood, post_prob)] for likelihood, post_prob in product(values_dict["likelihoods"], values_dict["post_probs"])])

    results = {}
    for dataset_id in data_dict_new.keys():
        results[dataset_id] = bh_test(data_dict_new[dataset_id].dropna())
    return results


def perform_bh(data_dict, values_dict,metric):
    data_dict_new = pd.concat([data_dict[metric][(dataset_id, likelihood, post_prob)] for dataset_id, likelihood, post_prob in product(values_dict["dataset_ids"], values_dict["likelihoods"], values_dict["post_probs"])])
    results = bh_test(data_dict_new.dropna())
    return results




def compare_models_by_penalty(data_dict, values_dict):
    # First, we group by the data by penalty and model
    data_dict_new = {}
    for model, penalty in product(values_dict["models"], values_dict["penalties"]):
        data_dict_new[model, penalty] = pd.concat([data_dict[(dataset_id, model, penalty)] for dataset_id in values_dict["dataset_ids"]])

    results = {}
    # We assume to be using ONLY two models. Otherwise, we shouldn't use Wilcoxon, but Friedman

    model1 = "clg"
    model2 ="nf"
    for penalty in values_dict["penalties"]:
        results[penalty] = {}
        diff_arr = data_dict_new[(model1, penalty)].values.flatten() - data_dict_new[(model2, penalty)].values.flatten()
        # Remove nas
        diff_arr = diff_arr[~np.isnan(diff_arr)]
        for alt_hyp in wx_alt:
            results[penalty][alt_hyp] = wilcoxon(diff_arr, alternative=alt_hyp)
    return results

def compare_models(data_dict, values_dict):
    model1 = "clg"
    model2 = "nf"
    data_model_1 = pd.concat([data_dict[(dataset_id, model1, penalty)] for dataset_id, penalty in product(values_dict["dataset_ids"], values_dict["penalties"])])
    data_model_2 = pd.concat([data_dict[(dataset_id, model2, penalty)] for dataset_id, penalty in product(values_dict["dataset_ids"], values_dict["penalties"])])
    diff_arr = data_model_1.values.flatten() - data_model_2.values.flatten()
    # Remove nas
    diff_arr = diff_arr[~np.isnan(diff_arr)]
    results = {}
    for alt_hyp in wx_alt:
        results[alt_hyp] = wilcoxon(diff_arr, alternative=alt_hyp)
    return results

# Run the main function when the script is executed
if __name__ == "__main__":
    # Get the values for dataset_id, model, penalty and n_vertex
    values_dict = get_values(root_dir)
    print(values_dict["metrics"])

    # Load the data
    data_dict = load_data(root_dir, values_dict)

    print(data_dict.keys())

    for metric in ["distance", "time"]:
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
        friedman_bh_results = perform_bh_by_thresholds(data_dict, values_dict, metric)
        for i in friedman_bh_results.keys():
            sp.critical_difference_diagram(friedman_bh_results[i]["summary_ranks"], friedman_bh_results[i]["p_adjusted"], label_fmt_left="{label} vertices", label_fmt_right="{label} vertices")
            plt.title(f"Likelihood: {i[0]}, Post_prob: {i[1]}")
            plt.show()

        # Perform BH test for the metric distances globally
        friedman_bh_results = perform_bh(data_dict, values_dict, metric)
        for i in friedman_bh_results.keys():
            sp.critical_difference_diagram(friedman_bh_results[i]["summary_ranks"], friedman_bh_results[i]["p_adjusted"], label_fmt_left="{label} vertices", label_fmt_right="{label} vertices")
            plt.title(f"Model: {i}")
            plt.show()

