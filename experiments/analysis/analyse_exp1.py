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

# Regex to match filenames like distances_data44123_pen1.csv
file_pattern = re.compile(r"distances_data(\d+)_penalty(\d+)\.csv")

# Get all the values for penalty, dataset_id, models and n_vertex
def get_values(root_dir):
    dataset_ids = []
    penalties = []
    models = []

    # Get datasets ids
    for dataset_id in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset_id)
        if os.path.isdir(dataset_path):
            dataset_ids.append(dataset_id)

    # Get the list of models
    dataset_path = os.path.join(root_dir, dataset_ids[0])
    for model in os.listdir(dataset_path):  # "clg" and "nf" subfolders (models)
        model_path = os.path.join(dataset_path, model)
        if os.path.isdir(model_path):
            models.append(model)

    # Get the list of penalties
    model_path = os.path.join(dataset_path, models[0])
    file_pattern_mod = re.compile(r"distances_data"+str(dataset_ids[0])+"_penalty(\d+)\.csv")
    for file in os.listdir(model_path):
        match = file_pattern_mod.match(file)
        if match:
            # Extract dataset_id and penalty from filename
            penalty = match.group(1)
            penalties.append(penalty)

    # Get the list of n_vertex opening any file
    any_file = os.listdir(model_path)[0]
    file_path = os.path.join(model_path, any_file)
    df = pd.read_csv(file_path, index_col=0)
    vertices = list(df.columns)
    return {"dataset_ids": dataset_ids, "models": models, "penalties": penalties, "vertices": vertices}



# Function to load and organize data from the directory
def load_data(root_dir, values_dict):
    data_dict = {}
    for dataset_id in values_dict["dataset_ids"]:
        for model in values_dict["models"]:
            for penalty in values_dict["penalties"]:
                # Get the path to the file
                file_name = "distances_data"+str(dataset_id)+"_penalty"+str(penalty)+".csv"
                file_path = os.path.join(root_dir, dataset_id, model, file_name)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, index_col=0)
                    df = df.replace([np.inf, -np.inf], np.nan)
                    data_dict[(dataset_id, model, penalty)] = df
    return data_dict


# Function to run Friedman test and Nemenyi posthoc for each dataset/model/penalty
def perform_bh_by_all(data_dict, values_dict):
    results = {}
    for dataset_id, model, penalty in product(values_dict["dataset_ids"], values_dict["models"], values_dict["penalties"]):
        print(f"Dataset: {dataset_id}, Model: {model}, Penalty: {penalty}")
        results[(dataset_id, model, penalty)] = bh_test(data_dict[(dataset_id, model, penalty)].dropna())
    return results

def perform_bh_by_penalty(data_dict, values_dict):
    # First, we group by the data by penalty and model
    data_dict_new = {}
    for model, penalty in product(values_dict["models"], values_dict["penalties"]):
        data_dict_new[model, penalty] = pd.concat([data_dict[(dataset_id, model, penalty)] for dataset_id in values_dict["dataset_ids"]])

    results = {}
    for model, penalty in data_dict_new.keys():
        results[(model,penalty)] = bh_test(data_dict_new[(model, penalty)].dropna())
    return results

# Function to concatenate datasets for global analysis
def concatenate_datasets(data_by_penalty):
    concatenated_results = {}
    for penalty, datasets_models in data_by_penalty.items():
        concatenated_df = pd.concat([df for df in datasets_models.values()])
        concatenated_results[penalty] = concatenated_df
    return concatenated_results


# Function to plot Nemenyi test results as a heatmap
def plot_nemenyi_heatmap(nemenyi_result, dataset_id, model, penalty):
    plt.figure(figsize=(10, 8))
    sns.heatmap(nemenyi_result, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(f"Nemenyi Posthoc Test - Dataset: {dataset_id}, Model: {model}, Penalty: {penalty}")
    plt.xlabel("Vertex")
    plt.ylabel("Vertex")
    plt.show()


# Function to print and save the Friedman/Nemenyi results
def print_and_save_results(results, output_dir="results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for penalty, models_data in results.items():
        print(f"Penalty {penalty}:")

        for (dataset_id, model), test_results in models_data.items():
            friedman_p = test_results["friedman"].pvalue
            print(f"  Dataset: {dataset_id}, Model: {model}")
            print(f"    Friedman p-value: {friedman_p}")

            if test_results["nemenyi"] is not None:
                nemenyi_df = test_results["nemenyi"]
                print(f"    Nemenyi Test (Vertex Comparison):")
                print(nemenyi_df)

                # Save the Nemenyi results as a CSV
                nemenyi_file = f"nemenyi_data{dataset_id}_model{model}_pen{penalty}.csv"
                nemenyi_df.to_csv(os.path.join(output_dir, nemenyi_file))


def perform_wilcoxon_test(data_by_penalty):
    wilcoxon_results = {}

    for penalty, datasets_models in data_by_penalty.items():
        wilcoxon_results[penalty] = {}
        for dataset_id in {k[0] for k in datasets_models.keys()}:  # Unique dataset IDs
            # Ensure both models (clg and nf) are available for comparison
            if (dataset_id, "clg") in datasets_models and (dataset_id, "nf") in datasets_models:
                print(datasets_models)
                clg_df = datasets_models[(dataset_id, "clg")]
                nf_df = datasets_models[(dataset_id, "nf")]

                # Wilcoxon test across the values of all vertex columns (paired test)
                wilcoxon_stats = []
                for vertex in clg_df.columns:
                    stat, p_value = wilcoxon(clg_df[vertex], nf_df[vertex])
                    wilcoxon_stats.append((vertex, stat, p_value))

                wilcoxon_results[penalty][dataset_id] = wilcoxon_stats

    return wilcoxon_results

def perform_global_wilcoxon_test(data_by_penalty):
    global_wilcoxon_results = {}

    for penalty, datasets_models in data_by_penalty.items():
        # Collect and concatenate data across all datasets for clg and nf
        combined_clg = pd.concat([df for (dataset_id, model), df in datasets_models.items() if model == "clg"])
        combined_nf = pd.concat([df for (dataset_id, model), df in datasets_models.items() if model == "nf"])

        # Wilcoxon test across all vertex columns
        global_wilcoxon_stats = []
        for vertex in combined_clg.columns:
            stat, p_value = wilcoxon(combined_clg[vertex], combined_nf[vertex])
            global_wilcoxon_stats.append((vertex, stat, p_value))

        global_wilcoxon_results[penalty] = global_wilcoxon_stats

    return global_wilcoxon_results


# Function to print Wilcoxon test results
def print_wilcoxon_results(wilcoxon_results, output_dir="wilcoxon_results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for penalty, datasets_data in wilcoxon_results.items():
        print(f"Penalty {penalty} - Wilcoxon Test:")
        for dataset_id, stats in datasets_data.items():
            print(f"  Dataset: {dataset_id}")
            for vertex, stat, p_value in stats:
                print(f"    Vertex {vertex}: stat = {stat}, p-value = {p_value}")

            # Save the Wilcoxon results to a CSV
            wilcoxon_file = f"wilcoxon_data{dataset_id}_pen{penalty}.csv"
            pd.DataFrame(stats, columns=["Vertex", "Statistic", "P-Value"]).to_csv(
                os.path.join(output_dir, wilcoxon_file), index=False
            )


# Run the main function when the script is executed
if __name__ == "__main__":
    # Get the values for dataset_id, model, penalty and n_vertex
    values_dict = get_values(root_dir)

    # Load the data
    data_dict = load_data(root_dir, values_dict)

    print(data_dict.keys())

    '''# Perform Friedman test and BH test for each dataset/model/penalty
    friedman_bh_results = perform_bh_by_all(data_dict, values_dict)
    for i in friedman_bh_results.keys():
        sp.critical_difference_diagram(friedman_bh_results[i]["summary"], friedman_bh_results[i]["p_adjusted"], label_fmt_left="{label}", label_fmt_right="{label}")
        plt.title(f"Dataset: {i[0]}, Model: {i[1]}, Penalty: {i[2]}")
        plt.show()'''

    friedman_bh_results = perform_bh_by_penalty(data_dict, values_dict)
    for i in friedman_bh_results.keys():
        sp.critical_difference_diagram(friedman_bh_results[i]["summary"], friedman_bh_results[i]["p_adjusted"], label_fmt_left="{label}", label_fmt_right="{label}")
        plt.title(f"Model: {i[0]}, Penalty: {i[1]}")
        plt.show()

    raise Exception("Stop here")

    # Perform Friedman test and Nemenyi posthoc for each dataset/model/penalty
    friedman_nemenyi_results = perform_friedman_and_bh(data_by_penalty)

    # Plot Nemenyi test results for each significant combination
    '''for penalty, models_data in friedman_nemenyi_results.items():
        print(models_data)
        for (dataset_id, model), test_results in models_data.items():
            if model == "clg":
                sp.critical_difference_diagram(test_results["summary"],test_results["p_values"],label_fmt_left="{label}",label_fmt_right="{label}")
                plt.show()
                sp.critical_difference_diagram(test_results["summary"], test_results["p_adjusted"],
                                               label_fmt_left="{label}", label_fmt_right="{label}")
                plt.show()'''
    # Print and save the results
    #print_and_save_results(friedman_nemenyi_results)

    # For all datasets combined, concatenate and perform global analysis
    combined_data_by_penalty = concatenate_datasets(data_by_penalty)
    for (dataset_id, model), df in combined_data_by_penalty["combined"].items():
        # Perform Friedman test across the vertex columns
        results = bh_test(df)
        sp.critical_difference_diagram(results["summary"], results["p_adjusted"], label_fmt_left="{label}",)
        plt.show()

    # Print and save combined results
    print("Global Analysis (All Datasets Combined):")
    #print_and_save_results(friedman_nemenyi_combined, output_dir="global_results")

    # Perform Wilcoxon test between clg and nf for each dataset and penalty
    wilcoxon_results = perform_wilcoxon_test(data_by_penalty)
    print_wilcoxon_results(wilcoxon_results)

    # For all datasets combined, concatenate and perform global analysis
    combined_data_by_penalty = concatenate_datasets(data_by_penalty)
    friedman_bh_combined = perform_friedman_and_bh({"combined": combined_data_by_penalty})

    # Print and save combined results
    print("Global Analysis (All Datasets Combined):")
    print_and_save_results(friedman_bh_combined, output_dir="global_results")

    # Perform Wilcoxon test globally across all datasets combined
    global_wilcoxon_results = perform_global_wilcoxon_test(data_by_penalty)
    print_wilcoxon_results({"combined": global_wilcoxon_results}, output_dir="global_wilcoxon_results")

