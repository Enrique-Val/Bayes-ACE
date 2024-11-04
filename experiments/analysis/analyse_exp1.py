import os
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
import re

from experiments.utils import friedman_posthoc

# Path to dataset root
root_dir = "../results/exp_1/"

# Wilcoxon test alternative hypothesis
wx_alt = ["two-sided", "greater", "less"]

# Regex to match filenames like distances_data44123_pen1.csv
file_pattern = re.compile(r"distances_data(\d+)_model([A-Z]+)_penalty(\d+)\.csv")

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
    file_pattern_mod = re.compile(r"distances_data"+str(dataset_ids[0])+"_model"+models[0]+"_penalty(\d+)\.csv")
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
                file_name = "distances_data"+str(dataset_id)+"_model"+model+"_penalty"+str(penalty)+".csv"
                file_path = os.path.join(root_dir, dataset_id, model, file_name)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, index_col=0)
                    # Subsitute nans for inf (a good path was not found, hence infinite distance)
                    df = df.fillna(np.inf)
                    data_dict[(dataset_id, model, penalty)] = df
    return data_dict


# Function to run Friedman test and Nemenyi posthoc for each dataset/model/penalty
def perform_bh_by_all(data_dict, values_dict):
    results = {}
    for dataset_id, model, penalty in product(values_dict["dataset_ids"], values_dict["models"], values_dict["penalties"]):
        print(f"Dataset: {dataset_id}, Model: {model}, Penalty: {penalty}")
        results[(dataset_id, model, penalty)] = friedman_posthoc(data_dict[(dataset_id, model, penalty)].dropna())
    return results

def perform_bh_by_dataset(data_dict, values_dict) :
    results = {}
    for model,dataset_id in product(values_dict["models"],values_dict["dataset_ids"]):
        data_model = pd.concat([data_dict[(dataset_id, model, "1")] for penalty in values_dict["penalties"]]).reset_index(drop=True)
        results[model, dataset_id] = friedman_posthoc(data_model.dropna())
    return results

def perform_bh_by_penalty(data_dict, values_dict):
    # First, we group by the data by penalty and model
    data_dict_new = {}
    for model, penalty in product(values_dict["models"], values_dict["penalties"]):
        data_dict_new[model, penalty] = pd.concat([data_dict[(dataset_id, model, penalty)] for dataset_id in values_dict["dataset_ids"]]).reset_index(drop=True)

    results = {}
    for model, penalty in data_dict_new.keys():
        results[(model,penalty)] = friedman_posthoc(data_dict_new[(model, penalty)].dropna())
    return results

def perform_bh(data_dict, values_dict):
    results = {}
    for model in values_dict["models"]:
        data_model = pd.concat([data_dict[(dataset_id, model, penalty)] for dataset_id, penalty in
                                product(values_dict["dataset_ids"], values_dict["penalties"])]).reset_index(drop=True)
        results[model] = friedman_posthoc(data_model.dropna())
    return results

def compare_models_by_penalty(data_dict, values_dict):
    # First, we group by the data by penalty and model
    data_dict_new = {}
    for model, penalty in product(values_dict["models"], values_dict["penalties"]):
        data_dict_new[model, penalty] = pd.concat([data_dict[(dataset_id, model, penalty)] for dataset_id in values_dict["dataset_ids"]]).reset_index(drop=True)

    results = {}
    # We assume to be using ONLY two models. Otherwise, we shouldn't use Wilcoxon, but Friedman
    model1 = values_dict["models"][0]
    model2 = values_dict["models"][1]
    for penalty in values_dict["penalties"]:
        results[penalty] = {}
        diff_arr = data_dict_new[(model1, penalty)].values.flatten() - data_dict_new[(model2, penalty)].values.flatten()
        diff_arr[diff_arr == np.inf] = 1e300
        diff_arr[diff_arr == -np.inf] = -1e300
        # Remove nas
        diff_arr = diff_arr[~np.isnan(diff_arr)]
        for alt_hyp in wx_alt:
            results[penalty][alt_hyp] = wilcoxon(diff_arr, alternative=alt_hyp)
    return results

def compare_models(data_dict, values_dict):
    model1 = values_dict["models"][0]
    model2 = values_dict["models"][1]
    data_model_1 = pd.concat([data_dict[(dataset_id, model1, penalty)] for dataset_id, penalty in product(values_dict["dataset_ids"], values_dict["penalties"])]).reset_index(drop=True)
    data_model_2 = pd.concat([data_dict[(dataset_id, model2, penalty)] for dataset_id, penalty in product(values_dict["dataset_ids"], values_dict["penalties"])]).reset_index(drop=True)
    #print(data_model_1)
    #print(data_model_2)
    # There is an error in the substract. Iterate by rows and substract to try to locate it
    diff_arr = data_model_1.values.flatten() - data_model_2.values.flatten()
    diff_arr[diff_arr == np.inf] = 1e300
    diff_arr[diff_arr == -np.inf] = -1e300
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

    # Load the data
    data_dict = load_data(root_dir, values_dict)

    print(values_dict)
    print(data_dict.keys())

    '''
    # Perform Friedman test and BH test for each dataset/model/penalty
    friedman_bh_results = perform_bh_by_all(data_dict, values_dict)
    for i in friedman_bh_results.keys():
        sp.critical_difference_diagram(friedman_bh_results[i]["summary_ranks"], friedman_bh_results[i]["p_adjusted"], label_fmt_left="{label} vertices", label_fmt_right="{label} vertices")
        plt.title(f"Dataset: {i[0]}, Model: {i[1]}, Penalty: {i[2]}")
        plt.show()
    

    friedman_bh_results = perform_bh_by_dataset(data_dict, values_dict)
    for i in friedman_bh_results.keys():
        sp.critical_difference_diagram(friedman_bh_results[i]["summary_ranks"], friedman_bh_results[i]["p_adjusted"],
                                       label_fmt_left="{label} vertices", label_fmt_right="{label} vertices")
        plt.title(f"Model: {i[0]}, Dataset: {i[1]}")
        plt.show()
    '''

    friedman_bh_results = perform_bh_by_penalty(data_dict, values_dict)
    for i in friedman_bh_results.keys():
        sp.critical_difference_diagram(friedman_bh_results[i]["summary_ranks"], friedman_bh_results[i]["p_adjusted"], label_fmt_left="{label} vertices", label_fmt_right="{label} vertices")
        plt.title(f"Model: {i[0]}, Penalty: {i[1]}")
        plt.show()

    friedman_bh_results = perform_bh(data_dict, values_dict)
    for i in friedman_bh_results.keys():
        sp.critical_difference_diagram(friedman_bh_results[i]["summary_ranks"], friedman_bh_results[i]["p_adjusted"], label_fmt_left="{label} vertices", label_fmt_right="{label} vertices")
        plt.title(f"Model: {i}")
        plt.show()

    # Perform Wilcoxon test between clg and nf for each penalty
    wilcoxon_results_penalty = compare_models_by_penalty(data_dict, values_dict)
    for penalty in wilcoxon_results_penalty.keys():
        print(f"Penalty {penalty}:")
        print(wilcoxon_results_penalty[penalty])
        print()

    # Perform Wilcoxon test between clg and nf for all datasets combined
    wilcoxon_results_global = compare_models(data_dict, values_dict)
    print("Global comparative of models")
    print(wilcoxon_results_global)

