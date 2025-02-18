import os
import time
from itertools import product

import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
import re

from experiments.utils import friedman_posthoc, close_factors

from experiments.experiment2 import FACE_BASELINE, FACE_KDE, FACE_EPS, WACHTER, BAYESACE

# Path to dataset root
root_dir = "../results/exp_2/"

# Wilcoxon test alternative hypothesis
wx_alt = ["two-sided", "greater", "less"]

# Regex to match filenames like distances_data44123_pen1.csv
file_pattern = re.compile(r"distances_data(\d+)_penalty(\d+)\.csv")

# New names dictionary
new_names = {FACE_BASELINE: "FACE GT",
             FACE_KDE: "FACE KDE",
             FACE_EPS: "FACE ε",
             WACHTER: "Wachter"}

# This part is hard coded, might be changed in the future
for vertices in [0, 1, 2, 3]:
    new_names[BAYESACE + "_" + "clg" + "_v" + str(vertices)] = "BayesACE" + " " + str(vertices) + " vertices"

for vertices in [0, 1, 2, 3]:
    new_names[BAYESACE + "_" + "nf" + "_v" + str(vertices)] = "DAACE" + " " + str(vertices) + " vertices"

for vertices in [0, 1, 2, 3]:
    new_names[BAYESACE + "_" + "gt" + "_v" + str(vertices)] = "DAACE GT" + " " + str(vertices) + " vertices"


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

    return algorithms, metrics,{"Data ID": dataset_ids, "Penalty": penalties, "Log-likelihood": likelihoods,
            "Post probability": post_probs}


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
    data_dict_new = aggregate_data(data_dict[metric], values_dict, segregate)
    data_dict_new_dist = aggregate_data(data_dict["distance"], values_dict, segregate)

    data_dict_new = remove_redundant(data_dict_new, data_dict_new_dist)
    for key in data_dict_new.keys():
        data_dict_new[key] = data_dict_new[key].rename(mapper=new_names, axis=1, inplace=False)

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

def aggregate_data(data_dict, values_dict, segregate=None):
    data_dict_new = {}
    if segregate is None:
        data_dict_new["total"] = pd.concat(
            [data_dict[comb] for comb in
             product(*values_dict.values())]).reset_index(drop=True)
    elif len(segregate) == len(values_dict.keys()):
        for comb in product(*[values_dict[key] for key in segregate]):
            data_dict_new[comb] = data_dict[comb].copy()
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
                tmp_list.append(data_dict[reconstructed_combination])
            data_dict_new[combination_segregate] = pd.concat(tmp_list).reset_index(drop=True)
    return data_dict_new


def create_subplot(segregate, join_in_plot, values_dict, data_new, metric, joint_plots, comb=None):
    if len(join_in_plot) == 1:
        n_figs = np.prod([len(values_dict[i]) for i in join_in_plot])
        n_rows, n_cols = close_factors(n_figs)
    elif len(join_in_plot) == 2:
        n_rows = len(values_dict[join_in_plot[0]])
        n_cols = len(values_dict[join_in_plot[1]])
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 10))
    fig_box, ax_box = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 10))
    fig_sp, ax_sp = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 10))
    for i, plot_jointly in enumerate(product(*[values_dict[j] for j in join_in_plot])):
        if comb is not None:
            key_i = tuple(comb) + tuple(plot_jointly)
        else :
            key_i = plot_jointly
        fbh = friedman_posthoc(data_new[key_i])
        # List of the renamed algorithms
        new_algs = fbh["p_adjusted"].columns
        # Create the color palette for the algorithms
        palette = {}
        for method in new_algs:
            if "BayesACE" in method:
                palette[method] = "blue"
            elif "FACE" in method or "Wachter" in method:
                palette[method] = "orange"
            elif "DAACE" in method:
                palette[method] = "purple"
        sp.critical_difference_diagram(fbh["summary_ranks"], fbh["p_adjusted"],
                                       label_fmt_left="{label}", label_fmt_right="{label}",
                                       ax=ax[i // n_cols, i % n_cols], color_palette=palette)
        ax[i // n_cols, i % n_cols].set_title(
            " , ".join([join_in_plot[k] + " = " + plot_jointly[k] for k in range(len(plot_jointly))]))
        # For the boxplot, normalize by dividing by the value of FACE GT
        data_box = data_new[key_i].copy()
        for col in data_box.columns:
            data_box[col] = data_box[col] / data_box["FACE GT"]

        data_box = data_box.drop(columns=["FACE GT", "Wachter"])
        # Reorganize order. First, Wachter, then FACE, then DAACE, then BAYESACE
        sns.boxplot(data=data_box, ax=ax_box[i // n_cols, i % n_cols], showfliers=False, palette=palette)
        ax_box[i // n_cols, i % n_cols].set_title(
            " , ".join([join_in_plot[k] + " = " + plot_jointly[k] for k in range(len(plot_jointly))]))
        # Tilt the x-axis labels
        for tick in ax_box[i // n_cols, i % n_cols].get_xticklabels():
            tick.set_rotation(45)
        heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True,
                        'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
        #sp.sign_plot(fbh["p_adjusted"], ax=ax_sp[i // n_cols, i % n_cols], **heatmap_args)
        '''if len(join_in_plot) == 2 :
            ax[i // n_cols, i % n_cols].set_xlabel(join_in_plot[0])
            ax[i // n_cols, i % n_cols].set_ylabel(join_in_plot[1])'''
    # Set the title
    for curr_fig in [fig, fig_box]:
        curr_fig.suptitle(metric + " " + " ".join(segregate))
        curr_fig.tight_layout()
    if len(segregate) == 0 :
        fig.savefig(os.path.join(joint_plots, "total_cdd.pdf"))
        fig_box.savefig(os.path.join(joint_plots, "total_box.pdf"))
        #fig_sp.savefig(os.path.join(joint_plots, "total_sp.pdf"))
    else :
        fig.savefig(os.path.join(joint_plots, "_".join([segregate[k]+"-"+comb[k] for k in range(len(segregate))]) + "_cdd.pdf"))
        fig_box.savefig(os.path.join(joint_plots, "_".join([segregate[k]+"-"+comb[k] for k in range(len(segregate))]) + "_box.pdf"))
        #fig_sp.savefig(os.path.join(joint_plots, "_".join([segregate[k]+"-"+comb[k] for k in range(len(segregate))]) + "_sp.pdf"))

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
        print(i, metric)
        # Perform BH test for the metric distances globally
        friedman_bh_results = perform_bh_param(data_dict, values_dict, metric, segregate=["Penalty"])[("1",)]
        # Create the color palette for the algorithms
        new_algs = friedman_bh_results["p_adjusted"].columns
        palette = {}
        for method in new_algs:
            if "BayesACE" in method:
                palette[method] = "blue"
            elif "FACE" in method or "Wachter" in method:
                palette[method] = "orange"
            elif "DAACE" in method:
                palette[method] = "purple"
        sp.critical_difference_diagram(friedman_bh_results["summary_ranks"], friedman_bh_results["p_adjusted"],
                                       ax=axs[i // 2, i % 2],
                                       label_fmt_left="{label}", label_fmt_right="{label}",
                                       color_palette=palette)
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

    # Plots for each metric, putting al models together
    plots_dir = os.path.join(root_dir, "plots")
    segregate_list = [[], ["Data ID"], ["Penalty"], ["Data ID", "Penalty"]]
    join_in_plot = ["Log-likelihood", "Post probability"]
    for metric in metrics:
        for segregate in segregate_list:
            data_new = aggregate_data(data_dict[metric], values_dict, segregate+join_in_plot)
            data_new_dist = aggregate_data(data_dict["distance"], values_dict, segregate+join_in_plot)
            data_new = remove_redundant(data_new, data_new_dist)
            # Rename algorithms
            for key in data_new.keys():
                data_new[key] = data_new[key].rename(mapper=new_names, axis=1, inplace=False)


            if len(segregate) == 0:
                # Name of the dir is the product of the segregate list
                joint_plots = os.path.join(plots_dir, metric)
                if not os.path.exists(joint_plots):
                    os.makedirs(joint_plots)
                create_subplot(segregate, join_in_plot, values_dict, data_new, metric, joint_plots)
            else:
                # Name of the dir is the product of the segregate list
                joint_plots = os.path.join(plots_dir, metric, "_".join(segregate))
                if not os.path.exists(joint_plots):
                    os.makedirs(joint_plots)
                for i in product(*[values_dict[j] for j in segregate]):
                    create_subplot(segregate, join_in_plot, values_dict, data_new, metric, joint_plots, i)


