import os
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
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
        # Aggregate datasets in two groups:
        group_close = ["44089","44090","44091","44121","44125","44126","44127","44128"]

        agg_dict = defaultdict(list)

        for (dataset_id, penalty, likelihood, post_prob), df in data_dict[metric].items():
            if dataset_id in group_close:
                new_key = ("close", penalty, likelihood, post_prob)
            else:
                new_key = ("different", penalty, likelihood, post_prob)
            agg_dict[new_key].append(df)

        final_dict = {key : pd.concat(value, ignore_index=True).reset_index(drop=True) for key, value in agg_dict.items()}
        data_dict[metric] = final_dict
    # Readapt values_dict
    new_values_dict = values_dict.copy()
    new_values_dict["Data ID"] = ["close", "different"]
    print(new_values_dict)
    return data_dict, new_values_dict


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
    return results, data_dict_new

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

def plot_segregate(segregate, join_in_plot, values_dict, data_new, metric, joint_plots, box_plot=True) :
    if len(segregate) == 0:
        # Name of the dir is the product of the segregate list
        if not os.path.exists(joint_plots):
            os.makedirs(joint_plots)
        create_subplot(segregate, join_in_plot, values_dict, data_new, metric, joint_plots, box_plot=box_plot)
    else:
        # Name of the dir is the product of the segregate list
        if not os.path.exists(joint_plots):
            os.makedirs(joint_plots)
        for i in product(*[values_dict[j] for j in segregate]):
            create_subplot(segregate, join_in_plot, values_dict, data_new, metric, joint_plots, comb=i,
                           box_plot=box_plot)

def get_palette(new_algs):
    palette = {}
    for method in new_algs:
        if "BayesACE" in method:
            palette[method] = "blue"
        elif "FACE" in method or "Wachter" in method:
            palette[method] = "orange"
        elif "DAACE" in method:
            palette[method] = "purple"
    return palette


def create_subplot(segregate, join_in_plot, values_dict, data_new, metric, joint_plots, comb=None, box_plot=True):
    p_values_dir = os.path.join(joint_plots, "p_values")
    if not os.path.exists(p_values_dir):
        os.makedirs(p_values_dir)
    if len(join_in_plot) == 1:
        n_figs = np.prod([len(values_dict[i]) for i in join_in_plot])
        print(n_figs)
        n_rows, n_cols = close_factors(n_figs)
        print(n_rows, n_cols)
    elif len(join_in_plot) == 2:
        n_rows = len(values_dict[join_in_plot[0]])
        n_cols = len(values_dict[join_in_plot[1]])
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 10))
    if box_plot:
        fig_box, ax_box = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 10))
        # Another Boxplots without infs or nans
        fig_box_nans, ax_box_nans = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 10))
        figures = [fig, fig_box, fig_box_nans]
        figures_str = ["Critical diff", "Box plot", "Box plot without nans"]
    else :
        figures = [fig]
        figures_str = ["Critical diff"]
    #fig_sp, ax_sp = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 10))
    for i, plot_jointly in enumerate(product(*[values_dict[j] for j in join_in_plot])):
        if n_cols > 1 :
            ax_i = ax[i // n_cols, i % n_cols]
            if box_plot:
                ax_box_i = ax_box[i // n_cols, i % n_cols]
                ax_box_nans_i = ax_box_nans[i // n_cols, i % n_cols]
        else :
            ax_i = ax[i]
            if box_plot:
                ax_box_i = ax_box[i]
                ax_box_nans_i = ax_box_nans[i]

        if comb is not None:
            key_i = tuple(comb) + tuple(plot_jointly)
        else :
            key_i = plot_jointly
        data_new_i = data_new[key_i].copy()
        if metric == "distance_to_face_baseline":
            data_new_i = data_new_i.drop(columns=["FACE GT"])
        fbh = friedman_posthoc(data_new_i)
        # List of the renamed algorithms
        new_algs = fbh["p_adjusted"].columns
        # Create the color palette for the algorithms
        palette = get_palette(new_algs)
        sp.critical_difference_diagram(fbh["summary_ranks"], fbh["p_adjusted"],
                                       label_fmt_left="{label}", label_fmt_right="{label}",
                                       ax=ax_i, color_palette=palette)
        p_values_subdir = "."
        if len(segregate) > 0:
            p_values_subdir = "_".join([segregate[k]+"-"+comb[k] for k in range(len(segregate))])
        if not os.path.exists(os.path.join(p_values_dir, p_values_subdir)):
            os.makedirs(os.path.join(p_values_dir, p_values_subdir))
        fbh["p_adjusted"].to_csv(os.path.join(p_values_dir, p_values_subdir, " , ".join([join_in_plot[k] + " = " + plot_jointly[k] for k in range(len(plot_jointly))]) + ".csv"))
        ax_i.set_title(
            " , ".join([join_in_plot[k] + " = " + plot_jointly[k] for k in range(len(plot_jointly))]))
        if box_plot:
            # For the boxplot, normalize by dividing by the value of FACE GT
            data_box = data_new_i.copy()
            # Normalize each row of data by standardizing it
            data_box = data_box.subtract(data_box.mean(axis=1), axis=0).divide(data_box.std(axis=1), axis=0)

            # Reorganize order. Order: DAACE GT, DAACE, BayesACE, FACE KDE, FACE epsilon, and Wachter
            new_order = [col for col in data_box.columns if "DAACE GT" in col] + [col for col in data_box.columns if "DAACE" in col and "GT" not in col] + [col for col in data_box.columns if "BayesACE" in col] + ["FACE GT","FACE KDE", "FACE ε", "Wachter"]
            if metric == "distance_to_face_baseline":
                new_order = [col for col in new_order if "FACE GT" != col]
            data_box = data_box[new_order]
            if metric == "distance" :
                print(key_i)
                print(data_box.median())
                print()
            sns.boxplot(data=data_box, ax=ax_box_i, showfliers=False, palette=palette)
            ax_box_i.set_title(
                " , ".join([join_in_plot[k] + " = " + plot_jointly[k] for k in range(len(plot_jointly))]))
            # Tilt the x-axis labels
            for tick in ax_box_i.get_xticklabels():
                tick.set_rotation(45)
            # Set scale to logarithmic for the y-axis IF a box goes above 100 (Q3 + 1.5*IQR)
            if (data_box.quantile(0.75) + 1.5 * (data_box.quantile(0.75) - data_box.quantile(0.25)) > 100).any():
                ax_box_i.set_yscale("log")
            # Do the same, but without infs or nans
            data_box_nans = data_box.replace([np.inf, -np.inf], np.nan)
            data_box_nans = data_box_nans.dropna().reset_index(drop=True)
            sns.boxplot(data=data_box_nans, ax=ax_box_nans_i, showfliers=False, palette=palette)
            ax_box_nans_i.set_title(
                " , ".join([join_in_plot[k] + " = " + plot_jointly[k] for k in range(len(plot_jointly))]))
            # Tilt the x-axis labels
            for tick in ax_box_nans_i.get_xticklabels():
                tick.set_rotation(45)
            # Set scale to logarithmic for the y-axis IF a box goes above 100 (Q3 + 1.5*IQR)
            if (data_box_nans.quantile(0.75) + 1.5 * (data_box_nans.quantile(0.75) - data_box_nans.quantile(0.25)) > 100).any():
                ax_box_nans_i.set_yscale("log")

    # Set the title
    for curr_fig, fig_str in zip(figures, figures_str):
        if not os.path.exists(os.path.join(joint_plots, fig_str)):
            os.makedirs(os.path.join(joint_plots, fig_str))
        if len(segregate) == 0 :
            curr_fig.suptitle(metric + " " + " ".join(segregate))
            curr_fig.tight_layout()
            curr_fig.savefig(os.path.join(joint_plots, fig_str, "total_segg_"+ "-".join(join_in_plot) +".pdf"))
        else :
            joint_comb_eq = " ".join([segregate[k] + " = " + comb[k] for k in range(len(segregate))])
            joint_comb = "_".join([segregate[k] + "-" + comb[k] for k in range(len(segregate))])
            curr_fig.suptitle(metric + " " + joint_comb_eq)
            curr_fig.tight_layout()
            curr_fig.savefig(os.path.join(joint_plots, fig_str, joint_comb + "_segg_"+ "-".join(join_in_plot) + ".pdf"))

def get_single_agregated_plot(data_dict, values_dict, metric, segregate, plot_dir):
    if len(segregate) > 0:
        subplot_dir = os.path.join(plot_dir, metric, "_".join(segregate))
    else:
        subplot_dir = os.path.join(plot_dir, metric, "joint")
    data_new = aggregate_data(data_dict[metric], values_dict, segregate)
    data_new_dist = aggregate_data(data_dict["distance"], values_dict, segregate)
    data_new = remove_redundant(data_new, data_new_dist)
    # Rename algorithms
    for key in data_new.keys():
        if len(segregate) > 0:
            title = "Metric: "+ metric + "    " + " , ".join([segregate[k] + " = " + key[k] for k in range(len(segregate))])
            file_name = "Metric_" + metric + "_".join([segregate[k] + "-" + key[k] for k in range(len(segregate))])
        else:
            title = "Metric: " + metric
            file_name = "Metric_" + metric
        fig = plt.figure()
        ax = fig.gca()
        fig_box = plt.figure()
        ax_box = fig_box.gca()
        data_new[key] = data_new[key].rename(mapper=new_names, axis=1, inplace=False)
        fbh = friedman_posthoc(data_new[key])
        # List of the renamed algorithms
        new_algs = fbh["p_adjusted"].columns
        # Create the color palette for the algorithms
        palette = get_palette(new_algs)
        sp.critical_difference_diagram(fbh["summary_ranks"], fbh["p_adjusted"],
                                       label_fmt_left="{label}", label_fmt_right="{label}",
                                       color_palette=palette, ax=ax)
        fig.suptitle(", ".join([segregate[k] + " = " + key[k] for k in range(len(segregate))]))
        fig.savefig(os.path.join(subplot_dir, file_name + ".pdf"), bbox_inches='tight')
        # For the boxplot, normalize by dividing by the value of FACE GT
        data_box = data_new[key].copy()
        for col in data_box.columns:
            data_box[col] = data_box[col] / data_box["FACE GT"]

        data_box = data_box.drop(columns=["FACE GT", "Wachter"])
        # Reorganize order. First, Wachter, then FACE, then DAACE, then BAYESACE
        sns.boxplot(data=data_box, ax=ax_box, showfliers=False, palette=palette)
        fig_box.suptitle(", ".join([segregate[k] + " = " + key[k] for k in range(len(segregate))]))
        fig_box.savefig(os.path.join(subplot_dir, file_name + "_box.pdf"), bbox_inches='tight')




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
    data_dict, values_dict = load_data(root_dir, metrics, values_dict)

    # Subplot, 2x4 for each metric
    fig, axs = plt.subplots(4, 2, figsize=(12, 12))
    fig_box, axs_box = plt.subplots(4, 2, figsize=(12, 12))
    p_values_dir = os.path.join(root_dir, "plots", "p_values")
    if not os.path.exists(p_values_dir):
        os.makedirs(p_values_dir)
    for i, metric in enumerate(metrics):
        print(i, metric)
        # Perform BH test for the metric distances globally
        friedman_bh_results, data_dict_new = perform_bh_param(data_dict, values_dict, metric)
        friedman_bh_results = friedman_bh_results["total"]
        data_dict_new = data_dict_new["total"]
        # Create the color palette for the algorithms
        new_algs = friedman_bh_results["p_adjusted"].columns
        palette = get_palette(new_algs)
        sp.critical_difference_diagram(friedman_bh_results["summary_ranks"], friedman_bh_results["p_adjusted"],
                                       ax=axs[i // 2, i % 2],
                                       label_fmt_left="{label}", label_fmt_right="{label}",
                                       color_palette=palette)
        axs[i // 2, i % 2].set_title(f"Metric: {metric}")
        # Create a box plot
        data_box = data_dict_new.copy()
        if metric == "distance_to_face_baseline":
            data_box = data_box.drop(columns=["FACE GT"])
        data_box = data_box.subtract(data_box.mean(axis=1), axis=0).divide(data_box.std(axis=1), axis=0)
        print(data_box.median())
        print(data_box.mean())
        print()
        # Reorganize order. Order: DAACE GT, DAACE, BayesACE, FACE KDE, FACE epsilon, and Wachter
        new_order = [col for col in data_box.columns if "DAACE GT" in col] + [col for col in data_box.columns if "DAACE" in col and "GT" not in col] + [col for col in data_box.columns if "BayesACE" in col] + ["FACE GT","FACE KDE", "FACE ε", "Wachter"]
        if metric == "distance_to_face_baseline":
            new_order = [col for col in new_order if "FACE GT" != col]
        data_box = data_box[new_order]
        sns.boxplot(data=data_box, ax=axs_box[i // 2, i % 2], showfliers=False, palette=palette)
        axs_box[i // 2, i % 2].set_title(f"Metric: {metric}")
        # Tilt the x-axis labels
        for tick in axs_box[i // 2, i % 2].get_xticklabels():
            tick.set_rotation(45)

        # Save the p-values
        friedman_bh_results["p_adjusted"].to_csv(os.path.join(p_values_dir, metric + ".csv"))
    fig.tight_layout()
    fig.savefig(os.path.join(root_dir, "plots", f"bh_all.pdf"), bbox_inches='tight')
    fig_box.tight_layout()
    fig_box.savefig(os.path.join(root_dir, "plots", f"bh_all_box.pdf"), bbox_inches='tight')


    # Analyse the optimal amount of vertices for BayesACE
    # Plots for each metric, putting al models together
    plots_dir = os.path.join(root_dir, "plots","vertices")

    segregate_list = [[], ["Data ID"], ["Penaly"], ["Data ID", "Penalty"]]
    join_in_plot = ["Log-likelihood"]
    for metric in ["distance", "distance_to_face_baseline"]:
        for segregate in segregate_list:
            data_new = aggregate_data(data_dict[metric], values_dict, segregate + join_in_plot)
            for model in ["clg", "nf", "gt"]:
                model_str = BAYESACE + "_" + model
                data_new_model = {}
                for key in data_new.keys():
                    data_new_model[key] = data_new[key][[col for col in data_new[key].columns if model_str in col]]
                # Rename algorithms
                for key in data_new_model.keys():
                    data_new_model[key] = data_new_model[key].rename(mapper=new_names, axis=1, inplace=False)
                    # Drop rows with nans
                    data_new_model[key] = data_new_model[key].dropna()

                joint_plots = os.path.join(plots_dir, metric, model)
                joint_plots = os.path.join(joint_plots, "_".join(segregate + join_in_plot),
                                           "joint_" + "_".join(join_in_plot))
                plot_segregate(segregate, join_in_plot, values_dict, data_new_model, metric, joint_plots, box_plot=False)

    # In addition, generate a plot for all data aggregated
    for metric in ["distance", "distance_to_face_baseline"]:
        data_new = aggregate_data(data_dict[metric], values_dict, None)["total"]
        for model in ["clg", "nf", "gt"]:
            fig = plt.figure()
            ax = fig.gca()
            model_str = BAYESACE + "_" + model
            data_new_model = data_new[[col for col in data_new.columns if model_str in col]]
            # Rename algorithms
            data_new_model = data_new_model.rename(mapper=new_names, axis=1, inplace=False)

            joint_plots = os.path.join(plots_dir, metric)
            fbh = friedman_posthoc(data_new_model)
            palette = get_palette(fbh["p_adjusted"].columns)
            sp.critical_difference_diagram(fbh["summary_ranks"], fbh["p_adjusted"],
                                           label_fmt_left="{label}", label_fmt_right="{label}",
                                           color_palette=palette, ax=ax)
            fig.suptitle(f"Metric: {metric} and model: {model}")
            fig.tight_layout()
            fig.savefig(os.path.join(joint_plots, model + "_total.pdf"), bbox_inches='tight')
            fbh["p_adjusted"].to_csv(os.path.join(joint_plots, model + "_total_p_values.csv"))


    # Plots for each metric, putting al models together
    plots_dir = os.path.join(root_dir, "plots")
    segregate_list = [[], ["Data ID"], ["Penalty"], ["Data ID", "Penalty"]]
    join_in_plot = ["Log-likelihood"]
    for metric in metrics:
        for segregate in segregate_list:
            data_new = aggregate_data(data_dict[metric], values_dict, segregate+join_in_plot)
            data_new_dist = aggregate_data(data_dict["distance"], values_dict, segregate+join_in_plot)
            data_new = remove_redundant(data_new, data_new_dist)
            # Rename algorithms
            for key in data_new.keys():
                data_new[key] = data_new[key].rename(mapper=new_names, axis=1, inplace=False)

            joint_plots = os.path.join(plots_dir, metric)
            joint_plots = os.path.join(joint_plots, "_".join(segregate + join_in_plot),
                                       "joint_" + "_".join(join_in_plot))
            plot_segregate(segregate, join_in_plot, values_dict, data_new, metric, joint_plots)


    plots_dir = os.path.join(root_dir, "plots")
    # Now, do it the opposite way. We want to plot all the metrics together for each segregate
    metric_plots_dir = os.path.join(plots_dir, "metrics")
    segregate_list = [["Data ID", "Penalty","Log-likelihood"]]
    #  ["Penalty"], ["Data ID"], ["Log-likelihood"], ["Data ID", "Penalty"], ["Data ID", "Log-likelihood"],
    #                       ["Penalty","Log-likelihood"],
    for segregate in segregate_list:

        metric_plots_subdir = os.path.join(metric_plots_dir, "_".join(segregate))
        if not os.path.exists(metric_plots_subdir):
            os.makedirs(metric_plots_subdir)
        # Get a combination of the segregate values
        for comb in product(*[values_dict[key] for key in segregate]):
            fig, ax = plt.subplots(4, 2, figsize=(12, 12))
            for i, metric in enumerate(metrics):
                data_new = aggregate_data(data_dict[metric], values_dict, segregate)
                data_new_dist = aggregate_data(data_dict["distance"], values_dict, segregate)
                data_comb = remove_redundant(data_new, data_new_dist)[comb]
                # Rename algorithms
                data_comb = data_comb.rename(mapper=new_names, axis=1, inplace=False)
                # Perform BH test for the metric distances globally
                friedman_bh_results = friedman_posthoc(data_comb)
                # Create the color palette for the algorithms
                new_algs = friedman_bh_results["p_adjusted"].columns
                palette = get_palette(new_algs)
                sp.critical_difference_diagram(friedman_bh_results["summary_ranks"], friedman_bh_results["p_adjusted"],
                                               ax=ax[i // 2, i % 2],
                                               label_fmt_left="{label}", label_fmt_right="{label}",
                                               color_palette=palette)
                ax[i // 2, i % 2].set_title(f"Metric: {metric}")
            fig.tight_layout()
            segregate_str = "_".join([segregate[k] + "-" + comb[k] for k in range(len(segregate))])
            fig.savefig(os.path.join(metric_plots_subdir, segregate_str + ".pdf"), bbox_inches='tight')
