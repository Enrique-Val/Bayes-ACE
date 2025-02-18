import os
import pickle
from itertools import product

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon
import scikit_posthocs as sp

from bayesace import brier_score, auc, square_diff
from bayesace.models.conditional_kde import ConditionalKDE
from experiments.analysis.analyse_exp import aggregate_data
from experiments.utils import friedman_posthoc, close_factors
import matplotlib.pyplot as plt
import seaborn as sns

# Path to dataset root
root_dir = "../results/exp_cv_2/"

metrics = ["Logl", "Square_diff"]
dataset_ids = []
models_str = ["gt", "nf", "clg"]

data = {}
for metric in metrics:
    data[metric] = {}

for dataset_id in os.listdir(root_dir):
    dataset_path = os.path.join(root_dir, dataset_id)
    if os.path.isdir(dataset_path) and dataset_id != "plots" and dataset_id != "plots_analysis":
        dataset_ids.append(dataset_id)
        # Load the models
        models = {}
        for model in models_str:
            path = os.path.join(dataset_path, model + "_" + dataset_id + ".pkl")
            models[model] = pickle.load(open(path, "rb"))
        # Sample from the ground truth
        gt_samples = models["gt"].sample(1000, seed=0)
        X = gt_samples.drop(models["gt"].get_class_var_name(), axis=1)
        y = gt_samples[models["gt"].get_class_var_name()]
        # Get the log likelihoods, brier and auc for all the models
        loglik = pd.DataFrame(index=range(1000), columns=models_str)
        sqdiff = pd.DataFrame(index=range(1000), columns=models_str)
        for model in models_str:
            loglik[model] = pd.DataFrame(models[model].logl(X, y))
            predictions = models[model].predict_proba(X.to_numpy(), output="pandas")
            sqdiff[model] = pd.DataFrame(square_diff(y.to_numpy(), predictions))
        data["Logl"][dataset_id] = loglik
        data["Square_diff"][dataset_id] = sqdiff

values_dict = {"dataset_ids": dataset_ids}


# Create a plots folder
plots_dir = os.path.join(root_dir, "plots_analysis")
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Plots for each metric, putting al models together
clg_means = {}
clg_minus_nf_means = {}
for metric in metrics:
    nrows, ncols = close_factors(len(dataset_ids))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    # Create also a boxplot for each dataset
    fig_box, ax_box = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    for i,dataset_id in enumerate(dataset_ids) :
        data_new = data[metric][dataset_id].copy()
        if metric == "Logl":
            data_new = data_new * -1
        fbh = friedman_posthoc(data_new)
        sp.critical_difference_diagram(fbh["summary_ranks"], fbh["p_adjusted"],
                                        label_fmt_left="{label}", label_fmt_right="{label}",
                                        ax=ax[i // ncols, i % ncols],
                                       color_palette={"clg": "green", "nf": "orange", "gt": "blue"})
        # Set the title
        ax[i // ncols, i % ncols].set_title(dataset_id)
        # Additionally, Wilcoxon test for clg and nf
        #wxr = wilcoxon(data_new["clg"], data_new["nf"], alternative="greater")
        #print("Wilcoxon for", dataset_id, "is", wxr)
        # Boxplot
        # First, if we are analysing the logl, we normalising using the mean and std of gt
        if metric == "Logl":
            data_new = data_new * - 1
            data_new = (data_new - data_new["gt"].mean()) / data_new["gt"].std()
            clg_means[dataset_id] = data_new["clg"].mean()
            clg_minus_nf_means[dataset_id] = (data_new["clg"] - data_new["nf"]).mean()
        sns.boxplot(data=data_new, ax=ax_box[i // ncols, i % ncols], showfliers=False)
        ax_box[i // ncols, i % ncols].set_title(dataset_id)

    # Set global title
    fig.suptitle(metric)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, metric + "_cdd.pdf"))
    fig_box.suptitle(metric)
    fig_box.tight_layout()
    fig_box.savefig(os.path.join(plots_dir, metric + "_box.pdf"))

# Print the means of the clg. Descendent order
print("Means of clg")
for key, value in sorted(clg_means.items(), key=lambda item: item[1], reverse=True):
    print(key,":", value)
print()

# Print the means of the clg. Descendent order
print("Means of clg minus nf")
for key, value in sorted(clg_minus_nf_means.items(), key=lambda item: item[1], reverse=True):
    print(key, ":", value)


