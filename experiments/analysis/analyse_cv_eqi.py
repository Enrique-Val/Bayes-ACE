import os
import pickle
from itertools import product

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon
import scikit_posthocs as sp

from bayesace import brier_score, auc, square_diff
from bayesace.models.conditional_kde import ConditionalKDE
from bayesace.models.lingam_cat import LingamClassifier
from experiments.analysis.analyse_exp import aggregate_data
from experiments.utils import friedman_posthoc, close_factors
import matplotlib.pyplot as plt
import seaborn as sns

# Path to dataset root
root_dir = "../results/exp_eqi/"

metrics = ["Negative log-likelihood", "Class prob. squared error"]
dataset_ids = []
models_str = ["nf", "lingam_simple"]
NF_STR_PLOT = "Normalizing\n flow"
BN_STR_PLOT = "Structured\n LinGaM\n classifier"
models_plot_str = [NF_STR_PLOT, BN_STR_PLOT]

palette={
    NF_STR_PLOT: "tab:orange",
    BN_STR_PLOT: "tab:green"
}

data = {}


# Load the models
models = {}
models_path = os.path.join(root_dir, "models")
data_path = os.path.join(root_dir, "data_processed")
for model, model_plot in zip(models_str, models_plot_str):
    path = os.path.join(models_path, model + ".pkl")
    models[model_plot] = pickle.load(open(path, "rb"))
cvar_name = models[models_plot_str[0]].get_class_var_name()
lingam_model : LingamClassifier = models[BN_STR_PLOT]
# Print all the noise dist types and params
print("Lingam noise dist types and params:")
adj = lingam_model.lingam_model.adjacency_matrix_
# Count number of non-zero entries in each row of the adjacency matrix
num_parents = np.sum(adj != 0, axis=1)
for i, var in enumerate(lingam_model.columns):
    print(f"{var}: {num_parents[i]} parents")

print("Lingam noise dist params:")
dist = lingam_model.noise_config_
vars = lingam_model.columns + [lingam_model.get_class_var_name()]
type_distribution = {"Normal" : 0, "Laplace" : 0, "StudentT" : 0, "Logistic" : 0, "2GMM" : 0, "3GMM" : 0}
for i, var in zip(dist,vars):
    print(f"{var}: {dist[i][0]} with params {dist[i][1]}")
    type_distribution[dist[i][0]] += 1

# Plot the distribution of noise types across variables
plt.figure(figsize=(6,4))
sns.barplot(x=list(type_distribution.keys()), y=list(type_distribution.values()), palette="Set2")
plt.title("Distribution of Noise Types in LinGaM Classifier")
plt.xlabel("Noise Distribution Type")
plt.ylabel("Number of Variables")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
raise ValueError

test_data = pd.read_csv(os.path.join(data_path, "data_test.csv"), index_col=0)
test_data[cvar_name] = (test_data[models[models_plot_str[0]].get_class_var_name()].
                                                              astype("str").astype("category"))
X = test_data.drop(columns=[cvar_name, cvar_name+"_cont"], axis=1)
y = test_data[models[models_plot_str[0]].get_class_var_name()]
# Scale X
scaler = pickle.load(open(os.path.join(models_path, "scaler.pkl"), "rb"))
X = pd.DataFrame(scaler.transform(X), columns=X.columns)
# Get the log likelihoods, brier and auc for all the models
loglik = pd.DataFrame(index=X.index, columns=models_plot_str)
sqdiff = pd.DataFrame(index=X.index, columns=models_plot_str)
for model in models_plot_str:
    loglik[model] = pd.DataFrame(models[model].logl(X, y))
    predictions = models[model].predict_proba(X.to_numpy(), output="pandas")
    sqdiff[model] = pd.DataFrame(square_diff(y.to_numpy(), predictions))
data["Negative log-likelihood"] = loglik
data["Class prob. squared error"] = sqdiff

# Create a plots folder
plots_dir = os.path.join(root_dir, "plots_cv")
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Box plot for all models, aggregated over all datasets
for metric in metrics:
    fig = plt.figure(figsize = (3.5*1.2,2.5*1.2))
    data_new = data[metric].copy()
    if metric == "Negative log-likelihood" :
        data_new = data_new * -1
    ax = fig.gca()
    sns.boxplot(data=data_new, showfliers=False, ax=ax, palette=palette)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel(metric)
    # Label with the number of samples
    ax.text(0.5, 0.95, "n = " + str(data_new.shape[0]), horizontalalignment='center',
            verticalalignment='center', fontsize = 10)
    print("n", str(data_new.shape[0]))
    # Label with the median
    for i, model in enumerate(models_plot_str):
        ax.text(i, data_new[model].median()+0.03, str(round(data_new[model].median(), 2)), horizontalalignment='center',
                verticalalignment='center', color='black', fontsize = 10)
    #fig.savefig(os.path.join(plots_dir, metric + "_box_all.pdf"))
    plt.show()
    wxr = wilcoxon(data_new[NF_STR_PLOT], data_new[BN_STR_PLOT], alternative="two-sided")
    print("Wilcoxon is", wxr, "for", metric, "alternative less")


