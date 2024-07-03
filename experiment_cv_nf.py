import os
import random
import sys
from itertools import product

from bayesace.models.utils import get_data, preprocess_train_data

sys.path.append(os.getcwd())
import argparse

from bayesace.utils import *

import multiprocessing as mp

import time


def kfold_indices(data, k):
    fold_size = len(data) // k
    indices = np.arange(len(data))
    folds = []
    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        folds.append((train_indices, test_indices))
    return folds


# Define the number of folds (K)
k = 10

# Define how the preprocessing will be done
JIT_COEF = 0.3
ELIM_OUTL = True

# Define the possible parameter values for the NF
# The number of hidden units will be multiplied by number of features
param_grid = {
    "lr": [1e-2, 1e-3, 1e-4],
    "weight_decay": [0, 1e-4, 1e-3],
    "bins": [2, 4, 6],
    "hidden_u": [2, 5, 10],
    "layers": [1, 2],
    "n_flows": [1, 2, 4]
}


def cross_validate_nf(dataset, fold_indices=None, lr=None, wd=None, bins=None, hu=None, layers=None, n_flows=None):
    logl = []
    brier = []
    auc_list = []
    times = []
    for train_index, test_index in fold_indices:
        df_train = dataset.iloc[train_index].reset_index(drop=True)
        df_train = preprocess_train_data(df_train, jit_coef=JIT_COEF, eliminate_outliers=ELIM_OUTL)
        df_test = dataset.iloc[test_index].reset_index(drop=True)
        t0 = time.time()
        model = NormalizingFlowModel()
        model.train(df_train, lr=lr, weight_decay=wd, count_bins=bins, hidden_units=hu*len(df_train.columns)-1, hidden_layers=layers,
                    n_flows=n_flows)
        it_time = time.time() - t0
        times.append(it_time)
        logl.append(model.logl(df_test).mean())
        predictions = predict_class(df_test.drop("class", axis=1), model)
        brier.append(brier_score(df_test["class"].values, predictions))
        auc_list.append(auc(df_test["class"].values, predictions))
    print(str( {"lr": lr, "weight_decay": wd, "bins": bins, "hidden_u": hu, "layers": layers,
                                "n_flows": n_flows}), "normalizing flow learned")
    return np.mean(logl), np.std(logl), np.mean(brier), np.std(brier), np.mean(auc_list), np.std(auc_list), np.mean(
        times), np.std(times), {"lr": lr, "weight_decay": wd, "bins": bins, "hidden_u": hu, "layers": layers,
                                "n_flows": n_flows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset_id", nargs='?', default=44091, type=int)
    args = parser.parse_args()

    dataset_id = args.dataset_id
    print("Cross validation dataset: ", dataset_id)

    random.seed(0)

    # Load the dataset
    df = get_data(dataset_id)

    # Get the fold indices
    fold_indices = kfold_indices(df, k)
    mean_logl = []
    std_logl = []
    mean_brier = []
    std_brier = []
    auc_mean = []
    auc_std = []
    time_mean = []
    time_std = []
    labels = []

    # Validate Bayesian networks
    for network_type in ["CLG"]:
        slogl = []
        brier = []
        aucs = []
        times = []
        count = 0
        for train_index, test_index in fold_indices:
            df_train = df.iloc[train_index].reset_index(drop=True)
            df_train = preprocess_train_data(df_train, jit_coef=JIT_COEF, eliminate_outliers=ELIM_OUTL)
            df_test = df.iloc[test_index].reset_index(drop=True)
            t0 = time.time()
            network = hill_climbing(data=df_train, bn_type=network_type)
            times.append(time.time() - t0)
            slogl_i = network.logl(df_test).mean()
            slogl.append(slogl_i)
            predictions = predict_class(df_test.drop("class", axis=1), network)
            brier_i = brier_score(df_test["class"].values, predictions)
            brier.append(brier_i)
            auc_i = auc(df_test["class"].values, predictions)
            aucs.append(auc_i)
        mean_logl.append(np.mean(slogl))
        std_logl.append(np.std(slogl))
        mean_brier.append(np.mean(brier))
        std_brier.append(np.std(brier))
        auc_mean.append(np.mean(aucs))
        auc_std.append(np.std(aucs))
        time_mean.append(np.mean(times))
        time_std.append(np.std(times))
        labels.append(network_type)

    print("Bayesian Networks learned")

    # Validate normalizing flow with different params
    dataset = df
    d = len(dataset.columns) - 1

    # Create a list of all parameter combinations
    param_combinations = list(
        product(param_grid["lr"], param_grid["weight_decay"], param_grid["bins"], param_grid["hidden_u"],
                param_grid["layers"], param_grid["n_flows"]))

    results = []
    for i in param_combinations:
        results.append(cross_validate_nf(dataset, fold_indices, *i))
    nf_logl_means, nf_logl_stds, nf_brier_means, nf_brier_stds, nf_auc_means, nf_auc_stds, nf_time_means, nf_time_stds, params = zip(
        *results)
    for i in range(len(nf_logl_means)):
        mean_logl.append(nf_logl_means[i])
        std_logl.append(nf_logl_stds[i])
        mean_brier.append(nf_brier_means[i])
        std_brier.append(nf_brier_stds[i])
        auc_mean.append(nf_auc_means[i])
        auc_std.append(nf_auc_stds[i])
        time_mean.append(nf_time_means[i])
        time_std.append(nf_time_stds[i])
        labels.append("NF" + str(params[i]))

    to_ret = pd.DataFrame(data=[mean_logl, std_logl, mean_brier, std_brier, auc_mean, auc_std, time_mean, time_std],
                          columns=labels,
                          index=["mean_logl", "std_logl", "mean_brier", "std_brier", "auc_mean", "auc_std", "time_mean",
                                 "time_std"])
    print(to_ret)
    to_ret.to_csv('./results/exp_cv_2/data' + str(dataset_id) + '.csv')
