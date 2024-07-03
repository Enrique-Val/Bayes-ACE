import random
import csv
import os
import sys
from itertools import product

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from bayesace.models.conditional_normalizing_flow import NormalizingFlowModel
from bayesace.models.multi_bnaf import Arguments, MultiBnaf

from bayesace.models.utils import hill_climbing, get_data, preprocess_train_data

sys.path.append(os.getcwd())
import argparse

import pandas as pd
import numpy as np
from pybnesian import hc, CLGNetworkType, SemiparametricBNType, SemiparametricBN, CLGNetwork
# from drawdata import draw_scatter
import matplotlib.pyplot as plt

from bayesace.utils import *
from bayesace.algorithms.bayesace_algorithm import BayesACE
from bayesace.algorithms.face import FACE

from sklearn.preprocessing import StandardScaler
import openml as oml

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


# Define the number of folds (K) and parameters of our grid search for the normalizing flow
k = 5
layers_list = [1, 2]
hid_units_list = [10, 20, 30, 40]
regularization_list = [0, 1e-4, 1e-3]

def cross_validate_nf(dataset, fold_indices = None, lr=None, wd=None, bins=None, hu=None, layers = None, n_flows = None) :
    logl = []
    brier = []
    auc_list = []
    times = []
    for train_index, test_index in fold_indices:
        df_train = dataset.iloc[train_index].reset_index(drop=True)
        df_train = preprocess_train_data(df_train)
        df_test = dataset.iloc[test_index].reset_index(drop=True)
        t0 = time.time()
        model = NormalizingFlowModel()
        model.train(df_train, lr=lr, weight_decay=wd, count_bins=bins, hidden_units=hu, hidden_layers=layers,
                    n_flows=n_flows)
        it_time = time.time() - t0
        times.append(it_time)
        logl.append(model.logl(df_test).mean())
        predictions = predict_class(df_test.drop("class", axis=1), model)
        brier.append(brier_score(df_test["class"].values, predictions))
        auc_list.append(auc(df_test["class"].values, predictions))
    return np.mean(logl), np.std(logl), np.mean(brier), np.std(brier), np.mean(auc_list) , np.std(auc_list), np.mean(times), np.std(times),  {"lr": lr, "weight_decay": wd, "bins": bins, "hidden_u": hu, "layers": layers, "n_flows": n_flows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset_id", nargs='?', default=44130, type=int)
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
            df_train = preprocess_train_data(df_train)
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

    # Print KDE
    bandwidths = np.logspace(-1, 0, 20)

    # Initialize to store best parameters and highest score
    best_bandwidth = None
    best_score = -np.inf

    # Iterate over each parameter combination
    for bandwidth in bandwidths:
        fold_scores = []

        # Iterate over each fold
        for train_indices, test_indices in fold_indices:
            X_train, X_test = df.iloc[train_indices], df.iloc[test_indices]
            X_train = X_train.drop(columns=["class"])
            X_test = X_test.drop(columns=["class"])
            X_train = preprocess_train_data(X_train)

            print(X_train)

            # Initialize and train the model
            kde = KernelDensity(bandwidth=bandwidth)
            kde.fit(X_train)

            # Evaluate the model on the validation set
            log_likelihood = kde.score(X_test)
            fold_scores.append(log_likelihood)

        # Calculate the mean score for the current parameter set
        mean_score = np.mean(fold_scores)

        # Update the best parameters if the current score is better
        if mean_score > best_score:
            best_score = mean_score
            best_bandwidth = bandwidth

    print(f"Optimal bandwidth: {best_bandwidth}")
    print(f"Log-likelihood: {best_score / (len(df) / k)}")

    # Validate normalizing flow with different params
    dataset = df
    d = len(dataset.columns)
    # Define the parameter grid
    param_grid = {
        "lr": [1e-2, 1e-3, 1e-4],
        "weight_decay": [0, 1e-4, 1e-3],
        "bins": [2, 4, 6],
        "hidden_u": [2 * d, 5 * d, 10 * d],
        "layers": [1, 2],
        "n_flows": [1, 2, 4]
    }
    param_grid = {
        "lr": [1e-2],
        "weight_decay": [0],
        "bins": [2],
        "hidden_u": [2*d],
        "layers": [1],
        "n_flows": [1,2]
    }

    # Create a list of all parameter combinations
    param_combinations = list(
        product(param_grid["lr"], param_grid["weight_decay"], param_grid["bins"], param_grid["hidden_u"],
                param_grid["layers"], param_grid["n_flows"]))

    '''for i in param_combinations:
        train_and_evaluate(dataset, dataset_test, *i)'''

    # Use multiprocessing to speed up the grid search
    print(mp.cpu_count() - 2)
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(cross_validate_nf, [(dataset, fold_indices, lr, wd, bins, hu, layers, n_flows) for
                                                    lr, wd, bins, hu, layers, n_flows in param_combinations])
    nf_logl_means, nf_logl_stds, nf_brier_means,nf_brier_stds, nf_auc_means,nf_auc_stds,nf_time_means,nf_time_stds, params = zip(*results)
    for i in range(len(nf_logl_means)):
        mean_logl.append(nf_logl_means[i])
        std_logl.append(nf_logl_stds[i])
        mean_brier.append(nf_brier_means[i])
        std_brier.append(nf_brier_stds[i])
        auc_mean.append(nf_auc_means[i])
        auc_std.append(nf_auc_stds[i])
        time_mean.append(nf_time_means[i])
        time_std.append(nf_time_stds[i])
        labels.append("NF"+str(params[i]))


    to_ret = pd.DataFrame(data=[mean_logl, std_logl, mean_brier, std_brier, auc_mean, auc_std, time_mean, time_std], columns=labels,
                          index=["mean_logl", "std_logl", "mean_brier", "std_brier", "auc_mean", "auc_std", "time_mean", "time_std"])
    print(to_ret)
    to_ret.to_csv('./results/exp_cv_2/data' + str(dataset_id) + '.csv')
