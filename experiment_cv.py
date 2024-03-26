import random
import csv
import os
import sys

from bayesace.models.multi_bnaf import Arguments, MultiBnaf

from bayesace.models.utils import hill_climbing

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
k = 10
layers_list = [1, 2]
hid_units_list = [10, 20, 30, 40]
regularization_list = [0, 1e-4, 1e-3]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset_id", nargs='?', default=44130, type=int)
    args = parser.parse_args()

    dataset_id = args.dataset_id
    print("Cross validation dataset: ", dataset_id)

    random.seed(0)

    # Load the dataset
    df = get_and_process_data(dataset_id)

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
    for network_type in ["CLG", "SP"]:
        slogl = []
        brier = []
        aucs = []
        times = []
        count = 0
        for train_index, test_index in fold_indices:
            df_train = df.iloc[train_index].reset_index(drop=True)
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

    # Validate normalizing flow with different params
    for reg in regularization_list :
        for layers in layers_list:
            for hidden_units in hid_units_list:
                slogl = []
                brier = []
                times = []
                aucs = []
                bnaf_args = Arguments()
                bnaf_args.load = False
                bnaf_args.save = False
                bnaf_args.tensorboard = False
                bnaf_args.test_data = False
                # bnaf_args.device = "cpu"
                bnaf_args.layers = layers
                bnaf_args.hidden_dim = hidden_units
                bnaf_args.weight_decay = reg
                for train_index, test_index in fold_indices:
                    df_train = df.iloc[train_index].reset_index(drop=True)
                    df_test = df.iloc[test_index].reset_index(drop=True)
                    t0 = time.time()
                    mbnaf = MultiBnaf(bnaf_args, df_train, parallelize=True)
                    times.append(time.time() - t0)
                    slogl_i = mbnaf.logl(df_test).mean()
                    slogl.append(slogl_i)
                    predictions = predict_class(df_test.drop("class", axis=1), mbnaf)
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
                labels.append("MBNAF_reg"+str(reg)+"_l" + str(layers) + "_hu" + str(hidden_units))

    '''for layers in layers_list:
        for hidden_units in hid_units_list:
            slogl = []
            brier = []
            times = []
            aucs = []
            bnaf_args.layers = layers
            bnaf_args.hidden_dim = hidden_units
            for train_index, test_index in fold_indices:
                df_train = df.iloc[train_index].reset_index(drop=True)
                df_test = df.iloc[test_index].reset_index(drop=True)
                t0 = time.time()
                mbnaf = BnafJoint(bnaf_args, df_train)
                times.append(time.time() - t0)
                slogl_i = mbnaf.logl(df_test.copy()).mean()
                slogl.append(slogl_i)
                predictions = predict_class(df_test.drop("class", axis=1), mbnaf)
                brier_i = brier_score(df_test["class"].values, predictions)
                brier.append(brier_i)
                auc_i = auc(df_test["class"].values, predictions)
                aucs.append(auc_i)
                print(slogl_i)
                print(brier_i)
                new = predict_class(df_test, network)
                new["class"] = df_test["class"]
                print(new)
                break
            mean_logl.append(np.mean(slogl))
            std_logl.append(np.std(slogl))
            mean_brier.append(np.mean(brier))
            std_brier.append(np.std(brier))
            auc_mean.append(np.mean(aucs))
            auc_std.append(np.std(aucs))
            time_mean.append(np.mean(times))
            time_std.append(np.std(times))
            labels.append("JointBNAF_l" + str(layers) + "_hu" + str(hidden_units))'''

    to_ret = pd.DataFrame(data=[mean_logl, std_logl, mean_brier, std_brier, auc_mean, auc_std, time_mean, time_std], columns=labels,
                          index=["mean_logl", "std_logl", "mean_brier", "std_brier", "auc_mean", "auc_std", "time_mean", "time_std"])
    print(to_ret)
    to_ret.to_csv('./results/exp_cv/data' + str(dataset_id) + '.csv')
