import random
import csv
import os
import sys

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


def check_existance(data, bn, ll_thresh, acc_thresh):
    print("Length of data is ", len(data.index))
    data_no_class = data.drop("class", axis=1)
    ll = likelihood(data_no_class, bn)
    data_no_class = data_no_class[ll > ll_thresh]
    assert not data_no_class.empty
    for y in np.unique(data["class"]):
        acc = posterior_probability(x_cfx=data_no_class, y_og=y, bn=bn)
        assert (acc < acc_thresh).any()
        data_no_class_bis = data_no_class[acc < acc_thresh]
        print("Available", y, "cfx:", len(data_no_class_bis.index))


if __name__ == "__main__":
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset_id", nargs='?', default=44091, type=int)
    parser.add_argument("--network", nargs='?', default="CLG", type=str)
    args = parser.parse_args()

    dataset_id = args.dataset_id
    network_type = args.network

    for dataset_id in [44091, 44123, 44122, 44127, 44130]:
        print("Dataset:", dataset_id)
        for network_type in ["CLG"]:  # , "SP"]:
            print("Network:", network_type)
            random.seed(0)

            # Load the dataset
            df = get_data(dataset_id)

            # Split the dataset into train and test. Test only contains the 5 counterfactuals to be evaluated
            n_counterfactuals = 5
            df_train = df.head(len(df.index) - n_counterfactuals)
            df_test = df.tail(n_counterfactuals)

            network = None
            if network_type == 'CLG' or network_type == 'SP':
                network = hill_climbing(data=df_train, bn_type=network_type)
            elif network_type == "NN":
                args = Arguments()
                network = MultiBnaf(args, df_train)

            mean_logl, std_logl = get_mean_sd_logl(df_train, network_type, folds=10)

            np.random.seed(0)
            # Algorithm parameters (relatively high restriction on accuracy and likelihood)
            likelihood_threshold = mean_logl + 0*std_logl
            accuracy_threshold = 0.05
            n_vertices = [0]
            penalties = [1]
            chunks = 10
            for value in df_train["class"].cat.categories :
                df_class = df_train[df_train["class"] == value]
                opposite_class_mean_logl = np.mean(
                    [mean_logl[i] for i in mean_logl.keys() if i != instance["class"].values[0]])
                print("Opposite class", opposite_class_mean_logl)
                opposite_class_std_logl = np.sqrt(
                    np.mean([std_logl[i] for i in std_logl.keys() if i != instance["class"].values[0]]))
                likelihood_threshold = np.e ** (
                        opposite_class_mean_logl + likelihood_threshold_sigma * opposite_class_std_logl)
            check_existance(df_train, bn=network, acc_thresh=accuracy_threshold, ll_thresh=likelihood_threshold)
            print("Data exists!")

            '''np.seterr(divide='ignore')
            for penalty in penalties:
                # Result storage
                distances_mat = np.zeros((n_counterfactuals, len(n_vertices)))
                evaluations_mat = np.zeros((n_counterfactuals, len(n_vertices)))
                for i in range(0, n_counterfactuals):
                    distances = np.zeros(len(n_vertices))
                    evaluations = np.zeros(len(n_vertices))
                    for j, n_vertex in enumerate(n_vertices):
                        alg = BayesACE(bayesian_network=network, features=df_train.columns[:-1], n_vertex=n_vertex,
                                       accuracy_threshold=accuracy_threshold, likelihood_threshold=likelihood_threshold,
                                       chunks=chunks, penalty=penalty,
                                       seed=0, verbose=False)
                        result, res = alg.run(df_test.iloc[[i]], parallelize=True, return_info=True)
                        distances[j] = result.distance
                        evaluations[j] = res.algorithm.evaluator.n_eval
                    distances_mat[i] = distances
                    evaluations_mat[i] = evaluations

                print("Distances mat")
                print(distances_mat)
                print()

                distances_mean = distances_mat.mean(axis=0)
                distances_std = distances_mat.std(axis=0)
                evaluations_mean = evaluations_mat.mean(axis=0)
                evaluations_std = evaluations_mat.std(axis=0)'''

    print(time.time() - t0)
