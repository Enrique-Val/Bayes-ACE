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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset_id", nargs='?', default=44091, type=int)
    parser.add_argument("--network", nargs='?', default="CLG", type=str)
    args = parser.parse_args()

    dataset_id = args.dataset_id
    network_type = args.network

    random.seed(0)

    # Load the dataset
    df = get_data(dataset_id)

    # Split the dataset into train and test. Test only contains the 5 counterfactuals to be evaluated
    n_counterfactuals = 5
    df_train = df.head(len(df.index) - n_counterfactuals)
    df_test = df.tail(n_counterfactuals)

    network = hill_climbing(data=df_train, bn_type=network_type)

    np.random.seed(0)
    # Algorithm parameters (relatively high restriction on accuracy and likelihood)
    likelihood_threshold = 0.2 ** (len(df_train.columns) - 1)
    accuracy_threshold = 0.2
    likelihoods= [0.1, 0.2, 0.3]
    accs = [0.25,0.2,0.15]
    # Number of points for approximating integrals:
    chunks = 10

    # Launch a baseline


    np.seterr(divide='ignore')

    cfx_distance = 0
    data_ll = 0

    for likelihood_threshold in likelihoods :
        for accuracy_threshold in accs :
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
        print("Evaluations mat")
        print(evaluations_mat)
        print()
        print()


