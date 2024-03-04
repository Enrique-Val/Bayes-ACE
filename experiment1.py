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


def get_naive_structure(df: pd.DataFrame, type):
    naive = SemiparametricBN(df.columns)
    for i in [i for i in df.columns if i != "class"]:
        naive.add_arc("class", i)
    return naive


def check_copy(bn):
    return bn.fitted()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset_id", nargs='?', default=44091, type=int)
    parser.add_argument("--network", nargs='?', default="G", type=str)
    args = parser.parse_args()

    dataset_id = args.dataset_id
    network_type = args.network

    random.seed(0)

    # Load the dataset
    df = oml.datasets.get_dataset(dataset_id, download_data=True, download_qualities=False,
                                  download_features_meta_data=False).get_data()[0]

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=0)

    # Transform the class into a categorical variable
    df["class"] = df[df.columns[-1]].astype('string').astype('category')
    df = df.drop(df.columns[-2], axis=1)

    # Scale the rest of the dataset
    feature_columns = [i for i in df.columns if i != "class"]
    df[feature_columns] = StandardScaler().fit_transform(df[feature_columns].values)

    # Split the dataset into train and test. Test only contains the 5 counterfactuals to be evaluated
    n_counterfactuals = 5
    df_train = df.head(len(df.index) - n_counterfactuals)
    df_test = df.tail(n_counterfactuals)

    network = None
    # Train a conditional linear Gaussian network
    if network_type == "G":
        clg_network = hc(df_train, start=get_naive_structure(df_train, CLGNetwork), operators=["arcs"], score="bic",
                         seed=0)
        clg_network.fit(df_train)
        # Because of a Pybnesian bug, the copy method does not work properly at times. We have to shut down the experiment in that case
        pool = mp.Pool(1)
        res = pool.starmap(check_copy, [(clg_network,)])
        pool.close()
        if not res[0]:
            raise PybnesianParallelizationError(
                "As of version 0.4.3, PyBnesian Bayesian networks have internal and stochastic problems with the method "
                "\"copy()\"."
                "As such, the CLG network is not parallelized correctly and experiments cannot be launched.")
        network = clg_network

    if network_type == "S":
        # Train a semiparametric network
        spb_network = hc(df_train, start=get_naive_structure(df_train, SemiparametricBN),
                         operators=["node_type", "arcs"],
                         score="validated-lik", seed=0)
        spb_network.fit(df_train)
        # Because of a Pybnesian bug, the copy method does not work properly at times. We have to shut down the experiment in that case
        pool = mp.Pool(1)
        res = pool.starmap(check_copy, [(spb_network,)])
        pool.close()
        if not res[0]:
            raise PybnesianParallelizationError(
                "As of version 0.4.3, PyBnesian Bayesian networks have internal and stochastic problems with the method "
                "\"copy()\"."
                "As such, the SPB network is not parallelized correctly and experiments cannot be launched.")
        network = spb_network

    np.random.seed(0)
    # Algorithm parameters (relatively high restriction on accuracy and likelihood)
    likelihood_threshold = 0.2 ** (len(df_train.columns) - 1)
    accuracy_threshold = 0.01
    n_vertices = [0, 1, 2, 3, 4, 5]
    penalties = [1,2,3,4,5,7.5,10]#,15]
    chunks = 10
    
    np.seterr(divide='ignore')
    for penalty in penalties:
        # Result storage
        distances_mat = np.zeros((n_counterfactuals, len(n_vertices)))
        evaluations_mat = np.zeros((n_counterfactuals, len(n_vertices)))
        print("Distances raw")
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
            print(distances)
            distances -= distances.min()
            distances /= distances.ptp()
            distances_mat[i] = distances
            evaluations_mat[i] = evaluations
        print()

        print("Distances mat")
        print(distances_mat)
        print()
        print("Evaluations mat")
        print(evaluations_mat)
        print()
        print()

        distances_mean = distances_mat.mean(axis=0)
        distances_std = distances_mat.std(axis=0)
        evaluations_mean = evaluations_mat.mean(axis=0)
        evaluations_std = evaluations_mat.std(axis=0)

        with open('./results/exp_1/data'+str(dataset_id)+'_net'+network_type+'_penalty'+str(penalty)+'.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(distances_mean)
            w.writerow(evaluations_mean)
            w.writerow(distances_std)
            w.writerow(evaluations_std)
