import random
import csv
import os
import sys

from models.multi_bnaf import Arguments
from models.utils import hill_climbing

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


def get_best_nn(dataset_id):
    # TODO For now it is a dummy function
    cv_results = pd.read_csv('./results/exp_cv/' + dataset_id)


if __name__ == "__main__":
    # ALGORITHM PARAMETERS The likelihood parameter is relative. I.e. the likelihood threshold will be the mean logl
    # for that class plus "likelihood_threshold_sigma" sigmas of the logl std
    likelihood_threshold_sigma = 0
    accuracy_threshold = 0.2
    n_vertices = [0, 1, 2, 3, 4]
    penalties = [1, 5, 10, 15, 20]
    # Number of points for approximating integrals:
    chunks = 10
    # Number of counterfactuals
    n_counterfactuals = 10

    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset_id", nargs='?', default=44091, type=int)
    parser.add_argument("--network", nargs='?', default="CLG", type=str)
    args = parser.parse_args()

    dataset_id = args.dataset_id
    network_type = args.network

    random.seed(0)

    # Load the dataset
    df = get_data(dataset_id)

    # Get the bounds for the optimization problem. The initial sampling will rely on this, so we call it sampling_range
    xu = df.drop(columns=['class']).max().values + 0.0001
    xl = df.drop(columns=['class']).min().values - 0.0001
    sampling_range = (xl, xu)

    # Split the dataset into train and test. Test only contains the n_counterfactuals counterfactuals to be evaluated
    df_train = df.head(len(df.index) - n_counterfactuals)
    df_test = df.tail(n_counterfactuals)

    network = None
    if network_type == 'CLG' or network_type == 'SP':
        network = hill_climbing(data=df_train, bn_type=network_type)
    elif network_type == "NN":
        args = Arguments()
        network = MultiBnaf(args, df_train)

    print("Train mean", network.logl(df_train).mean())
    print("Test", network.logl(df_test).mean())
    print("Test median", np.median(network.logl(df_train)))
    print("Test std",network.logl(df_train).std())
    print("Test p90", np.percentile(network.logl(df_train),90))
    # mean_logl, std_logl = get_mean_sd_logl(df_train, network_type, folds=2)

    print("COMPARE MBNAF AND KDE")
    print("Test mean mbnaf", np.log(likelihood(df_test,network)).mean())
    print("Test mean kde", network.sampler.score_samples(df_test.drop("class",axis=1)).mean())

    mean_logl, std_logl = (dict(), dict())
    for i in df_train["class"].cat.categories:
        df_class = df_train[df_train["class"] == i]
        logls = network.logl(df_class)
        mean_logl[i] = logls.mean()
        std_logl[i] = logls.std()

    np.random.seed(0)

    np.seterr(divide='ignore')
    for penalty in penalties:
        # Result storage
        distances_mat = np.zeros((n_counterfactuals, len(n_vertices)))
        evaluations_mat = np.zeros((n_counterfactuals, len(n_vertices)))
        for i in range(0, n_counterfactuals):
            instance = df_test.iloc[[i]]
            print(instance["class"].values[0])
            print(mean_logl)
            opposite_class_mean_logl = np.mean(
                [mean_logl[i] for i in mean_logl.keys() if i != instance["class"].values[0]])
            print("Opposite class", opposite_class_mean_logl)
            opposite_class_std_logl = np.sqrt(
                np.mean([std_logl[i] for i in std_logl.keys() if i != instance["class"].values[0]]))
            likelihood_threshold = np.e ** (
                        opposite_class_mean_logl + likelihood_threshold_sigma * opposite_class_std_logl)
            distances = np.zeros(len(n_vertices))
            evaluations = np.zeros(len(n_vertices))
            for j, n_vertex in enumerate(n_vertices):
                alg = BayesACE(bayesian_network=network, features=df_train.columns[:-1], n_vertex=n_vertex,
                               accuracy_threshold=accuracy_threshold, likelihood_threshold=likelihood_threshold,
                               chunks=chunks, penalty=penalty, sampling_range=sampling_range, initialization="default",
                               seed=0, verbose=True, pop_size=100)
                result, res = alg.run(instance, parallelize=True, return_info=True)
                # print(result.distance)
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

        '''to_ret = pd.DataFrame(data=distances_mat, columns=n_vertices)
        to_ret.to_csv('./results/exp_1/distances_data' + str(dataset_id) + '_net' + network_type + '_penalty' + str(penalty) + '.csv')

        to_ret = pd.DataFrame(data=evaluations_mat, columns=n_vertices)
        to_ret.to_csv('./results/exp_1/evaluations_data' + str(dataset_id) + '_net' + network_type + '_penalty' + str(penalty) + '.csv')
        '''
