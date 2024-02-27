import random

import pandas as pd
import numpy as np
from pybnesian import hc, CLGNetworkType, SemiparametricBNType, SemiparametricBN
# from drawdata import draw_scatter
import matplotlib.pyplot as plt

from bayesace.utils import *
from bayesace.algorithms.bayesace_algorithm import BayesACE
from bayesace.algorithms.face import FACE

from sklearn.preprocessing import StandardScaler
import openml as oml

import multiprocessing as mp

import time


def get_naive_structure(df: pd.DataFrame):
    naive = SemiparametricBN(df.columns)
    for i in [i for i in df.columns if i != "class"]:
        naive.add_arc("class", i)
    return naive

def check_copy(bn) :
    return bn.fitted()



if __name__ == "__main__":
    random.seed(0)

    # Load the dataset
    df = oml.datasets.get_dataset(44091, download_data=True).get_data()[0]

    # Shuffle the dataset
    df = df.sample(frac=1, random_state= 0)

    # Transform the class into a categorical variable
    df["class"] = df[df.columns[-1]].astype('string').astype('category')
    df = df.drop(df.columns[-2], axis=1)

    # Scale the rest of the dataset
    feature_columns = [i for i in df.columns if i != "class"]
    df[feature_columns] = StandardScaler().fit_transform(df[feature_columns].values)

    # Split the dataset into train and test. Test only contains the 5 counterfactuals to be evaluated
    df_train = df.head(len(df.index) - 5)
    df_test = df.tail(5)

    # Train a conditional linear Gaussian network
    fitted_flag = False
    clg_network = None
    while not fitted_flag :
        clg_network = hc(df_train, bn_type=CLGNetworkType(), operators=["arcs"], score="bic", seed=0)
        # Because of a Pybnesian bug, the copy method does not work properly. We have to retrain the network in that case
        clg_network.fit(df_train)
        pool = mp.Pool(1)
        res = pool.starmap(check_copy, [(clg_network,)])
        pool.close()
        fitted_flag = res[0]

    # Train a semiparametric network
    start = get_naive_structure(df_train)
    # spb_network = hc(df_train, start=start, operators=["node_type", "arcs"], score="validated-lik", seed=0)
    # spb_network.fit(df_train)

    # Algorithm parameters (relatively low restriction on accuracy and likelihood)
    likelihood_threshold = 0.05**11
    accuracy_threshold = 0.1
    n_vertices = [0, 1, 2, 3, 4, 5, 10, 15, 20, 30]
    chunks = 10

    # Result storage
    optimised_path = None
    n_evals = None


    for n_vertex in n_vertices[0:1]:
        alg = BayesACE(bayesian_network=clg_network, features=df_train.columns[:-1], n_vertex=n_vertex,
                       accuracy_threshold=accuracy_threshold, likelihood_threshold=likelihood_threshold, chunks=chunks, seed = 0)
        result, res = alg.run(df_test.iloc[[0]], parallelize=True, return_info=True)
        print(res.F)
        print(result.distance)
    # Launch baseline
