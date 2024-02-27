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

import time


def get_naive_structure(df: pd.DataFrame):
    naive = SemiparametricBN(df.columns)
    for i in [i for i in df.columns if i != "class"]:
        naive.add_arc("class", i)
    return naive


if __name__ == "__main__":
    random.seed(0)

    # Load the dataset
    df = oml.datasets.get_dataset(44091, download_data=True).get_data()[0]

    # Shuffle the dataset
    df = df.sample(frac=1)

    # Transform the class into a categorical variable
    df["class"] = df[df.columns[0]].astype('string').astype('category')
    df = df.drop(df.columns[0], axis=1)

    # Scale the rest of the dataset
    feature_columns = [i for i in df.columns if i != "class"]
    df[feature_columns] = StandardScaler().fit_transform(df[feature_columns].values)

    # Split the dataset into train and test. Test only contains the 5 counterfactuals to be evaluated
    df_train = df.head(len(df.index) - 5)
    df_test = df.tail(5)

    # Train a conditional linear Gaussian network
    clg_network = hc(df_train, bn_type=CLGNetworkType(), operators=["arcs"], score="bic", seed=0)
    clg_network.fit(df_train)

    # Train a semiparametric network
    start = get_naive_structure(df_train)
    # spb_network = hc(df_train, start=start, operators=["node_type", "arcs"], score="validated-lik", seed=0)
    # spb_network.fit(df_train)

    # Algorithm parameters
    likelihood_thresholds = [0.2, 0.1, 0.05]
    accuracy_thresholds = [0.1, 0.05, 0.01]
    chunks = 10

    # Result storage
    diff_distance_baseline = None
    cf_distance = None
    optimised_path = None
    exec_time = None

    # Launch baseline