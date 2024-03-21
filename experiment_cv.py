import random
import csv
import os
import sys

from models.multi_bnaf import Arguments, MultiBnaf

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

# Define the number of folds (K)
k = 2
layers_list = [1,2]
hid_units_list = [10,20]#,30,40]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset_id", nargs='?', default=44122, type=int)
    args = parser.parse_args()

    dataset_id = args.dataset_id

    random.seed(0)

    # Load the dataset
    df = get_and_process_data(dataset_id)

    # Get the fold indices
    fold_indices = kfold_indices(df, k)
    mean_logls = np.zeros(2 + 2 * 2)
    std_logl = np.zeros(2 + 2 * 2)
    labels =  []

    # Validate Gaussian Bayesian network
    slogl = []
    for train_index, test_index in fold_indices:
        df_train = df.iloc[train_index].reset_index(drop=True)
        df_test = df.iloc[test_index].reset_index(drop = True)
        network = hill_climbing(data=df_train, bn_type="CLG")
        slogl_i = network.logl(df_test).mean()
        slogl.append(slogl_i)
    mean_logls[0] = np.mean(slogl)
    labels.append("CLG")
    print(np.mean(slogl))

    # Validate a SP Bayesian network
    slogl = []
    for train_index, test_index in fold_indices:
        df_train = df.iloc[train_index].reset_index(drop=True)
        df_test = df.iloc[test_index].reset_index(drop=True)
        network = hill_climbing(data=df_train, bn_type="SP")
        slogl_i = network.logl(df_test).mean()
        slogl.append(slogl_i)
    mean_logls[1] = np.mean(slogl)
    labels.append("SP")
    print(np.mean(slogl))

    # Validate a SP Bayesian network
    '''slogl = []
    for train_index, test_index in fold_indices:
        df_train = df.iloc[train_index].reset_index(drop=True)
        df_test = df.iloc[test_index].reset_index(drop=True)
        network = hill_climbing(data=df_train, bn_type="SP", score = "cv-lik")
        slogl_i = network.logl(df_test).mean()
        slogl.append(slogl_i)
    mean_logls[2] = np.mean(slogl)
    print(np.mean(slogl))'''

    # Validate normalizing flow with different paramas
    '''count = 2
    for layers in layers_list :
        for hidden_units in hid_units_list :
            slogl = []
            args = Arguments()
            args.layers = layers
            args.hidden_sim= hidden_units
            args.load =False
            args.save = False
            for train_index, test_index in fold_indices:
                df_train = df.iloc[train_index].reset_index(drop=True)
                df_test = df.iloc[test_index].reset_index(drop=True)
                mbnaf = MultiBnaf(args, df_train)
                slogl_i = mbnaf.logl(df_test).mean()
                slogl.append(slogl_i)
            mean_logls[count] = np.mean(slogl)
            labels.append("BNAF_l"+str(layers)+"_hu"+str(hidden_units))
            count += 1


    print(mean_logls)
    to_ret = pd.DataFrame(data=[mean_logls], columns=labels)
    print(to_ret)
    to_ret.to_csv('./results/exp_cv/mlogl' + str(dataset_id) + '.csv')'''


