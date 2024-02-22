import pybnesian as pb
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from bayesace.utils import *
from bayesace.algorithms.bayesace_algorithm import *
from itertools import product
import pickle
import os
import time


def mlog(x):
    return -np.log(x)


def check_bayesace(bayesian_network, dataset: pd.DataFrame, penalty, n_vertex, likelihood_thresh, acc_thresh):
    bayesace = BayesACE(bayesian_network=bayesian_network, features=df.columns[:-1], n_vertex=n_vertex, chunks=2,
                        penalty=penalty,
                        pop_size=50, likelihood_threshold=likelihood_thresh, accuracy_threshold=acc_thresh,
                        generations=5, verbose=False)
    res = bayesace.run(dataset.iloc[[0]])

    ## CHECK THE RESULTING_PATH
    cfx_path = res.path.values
    path_file = os.path.dirname(__file__) + "/bayesace_vars/cfxpath_" + str(penalty) + "_" + str(
        n_vertex) + "_" + str(likelihood_thresh) + "_" + str(acc_thresh) + ".pkl"
    #with open(path_file, "wb") as file :
    #   pickle.dump(cfx_path, file)
    with open(path_file, "rb") as file:
        assert (pickle.load(file) == cfx_path).all()

    ## CHECK THE RESULTING DISTANCE
    distance = res.distance
    path_file = os.path.dirname(__file__) + "/bayesace_vars/distance_" + str(penalty) + "_" + str(
        n_vertex) + "_" + str(likelihood_thresh) + "_" + str(acc_thresh) + ".pkl"
    #with open(path_file, "wb") as file :
    #   pickle.dump(distance, file)
    with open(path_file, "rb") as file:
        assert (pickle.load(file) == distance).all()


def round2(x):
    return np.round(x, 2)


if __name__ == "__main__":
    assert pb.__version__ == "0.4.3"
    t0 = time.time()
    np.random.seed(0)

    data_path = os.path.dirname(__file__) + "/test_dataset.csv"
    df = pd.read_csv(data_path)
    df["class"] = df["z"].astype('category')
    df = df.drop("z", axis=1)

    feature_columns = [i for i in df.columns if i != "class"]
    df[feature_columns] = StandardScaler().fit_transform(df[feature_columns].values)

    bn = pb.hc(df, bn_type=pb.CLGNetworkType(), operators=["arcs"], score="validated-lik", seed=0)
    bn.fit(df)

    list_pen = [0, 1]
    list_n_vertex = [0, 1]
    list_ll = [0, 0.00001,]
    list_acc = [0.5, 0.05]
    for i in product(list_pen, list_n_vertex, list_ll, list_acc):
        check_bayesace(bn, df, penalty = i[0], n_vertex=i[1], likelihood_thresh=i[2], acc_thresh=i[3])

    print(time.time() - t0)
    print("Bayes ACE tested succesfully")
