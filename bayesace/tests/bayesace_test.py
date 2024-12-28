import os
import sys

from bayesace.models.utils import PybnesianParallelizationError

sys.path.append(os.getcwd())
import pybnesian as pb
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from bayesace.algorithms.bayesace_algorithm import *
from itertools import product
import pickle
import os
import time


def mlog(x):
    return -np.log(x)


def check_copy(bn):
    return bn.fitted()


def get_naive_structure(df: pd.DataFrame, type):
    naive = type(df.columns)
    for i in [i for i in df.columns if i != "class"]:
        naive.add_arc("class", i)
    return naive


def check_bayesace(bayesian_network, dataset: pd.DataFrame, penalty, n_vertex, log_likelihood_threshold, acc_thresh):
    np.random.seed(0)
    bayesace = BayesACE(density_estimator=bayesian_network, features=df.columns[:-1], n_vertex=n_vertex, chunks=2,
                        penalty=penalty,
                        pop_size=50, log_likelihood_threshold=log_likelihood_threshold, posterior_probability_threshold=acc_thresh,
                        generations=5, verbose=False)
    res = bayesace.run(dataset.iloc[[0]], parallelize=True)

    ## CHECK THE RESULTING_PATH
    cfx_path = res.path.values
    path_file = os.path.dirname(__file__) + "/bayesace_vars/cfxpath_" + str(penalty) + "_" + str(
        n_vertex) + "_" + str(log_likelihood_threshold) + "_" + str(acc_thresh) + ".pkl"
    # with open(path_file, "wb") as file :
    #    pickle.dump(cfx_path, file)
    with open(path_file, "rb") as file:
        assert (pickle.load(file) == cfx_path).all()

    ## CHECK THE RESULTING DISTANCE
    distance = res.distance
    path_file = os.path.dirname(__file__) + "/bayesace_vars/distance_" + str(penalty) + "_" + str(
        n_vertex) + "_" + str(log_likelihood_threshold) + "_" + str(acc_thresh) + ".pkl"
    # with open(path_file, "wb") as file :
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

    start = get_naive_structure(df, pb.CLGNetwork)
    # Train a conditional linear Gaussian network
    # Because of a Pybnesian bug, the copy method does not work properly. We have to retrain the network in that case (we will try a maximum of 50 times)

    fitted_flag = False
    bn = None
    count = -1
    MAX_RETRIES = 50
    while not fitted_flag and count < MAX_RETRIES:
        bn = pb.hc(df, start=start, operators=["arcs"], score="bic", seed=0)
        bn.fit(df)
        pool = mp.Pool(1)
        res = pool.starmap(check_copy, [(bn,)])
        pool.close()
        fitted_flag = res[0]
        time.sleep(0.1)
    if not fitted_flag:
        raise PybnesianParallelizationError("BayesACE test failed in the parallelization efforts")

    list_pen = [0, 1]
    list_n_vertex = [0, 1]
    list_ll = [np.log(0), np.log(0.00001)]
    list_acc = [0.5, 0.05]
    for i in product(list_pen, list_n_vertex, list_ll, list_acc):
        check_bayesace(bn, df, penalty=i[0], n_vertex=i[1], log_likelihood_threshold=i[2], acc_thresh=i[3])

    print(time.time() - t0)
    print("Bayes ACE tested succesfully")
