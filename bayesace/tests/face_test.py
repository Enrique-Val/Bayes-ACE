import pybnesian as pb
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from bayesace.utils import *
from bayesace.algorithms.face import *
from itertools import product
import pickle
import os
import time

def mlog(x) :
    return -np.log(x)

def check_graph(bayesian_network, dataset : pd.DataFrame, graph_type, distance_threshold, likelihood_thresh, acc_thresh) :
    #print(graph_type,distance_threshold)
    if graph_type == 'epsilon' :
        func = identity
    else :
        func = mlog

    face = FACE(bayesian_network, dataset.columns[:-1], 3, dataset.drop("class", axis = 1), distance_threshold,
                 graph_type,
                 f_tilde=func, likelihood_threshold=likelihood_thresh, accuracy_threshold=acc_thresh)
    cfx_path = round2(face.run(dataset.iloc[[0]]).path.values)

    edges = face.graph.edges(data = True)
    weights = round2(np.array(list(dict( (x[:-1], x[-1]["weight"]) for x in edges if "weight" in x[-1] ).values())))

    ## CHECK THE GRAPH
    path_file = os.path.dirname(__file__) + "/face_vars/graph_weights_"+str(graph_type)+"_"+str(distance_threshold)+ "_" + str(likelihood_thresh) + "_" + str(acc_thresh) + ".pkl"
    #with open(path_file, "wb") as file :
    #    pickle.dump(weights, file)
    with open(path_file, "rb") as file :
        assert (pickle.load(file) == weights).all()

    ## CHECK THE RESULTING_PATH
    path_file = os.path.dirname(__file__) + "/face_vars/cfxpath_" + str(graph_type) + "_" + str(
        distance_threshold) + "_" + str(likelihood_thresh) + "_" + str(acc_thresh) + ".pkl"
    #with open(path_file, "wb") as file :
    #    pickle.dump(cfx_path, file)
    with open(path_file, "rb") as file :
        assert (pickle.load(file) == cfx_path).all()

def round2(x):
    return np.round(x, 2)


if __name__ == "__main__":
    t0 = time.time()
    np.random.seed(0)

    df = pd.read_csv("./test_dataset.csv")
    df["class"] = df["z"].astype('category')
    df = df.drop("z", axis=1)
    df = df.sample(frac = 1).iloc[0:100].reset_index(drop=True)

    bn = pb.hc(df, bn_type=pb.CLGNetworkType(), operators=["arcs"], score="validated-lik", seed=0)
    bn.fit(df)

    list_gt = ["epsilon", "kde", "integral"]
    list_eps = [50,100,500]
    list_ll = [0,0.00001]
    list_acc = [1,0.5,0.05]
    for i in product(list_gt,list_eps, list_ll,list_acc) :
        check_graph(bn, df, i[0], distance_threshold = i[1], likelihood_thresh = i[2], acc_thresh = i[3])

    print(time.time() - t0)
    print("FACE tested succesfully")
