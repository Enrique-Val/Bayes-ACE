import pandas as pd
import numpy as np
import pybnesian as pb
import warnings
import math
import multiprocessing as mp
import openml as oml
from sklearn.preprocessing import StandardScaler


def identity(x):
    return x


class PybnesianParallelizationError(Exception):
    pass


def separate_dataset_and_class(df: pd.DataFrame | pd.Series | np.ndarray, class_name=None):
    # If user passed a dataframe or series and no class_name, set it to "class"
    if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series) and class_name is None:
        class_name = "class"

    # Type case analysis
    if isinstance(df, pd.DataFrame):
        return df.drop(class_name, axis=1), df[class_name]
    if isinstance(df, pd.Series):
        return df.drop(class_name), df[class_name]
    if isinstance(df, np.ndarray):
        # If class_name was set by the user, tell him it is actually useless
        if class_name is not None:
            warnings.warn("The values of class_name is set but not used, as the dataset type is " + str(
                type(df)) + ". The last column is always considered the target attribute.")
        if len(df.shape) == 2:
            return df.transpose()[:-1].transpose(), df.transpose()[-1]
        if len(df.shape) == 1:
            return df[:-1], df[-1]


def euclidean_distance(x_cfx: np.ndarray, x_og: np.ndarray):
    # Make sure attributes go in the same order
    # x_og = x_og[x_cfx.index]

    # Return Euclidean distance
    return np.sqrt(np.sum((x_cfx - x_og) ** 2))


def delta_distance(x_cfx, x_og, eps=0.1):
    abs_distance = abs(x_cfx.values - x_og.values)
    return sum(map(lambda i: i > eps, abs_distance[0]))


def likelihood(x_cfx: pd.DataFrame, bn) -> np.ndarray:
    class_cpd = bn.cpd("class")
    class_values = class_cpd.variable_values()
    cfx = x_cfx.copy()
    n_samples = x_cfx.shape[0]
    likelihood_val = 0
    for v in class_values:
        cfx["class"] = pd.Categorical([v] * n_samples, categories=class_values)
        likelihood_val = likelihood_val + math.e ** bn.logl(cfx)
    return likelihood_val


def log_likelihood(x_cfx: pd.DataFrame, bn) -> np.ndarray:
    l = likelihood(x_cfx, bn)

    '''if l == 0:
        return -np.inf'''
    if not ((l < 1).all()):
        Warning(
            "Likelihood of some points in the space is higher than 1. Computing the log likelihood may not make sense.")
    return np.log(l)


def accuracy(x_cfx: pd.DataFrame, y_og: str | list, bn):
    class_cpd = bn.cpd("class")
    class_values = class_cpd.variable_values()
    cfx = x_cfx.copy()
    if isinstance(y_og, str):
        cfx["class"] = pd.Categorical([y_og], categories=class_values)
    else:
        cfx["class"] = pd.Categorical(y_og, categories=class_values)
    prob = math.e ** bn.logl(cfx)
    ll = likelihood(x_cfx, bn)
    to_ret = np.empty(shape=len(x_cfx.index))
    to_ret[ll < 0] = np.nan
    to_ret[ll > 0] = prob[ll > 0] / ll[ll > 0]
    # if not (ll > 0).any():
    #    warnings("The instance with features "+str(x_cfx.iloc[0])+" and class " + str(y_og))
    return to_ret


def predict_class(data: pd.DataFrame, bayesian_network):
    class_values = bayesian_network.cpd("class").variable_values()
    to_ret = pd.DataFrame(columns=class_values)
    for i in class_values:
        to_ret[i] = accuracy(data, [i] * len(data.index), bayesian_network)
    return to_ret


def straight_path(x_1: np.ndarray, x_2: np.ndarray, chunks=2):
    assert chunks > 1
    return np.linspace(x_1, x_2, chunks)


def path(vertex_array: np.ndarray, chunks=2):
    straight_path_list = list()
    for i in range(vertex_array.shape[0] - 1):
        x_1 = vertex_array[i]
        x_2 = vertex_array[i + 1]
        straight_path_list.append(straight_path(x_1, x_2, chunks))
    return straight_path_list


def path_likelihood_length(path: pd.DataFrame, bayesian_network, penalty=1):
    separation = euclidean_distance(path.iloc[0], path.iloc[1])
    medium_points = ((path + path.shift()) / 2).drop(0).reset_index()
    likelihood_path = (-log_likelihood(medium_points, bayesian_network)) ** penalty * separation
    return np.sum(likelihood_path)


def get_naive_structure(data: pd.DataFrame, type):
    naive = type(data.columns)
    for i in [i for i in data.columns if i != "class"]:
        naive.add_arc("class", i)
    return naive


def copy_structure(bn: pb.BayesianNetwork):
    copy = type(bn)(bn.nodes())
    for i in bn.arcs():
        copy.add_arc(i[0], i[1])
    return copy


def check_copy(bn):
    return bn.fitted()


def hill_climbing(data: pd.DataFrame, bn_type: str, score=None, seed=0):
    bn = None
    if bn_type == "CLG":
        if score is None:
            score = "bic"
        bn = pb.hc(data, start=get_naive_structure(data, pb.CLGNetwork), operators=["arcs"], score=score,
                   seed=seed)
        bn = copy_structure(bn)
    elif bn_type == "SP":
        if score is None:
            score = "validated-lik"
        bn = pb.hc(data, start=get_naive_structure(data, pb.SemiparametricBN), operators=["arcs", "node_type"],
                   score=score,
                   seed=seed)
        bn = copy_structure(bn)
    elif bn_type == "Gaussian":
        if score is None:
            score = "bic"
        bn = pb.hc(data, start=get_naive_structure(data, pb.GaussianNetwork), operators=["arcs"], score=score,
                   seed=seed)
        bn = copy_structure(bn)
    else:
        raise PybnesianParallelizationError(
            "Only valid types are CLG, SP and Gaussian. For more customization use the hc method of pybnesian")
    bn.fit(data)
    bn.include_cpd = True
    pool = mp.Pool(1)
    res = pool.starmap(check_copy, [(bn,)])
    pool.close()
    if not res[0]:
        raise PybnesianParallelizationError(
            "As of version 0.4.3, PyBnesian Bayesian networks have internal and stochastic problems with the method "
            "\"copy()\"."
            "As such, the network is not parallelized correctly and experiments cannot be launched.")
    return bn


def get_and_process_data(dataset_id: int):
    # Load the dataset
    data = oml.datasets.get_dataset(dataset_id, download_data=True, download_qualities=False,
                                    download_features_meta_data=False).get_data()[0]

    # Shuffle the dataset
    data = data.sample(frac=1, random_state=0)

    # Transform the class into a categorical variable
    data["class"] = data[data.columns[-1]].astype('string').astype('category')
    data = data.drop(data.columns[-2], axis=1)

    # Scale the rest of the dataset
    feature_columns = [i for i in data.columns if i != "class"]
    data[feature_columns] = StandardScaler().fit_transform(data[feature_columns].values)

    for i in data.columns[:-1]:
        data = data[data[i] < 3]
        data = data[data[i] > -3]
    return data
