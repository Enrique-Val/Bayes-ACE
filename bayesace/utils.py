import pandas as pd
import numpy as np
import warnings
import math


def identity(x):
    return x


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


def euclidean_distance(x_cfx : np.ndarray, x_og : np.ndarray) :
    # Make sure attributes go in the same order
    # x_og = x_og[x_cfx.index]

    # Return Euclidean distance
    return np.sqrt(np.sum((x_cfx - x_og) ** 2))


def delta_distance(x_cfx, x_og, eps=0.1):
    abs_distance = abs(x_cfx.values - x_og.values)
    return sum(map(lambda i: i > eps, abs_distance[0]))


def likelihood(x_cfx: pd.DataFrame, bn):
    class_cpd = bn.cpd("class")
    class_values = class_cpd.variable_values()
    cfx = x_cfx.copy()
    n_samples = x_cfx.shape[0]
    likelihood_val = 0
    for v in class_values:
        cfx["class"] = pd.Categorical([v] * n_samples, categories=class_values)
        likelihood_val = likelihood_val + math.e ** bn.logl(cfx)
    return likelihood_val


def log_likelihood(x_cfx: pd.DataFrame, bn):
    return np.log(likelihood(x_cfx, bn))


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
    if ll > 0:
        return prob / ll
    else:
        return 1


def straight_path(x_1: np.ndarray, x_2: np.ndarray, chunks=2):
    assert chunks > 1
    return np.linspace(x_1, x_2, chunks)


def path(vertex_array : np.ndarray, chunks=2):
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
