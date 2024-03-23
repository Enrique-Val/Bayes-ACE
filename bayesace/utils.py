import pandas as pd
import numpy as np
import pybnesian as pb
import warnings
import math
import multiprocessing as mp
import openml as oml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from collections import Counter

from bayesace.models.multi_bnaf import MultiBnaf


def identity(x):
    return x


def neg_log(x):
    return -np.log(x)


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


def likelihood(data: pd.DataFrame, density_estimator, class_var_name="class") -> np.ndarray:
    '''
    Computes the likelihood of the data given the density estimator, marginalizing over all possible values of the
    class variable. Even if provided, the method will ignore it.

    :param data: The data to compute the likelihood
    '''
    if class_var_name in data.columns:
        data = data.drop(class_var_name, axis=1)
    if isinstance(density_estimator, MultiBnaf):
        return density_estimator.likelihood(data, class_var_name=class_var_name)
    class_cpd = density_estimator.cpd(class_var_name)
    class_values = class_cpd.variable_values()
    n_samples = data.shape[0]
    likelihood_val = 0
    for v in class_values:
        data[class_var_name] = pd.Categorical([v] * n_samples, categories=class_values)
        likelihood_val = likelihood_val + math.e ** density_estimator.logl(data)
    return likelihood_val


def log_likelihood(x_cfx: pd.DataFrame, bn) -> np.ndarray:
    l = likelihood(x_cfx, bn)

    if not ((l < 1).all()):
        Warning(
            "Likelihood of some points in the space is higher than 1. Computing the log likelihood may not make sense.")
    return np.log(l)


def posterior_probability(x_cfx: pd.DataFrame, y_og: str | list, density_estimator, class_var_name="class"):
    # Obtain the labels accesing either the MultiBNAF model or the cpd of the bn
    class_labels = None
    if isinstance(density_estimator, MultiBnaf):
        class_labels = density_estimator.get_class_labels()
    else:
        class_cpd = density_estimator.cpd(class_var_name)
        class_labels = class_cpd.variable_values()
    cfx = x_cfx.copy()
    if isinstance(y_og, str):
        cfx[class_var_name] = pd.Categorical([y_og] * len(x_cfx.index), categories=class_labels)
    else:
        assert len(y_og) == len(x_cfx.index)
        cfx[class_var_name] = pd.Categorical(y_og, categories=class_labels)
    prob = math.e ** density_estimator.logl(cfx)
    ll = likelihood(x_cfx, density_estimator)
    to_ret = np.empty(shape=len(x_cfx.index))
    to_ret[ll < 0] = np.nan
    to_ret[ll > 0] = prob[ll > 0] / ll[ll > 0]
    # if not (ll > 0).any():
    #    warnings("The instance with features "+str(x_cfx.iloc[0])+" and class " + str(y_og))
    return to_ret


def predict_class(data: pd.DataFrame, density_estimator, class_var_name="class"):
    if class_var_name in data.columns:
        Warning("The class variable is already in the dataset. It will be removed for the prediction.")
        data = data.drop(class_var_name, axis=1)
    if isinstance(density_estimator, MultiBnaf):
        return pd.DataFrame(density_estimator.predict(data.values), columns=density_estimator.get_class_labels())
    else:
        class_values = density_estimator.cpd(class_var_name).variable_values()
        to_ret = pd.DataFrame(columns=class_values)
        for i in class_values:
            to_ret[i] = posterior_probability(data, [i] * len(data.index), density_estimator)
        return to_ret


def brier_score(y_true: np.ndarray, y_pred: pd.DataFrame) -> float:
    encoder = OneHotEncoder(sparse_output=False)
    y_true_coded = encoder.fit_transform(y_true.reshape(-1, 1))
    class_labels = encoder.categories_[0]
    y_pred = y_pred[class_labels]
    return np.sum((y_true_coded - y_pred.values) ** 2)/len(y_true)


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
    medium_points = ((path + path.shift()) / 2).drop(0).reset_index(drop = True)
    likelihood_path = (-log_likelihood(medium_points, bayesian_network)) ** penalty * separation
    return np.sum(likelihood_path)


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


def L0_norm(x_1, x_2, eps=0.01):
    return Counter(np.abs(x_1 - x_2) > eps)[True]


def get_probability_plot(density_estimator, class_var_name="class", limit=3, step=0.01):
    grid = np.array([
        [a, b]
        for a in np.arange(-limit, limit, step)
        for b in np.arange(-limit, limit, step)
    ])
    resolution = len(np.arange(-limit, limit, step))
    class_labels = None
    if isinstance(density_estimator, MultiBnaf):
        class_labels = density_estimator.get_class_labels()
    else:
        class_labels = density_estimator.cpd(class_var_name).variable_values()
    if len(class_labels) > 3:
        raise ValueError("The number of classes is too high to plot the probability plot")
    prob_list = []
    for label in class_labels:
        grid_df = pd.DataFrame(grid, columns=["x", "y"])
        grid_df[class_var_name] = pd.Categorical([label] * len(grid), categories=class_labels)
        post = np.e ** density_estimator.logl(grid_df)
        post -= np.min(post)
        post /= np.ptp(post)
        post = np.flip(np.resize(post, (resolution, resolution)).transpose(), axis=0)
        prob_list.append(post)
    while len(prob_list) < 3:
        prob_list.append(np.zeros((resolution, resolution)))
    return np.dstack(prob_list)
