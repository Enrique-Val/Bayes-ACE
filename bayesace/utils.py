import pandas as pd
import numpy as np
import pybnesian as pb
import warnings
import math
import multiprocessing as mp
import openml as oml
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from collections import Counter

from bayesace.models.conditional_normalizing_flow import ConditionalNF
from bayesace.models.utils import hill_climbing
import time


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


def likelihood(data: pd.DataFrame, density_estimator, class_var_name="class", mutable = False) -> np.ndarray:
    '''
    Computes the likelihood of the data given the density estimator, marginalizing over all possible values of the
    class variable. Even if provided, the method will ignore it.

    :param data: The data to compute the likelihood
    '''
    if not mutable :
        data = data.copy()
    if isinstance(density_estimator, ConditionalNF) :
        return density_estimator.likelihood(data, class_var_name=class_var_name)
    if class_var_name in data.columns:
        data = data.drop(class_var_name, axis=1)
    class_cpd = density_estimator.cpd(class_var_name)
    class_values = class_cpd.variable_values()
    n_samples = data.shape[0]
    likelihood_val = 0.0
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

# Get the probability P(y|x)
def posterior_probability(x_cfx: pd.DataFrame, y_og: str | list, density_estimator, class_var_name="class"):
    # Obtain the labels accesing either the MultiBNAF model or the cpd of the bn
    class_labels = None
    if isinstance(density_estimator, ConditionalNF) :
        class_labels = density_estimator.get_class_labels()
    else:
        class_cpd = density_estimator.cpd(class_var_name)
        class_labels = class_cpd.variable_values()
    x_cfx = x_cfx.copy()
    if isinstance(y_og, str):
        x_cfx[class_var_name] = pd.Categorical([y_og] * len(x_cfx.index), categories=class_labels)
    else:
        assert len(y_og) == len(x_cfx.index)
        x_cfx[class_var_name] = pd.Categorical(y_og, categories=class_labels)
    prob = np.e ** density_estimator.logl(x_cfx)
    ll = likelihood(x_cfx, density_estimator, mutable = True)
    to_ret = np.empty(shape=len(x_cfx.index))
    # If the likelihood is 0, then classification is done with uniform probability
    to_ret[ll <= 0] = 1/len(class_labels)
    to_ret[ll > 0] = prob[ll > 0] / ll[ll > 0]
    # if not (ll > 0).any():
    #    warnings("The instance with features "+str(x_cfx.iloc[0])+" and class " + str(y_og))
    return to_ret


def predict_class(data: pd.DataFrame, density_estimator, class_var_name="class"):
    if class_var_name in data.columns:
        Warning("The class variable is already in the dataset. It will be removed for the prediction.")
        data = data.drop(class_var_name, axis=1)
    if isinstance(density_estimator, ConditionalNF):
        return pd.DataFrame(density_estimator.predict_proba(data.values), columns=density_estimator.get_class_labels())
    else:
        class_values = density_estimator.cpd(class_var_name).variable_values()
        to_ret = pd.DataFrame(columns=class_values)
        for i in class_values:
            to_ret[i] = posterior_probability(data, [i] * len(data.index), density_estimator)
        # Check that the sum of the probabilities is 1
        sum_pred = to_ret.sum(axis=1)
        assert((np.less.outer(sum_pred,1+0.001) & np.greater_equal.outer(sum_pred,1-0.001)).all())
        return to_ret


def brier_score(y_true: np.ndarray, y_pred: pd.DataFrame) -> float:
    encoder = OneHotEncoder(sparse_output=False)
    y_true_coded = encoder.fit_transform(y_true.reshape(-1, 1))
    class_labels = encoder.categories_[0]
    if len(y_pred.columns) == 2:
        y_bin = y_true_coded[:, 0]
        y_pred = y_pred[class_labels[0]]
        return np.sum((y_bin - y_pred.values) ** 2) / len(y_true)
    else :
        y_pred = y_pred[class_labels]
        return np.sum((y_true_coded - y_pred.values) ** 2) / len(y_true)


def auc(y_true: np.ndarray, y_pred: pd.DataFrame) -> float:
    encoder = OneHotEncoder(sparse_output=False)
    y_true_coded = encoder.fit_transform(y_true.reshape(-1, 1))
    class_labels = encoder.categories_[0]
    if len(y_pred.columns) == 2:
        y_bin = y_true_coded[:, 0]
        y_pred = y_pred[class_labels[0]]
        return roc_auc_score(y_bin, y_pred.values)
    else:
        y_pred = y_pred[class_labels]
        return roc_auc_score(y_true_coded, y_pred.values)

def path(vertex_array: np.ndarray, chunks=2) -> np.ndarray:
    assert chunks > 1
    assert vertex_array.shape[0] > 1
    if vertex_array.shape[0] == 2 :
        return np.linspace(vertex_array[0], vertex_array[1], chunks)
    to_ret_path = np.linspace(vertex_array[0], vertex_array[1], chunks)
    for i in range(vertex_array.shape[0] - 2):
        x_1 = vertex_array[i + 1]
        x_2 = vertex_array[i + 2]
        to_ret_path = np.vstack((to_ret_path, np.linspace(x_1, x_2, chunks)[1:]))
    return to_ret_path


def path_likelihood_length(path: pd.DataFrame, bayesian_network, penalty=1):
    # Separation is computed between each row without for loops, fully vectorised
    separation = np.linalg.norm(path.diff(axis=0).drop(0), axis=1)

    medium_points = ((path + path.shift()) / 2).drop(0).reset_index(drop=True)
    point_evaluations = (-log_likelihood(medium_points, bayesian_network)) ** penalty
    assert (point_evaluations > 0).any()
    likelihood_path = np.multiply(point_evaluations, separation)
    return np.sum(likelihood_path)

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
    if isinstance(density_estimator, ConditionalNF) :
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


def get_decision_boundary_plot(density_estimator, class_var_name="class", limit=3, step=0.01):
    grid = np.array([
        [a, b]
        for a in np.arange(-limit, limit, step)
        for b in np.arange(-limit, limit, step)
    ])
    resolution = len(np.arange(-limit, limit, step))
    class_labels = None
    predictions = predict_class(pd.DataFrame(grid, columns=["x", "y"]), density_estimator)
    prob_list = []
    for label in predictions.columns:
        post = predictions[label].values
        post -= np.min(post)
        post /= np.ptp(post)
        post = np.flip(np.resize(post, (resolution, resolution)).transpose(), axis=0)
        prob_list.append(post)
    while len(prob_list) < 3:
        prob_list.append(np.zeros((resolution, resolution)))
    return np.dstack(prob_list)

def get_mean_sd_logl(data, model_type, folds = 10) :
    def kfold_indices(data, k):
        fold_size = len(data) // k
        indices = np.arange(len(data))
        folds = []
        for i in range(k):
            test_indices = indices[i * fold_size: (i + 1) * fold_size]
            train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
            folds.append((train_indices, test_indices))
        return folds

    # Define the number of folds (K) and parameters of our grid search for the normalizing flow
    k = folds

    fold_indices = kfold_indices(data, k)

    mean_logl = {}
    std_logl = {}
    for label in data["class"].cat.categories:
        mean_logl[label] = []
        std_logl[label] = []

    for train_index, test_index in fold_indices:
        # Train a model for this fold
        df_train = data.iloc[train_index].reset_index(drop=True)
        df_test = data.iloc[test_index].reset_index(drop=True)
        network = None
        if model_type == 'CLG' or model_type == 'SP':
            network = hill_climbing(data=df_train, bn_type=model_type)
        elif model_type == "NN":
            # TODO
            pass
            #args = Arguments()
            #network = MultiBnaf(args, df_train)
        # Iterate for all classes
        for label in df_test["class"].cat.categories:
            slogl_i = network.logl(df_test[df_test["class"] == label])
            # Store the mean and std logl for this fold and for this class
            mean_logl[label].append(slogl_i.mean())
            std_logl[label].append(slogl_i.std())

    for label in data["class"].cat.categories:
        mean_logl[label] = np.mean(mean_logl[label])
        std_logl[label] = np.mean(std_logl[label])
    return mean_logl, std_logl

def plot_path(df, res_b = None) :
    # I need you to generalize this code for any class values names
    #x_1 = x_og.drop("class", axis=1)
    assert len(df.columns) == 3
    class_values = df["class"].cat.categories
    to_plot = df.drop("class", axis=1)
    colours = df["class"].to_numpy()
    colour_palette = ["green", "blue", "yellow"]
    for i in range(len(class_values)):
        colours[colours == class_values[i]] = colour_palette[i]
    plt.scatter(to_plot[to_plot.columns[0]], to_plot[to_plot.columns[1]], color=colours)
    if res_b is not None:
        df_vertex = res_b.path
        plt.plot(df_vertex[to_plot.columns[0]], df_vertex[to_plot.columns[1]], color="red")

