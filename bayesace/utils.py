import pandas as pd
import numpy as np
import warnings
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import  OneHotEncoder
from collections import Counter

from bayesace.models.conditional_density_estimator import ConditionalDE

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


def likelihood(data: pd.DataFrame, density_estimator, class_var_name="class", mutable=False) -> np.ndarray:
    '''
    Computes the likelihood of the data given the density estimator, marginalizing over all possible values of the
    class variable. Even if provided, the method will ignore it.

    :param data: The data to compute the likelihood
    '''
    if not mutable:
        data = data.copy()
    if isinstance(density_estimator, ConditionalDE):
        return density_estimator.likelihood(data, class_var_name=class_var_name)
    if class_var_name in data.columns:
        data = data.drop(class_var_name, axis=1)
    class_cpd = density_estimator.cpd(class_var_name)
    class_values = class_cpd.variable_values()
    n_samples = data.shape[0]
    likelihood_val = 0.0
    for v in class_values:
        data[class_var_name] = pd.Categorical([v] * n_samples, categories=class_values)
        likelihood_val = likelihood_val + np.e ** density_estimator.logl(data)
    if (likelihood_val > 1).any():
        Warning(
            "Likelihood of some points in the space is higher than 1.")
    return likelihood_val

def log_likelihood_array(data: np.ndarray, features:list, density_estimator, class_var_name="class", mutable=False) -> np.ndarray:
    return log_likelihood(pd.DataFrame(data, columns=features), density_estimator, class_var_name, mutable)

def log_likelihood(data: pd.DataFrame, density_estimator, class_var_name="class", mutable=False) -> np.ndarray:
    ll = likelihood(data, density_estimator, class_var_name, mutable)
    logl = np.empty(shape=len(ll))
    logl[ll > 0] = np.log(ll[ll > 0])
    logl[ll <= 0] = -np.inf
    return logl
'''if not mutable:
    data = data.copy()
if isinstance(density_estimator, ConditionalNF):
    return density_estimator.log_likelihood(data, class_var_name=class_var_name)
if class_var_name in data.columns:
    data = data.drop(class_var_name, axis=1)
class_cpd = density_estimator.cpd(class_var_name)
class_values = class_cpd.variable_values()
n_samples = data.shape[0]
# Set the value of the logl for the first iteration
data[class_var_name] = pd.Categorical([class_values[0]] * n_samples, categories=class_values)
log_likelihood_val = density_estimator.logl(data)
for v in class_values[1:]:
    data[class_var_name] = pd.Categorical([v] * n_samples, categories=class_values)
    logly = density_estimator.logl(data)
    log_likelihood_val = log_likelihood_val + np.log(1 + np.e ** (logly - log_likelihood_val))

if (log_likelihood_val > 0).any():
    Warning(
        "Log likelihood of some points in the space is higher than 0.")
return log_likelihood_val'''

def brier_score(y_true: np.ndarray, y_pred: pd.DataFrame) -> float:
    encoder = OneHotEncoder(sparse_output=False)
    y_true_coded = encoder.fit_transform(y_true.reshape(-1, 1))
    class_labels = encoder.categories_[0]
    if len(y_pred.columns) == 2:
        y_bin = y_true_coded[:, 0]
        y_pred = y_pred[class_labels[0]]
        return np.sum((y_bin - y_pred.values) ** 2) / len(y_true)
    else:
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


def total_l0_path(path: np.ndarray, eps=0.1) -> int:
    # Preallocate L0 array for columns
    l0_array = np.zeros(path.shape[1])
    N = path.shape[0]

    for i in range(N):
        # Compute differences for all rows compared to row `i`
        diff = path[i] - path[i:N]  # Shape: (N-i, M)

        # Apply threshold and update l0_array
        l0_array = np.max(((np.abs(diff) > eps).any(axis=0).astype(int), l0_array), axis=0)

    return np.sum(l0_array)


def mae_samples(y_true: np.ndarray, y_pred: pd.DataFrame) -> float:
    encoder = OneHotEncoder(sparse_output=False)
    y_true_coded = encoder.fit_transform(y_true.reshape(-1, 1))
    class_labels = encoder.categories_[0]
    if len(y_pred.columns) == 2:
        y_bin = y_true_coded[:, 0]
        y_pred = y_pred[class_labels[0]]
        return np.abs(y_bin - y_pred.values)
    else:
        y_pred = y_pred[class_labels]
        return np.abs(y_true_coded - y_pred.values)


def path(vertex_array: np.ndarray, chunks=2) -> np.ndarray:
    assert chunks > 1
    assert vertex_array.shape[0] > 1
    if vertex_array.shape[0] == 2:
        return np.linspace(vertex_array[0], vertex_array[1], chunks)
    to_ret_path = np.linspace(vertex_array[0], vertex_array[1], chunks)
    for i in range(vertex_array.shape[0] - 2):
        x_1 = vertex_array[i + 1]
        x_2 = vertex_array[i + 2]
        to_ret_path = np.vstack((to_ret_path, np.linspace(x_1, x_2, chunks)[1:]))
    return to_ret_path


'''def path_likelihood_length2(vertex_array: pd.DataFrame, density_estimator, penalty=1):
    columnas = list(vertex_array.columns)
    def neg_log_likelihood(x):
        #print("x",x)
        return -log_likelihood_array(x, columnas, density_estimator)

    def pt(t, x1, x2):
        return x1 + t * (x2 - x1)

    def line_integral(x1, x2, f):
        def integrand(t):
            return f(pt(t, x1, x2)) * np.linalg.norm(x2 - x1)

        result,_ = quad(integrand, 0, 1)
        print("result",result)
        return result

    try:
        sum = 0
        for i in range(vertex_array.shape[0] - 1):
            x_1 = vertex_array.iloc[i].values
            x_2 = vertex_array.iloc[i + 1].values
            sum += line_integral(x_1, x_2, neg_log_likelihood)

        return sum
    except NanLogProb:
        return np.inf
'''

def path_likelihood_length(path: pd.DataFrame, density_estimator, penalty=1):
    # Separation is computed between each row without for loops, fully vectorised
    separation = np.linalg.norm(path.diff(axis=0).drop(0), axis=1)

    medium_points = ((path + path.shift()) / 2).drop(0).reset_index(drop=True)
    logl_points = -log_likelihood(medium_points, density_estimator)
    # Array with 1 if positive, 0 if 0 and -1 if negative
    sign_array = np.sign(logl_points)
    point_evaluations = logl_points ** penalty
    # If the logl is negative, and the penalty is even, then we need to multiply for the sign array
    if penalty % 2 == 0:
        point_evaluations = np.multiply(point_evaluations, sign_array)
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
    if isinstance(density_estimator, ConditionalDE):
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
    predictions = density_estimator.predict_proba(grid, output="pandas")
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

def plot_path(df, res_b=None):
    # I need you to generalize this code for any class values names
    # x_1 = x_og.drop("class", axis=1)
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


def get_other_class(class_values, class_value):
    return class_values[class_values != class_value][0]


def median_absolute_deviation(x, axis=None):
    return np.median(np.abs(x - np.median(x, axis=axis)), axis=axis)