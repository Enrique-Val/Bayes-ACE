import pandas as pd
import numpy as np
from pybnesian import hc, CLGNetworkType
# from drawdata import draw_scatter
import matplotlib.pyplot as plt
import math

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize

from bayesace.utils import *

from sklearn.preprocessing import OneHotEncoder


def euclidean_distance(x_cfx, x_og):
    # Make sure attributes go in the same order
    # x_og = x_og[x_cfx.index]

    # Return Euclidean distance
    return np.sqrt(np.sum((x_cfx.values - x_og.values) ** 2))


def delta_distance(x_cfx, x_og, eps=0.1):
    abs_distance = abs(x_cfx.values - x_og.values)
    return sum(map(lambda i: i > eps, abs_distance[0]))


def likelihood(x_cfx, bn):
    class_cpd = bn.cpd("class")
    class_values = class_cpd.variable_values()
    cfx = x_cfx.copy()
    n_samples = x_cfx.shape[0]
    likelihood = 0
    for v in class_values:
        cfx["class"] = pd.Categorical([v] * n_samples, categories=class_values)
        likelihood = likelihood + math.e ** bn.logl(cfx)
    return likelihood


def log_likelihood(x_cfx, bn):
    return np.log(likelihood(x_cfx, bn))


def accuracy(x_cfx, y_og: str | list, bn):
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
        print("Peligro")
        return 1


def straight_path(x_1, x_2):
    spacing = int(euclidean_distance(x_1, x_2) / 10)
    if spacing < 2 :
        spacing = 2
    points = np.zeros(shape=(x_2.shape[1], spacing))
    for i, att in enumerate(x_2.columns):
        points[i] = np.linspace(x_1[att].values[0], x_2[att].values[0], spacing)
    to_ret = pd.DataFrame(columns=x_2.columns, index=range(0, spacing))
    to_ret[:] = points.transpose()
    if to_ret.empty:
        print(x_1)
        print(x_2)
        raise Exception("Straight path")
    return to_ret


def path(df_vertex):
    to_ret = pd.DataFrame(columns=df_vertex.columns)
    for i in range(len(df_vertex.index) - 1):
        x_1 = df_vertex.iloc[[i]]
        x_2 = df_vertex.iloc[[i + 1]]
        to_ret = pd.concat([to_ret, straight_path(x_1, x_2)])
    return to_ret.reset_index()


def avg_path_logl(x_cfx, x_og, bn, penalty):
    likelihood_path = (-log_likelihood(straight_path(x_og, x_cfx), bn) + 1) ** penalty
    return np.sum(likelihood_path)


def bayes_ace(bayesian_network, instance, n_vertex=1, penalty=1):
    class BestPathFinder(ElementwiseProblem):
        def __init__(self, bayesian_network, instance, n_vertex=1, penalty=1):
            n_features = (len(instance.columns) - 1)
            super().__init__(n_var=n_features * (n_vertex + 1),
                             n_obj=1,
                             n_ieq_constr=2,
                             xl=np.array([-2] * (n_features * (n_vertex + 1))),
                             xu=np.array([2] * (n_features * (n_vertex + 1))))
            self.x_og = instance.drop("class", axis=1)
            self.y_og = "a" #instance["class"]
            self.n_vertex = n_vertex
            self.penalty = penalty
            self.n_features = n_features
            self.bayesian_network = bayesian_network

        def _evaluate(self, x, out, *args, **kwargs):
            df_vertex = pd.DataFrame(columns=self.x_og.columns,
                                     data=np.resize(x, new_shape=(self.n_vertex+1, self.n_features)))
            df_vertex = pd.concat([self.x_og, df_vertex])
            df_vertex = df_vertex.reset_index()
            path_x = path(df_vertex)
            likelihood_path = (-log_likelihood(path_x, self.bayesian_network) + 1) ** self.penalty
            f1 = np.sum(likelihood_path)
            out["F"] = np.column_stack([f1])

            x_cfx = self.x_og.copy()
            x_cfx[:] = x[-self.n_features:]
            #print(accuracy(self.x_cfx, self.y_og, self.bayesian_network))
            g1 = accuracy(x_cfx, self.y_og, self.bayesian_network)-0.05  # -likelihood(x_cfx, learned)+0.0000001
            g2 = -0.1 #likelihood(x_cfx, self.bayesian_network)+0.005
            out["G"] = np.column_stack([g1,g2])

    problem = BestPathFinder(bayesian_network=bayesian_network, instance=instance, n_vertex=n_vertex, penalty=penalty)
    algorithm = NSGA2(pop_size=200)

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 10),
                   seed=1,
                   verbose=True)

    print(res.X)
    print(res.F)

    return res.X
