import numpy as np
import pandas as pd

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from bayesace.utils import *
from bayesace.algorithms.algorithm import ACE


class BestPathFinder(ElementwiseProblem):
    def __init__(self, bayesian_network, instance, n_vertex=1, penalty=1, chunks=2, likelihood_threshold = 0.05, accuracy_threshold = 0.05):
        n_features = (len(instance.columns) - 1)
        super().__init__(n_var=n_features * (n_vertex + 1),
                         n_obj=1,
                         n_ieq_constr=2,
                         xl=np.array([-2] * (n_features * (n_vertex + 1))),
                         xu=np.array([2] * (n_features * (n_vertex + 1))))
        self.x_og = instance.drop("class", axis=1).values
        self.y_og = "a"  # instance["class"]
        self.n_vertex = n_vertex
        self.penalty = penalty
        self.features = instance.drop("class", axis=1).columns
        self.n_features = n_features
        self.bayesian_network = bayesian_network
        self.chunks = chunks
        self.likelihood_threshold = likelihood_threshold
        self.accuracy_threshold = accuracy_threshold

    def _evaluate(self, x, out, *args, **kwargs):
        vertex_array = np.resize(np.append(self.x_og, x), new_shape=(self.n_vertex + 2, self.n_features))
        paths = path(vertex_array, chunks=self.chunks)
        sum_path = 0
        for path_i in paths:
            sum_path += path_likelihood_length(pd.DataFrame(path_i, columns=self.features),
                                               bayesian_network=self.bayesian_network, penalty=self.penalty)
        f1 = sum_path
        out["F"] = np.column_stack([f1])

        x_cfx = self.x_og.copy()
        x_cfx[:] = x[-self.n_features:]
        # print(accuracy(self.x_cfx, self.y_og, self.bayesian_network))
        g1 = accuracy(pd.DataFrame(x_cfx, columns=self.features), self.y_og,
                      self.bayesian_network) - self.accuracy_threshold  # -likelihood(x_cfx, learned)+0.0000001
        g2 = likelihood(x_cfx, self.bayesian_network)+self.likelihood_threshold
        out["G"] = np.column_stack([g1, g2])


class BayesACE(ACE):
    def __init__(self, n_vertex, pop_size=100, generations=10, likelihood_threshold = 0.05, accuracy_threshold = 0.05):
        super().__init__()
        self.n_vertex = n_vertex
        self.generations = generations
        self.population_size = pop_size
        self.likelihood_threshold = likelihood_threshold
        self.accuracy_threshold = accuracy_threshold

    def run(self, instance: pd.DataFrame):
        problem = BestPathFinder(bayesian_network=self.bayesian_network, instance=instance, n_vertex=self.n_vertex,
                                 penalty=self.penalty, chunks=self.chunks, likelihood_threshold=self.likelihood_threshold, accuracy_threshold=self.accuracy_threshold)
        algorithm = NSGA2(pop_size=self.population_size)
        res = minimize(problem,
                       algorithm,
                       ('n_gen', self.generations),
                       seed=self.seed,
                       verbose=self.verbose)
        total_path = np.append([separate_dataset_and_class(instance)[0].values[0], res.X])
        path_to_ret = pd.DataFrame(data=np.resize(total_path, new_shape=(self.n_vertex + 2, self.n_features)),
                                   columns=self.features)
        counterfactual = path_to_ret.iloc[-1]
        return counterfactual, path_to_ret, res.F
