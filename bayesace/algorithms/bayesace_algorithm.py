import time

import numpy as np
import pandas as pd
import multiprocessing as mp

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import StarmapParallelization
from pymoo.termination.default import DefaultSingleObjectiveTermination

from bayesace.utils import *
from bayesace.algorithms.algorithm import ACE, ACEResult


class PybnesianParallelizationError(Exception):
    pass


class BestPathFinder(ElementwiseProblem):
    def __init__(self, bayesian_network, instance, n_vertex=1, penalty=1, chunks=2, likelihood_threshold=0.05,
                 accuracy_threshold=0.05, sampling_range=(-4, 4), **kwargs):
        n_features = (len(instance.columns) - 1)
        super().__init__(n_var=n_features * (n_vertex + 1),
                         n_obj=1,
                         n_ieq_constr=2,
                         xl=np.array([sampling_range[0]] * (n_features * (n_vertex + 1))),
                         xu=np.array([sampling_range[1]] * (n_features * (n_vertex + 1))), **kwargs)
        self.x_og = instance.drop("class", axis=1).values
        self.y_og = instance["class"].values[0]
        self.n_vertex = n_vertex
        self.penalty = penalty
        self.features = instance.drop("class", axis=1).columns
        self.n_features = n_features
        self.bayesian_network = bayesian_network
        assert self.bayesian_network.fitted()
        self.chunks = chunks
        self.likelihood_threshold = likelihood_threshold
        self.accuracy_threshold = accuracy_threshold

    def _evaluate(self, x, out, *args, **kwargs):
        if not self.bayesian_network.fitted():
            raise PybnesianParallelizationError(
                "As of version 0.4.3, PyBnesian Bayesian networks have internal and stochastic problems with the method \"copy()\"."
                "As such, some parallelization efforts of the code may fail. We recommend either "
                "a multiple restart approach after it randomly functions or"
                "completely remove parallelization.")
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
        g2 = -likelihood(pd.DataFrame(x_cfx, columns=self.features), self.bayesian_network) + self.likelihood_threshold
        out["G"] = np.column_stack([g1, g2])


class BayesACE(ACE):
    def __init__(self, bayesian_network, features, chunks, n_vertex, pop_size=100,
                 generations=10, likelihood_threshold=0.00, accuracy_threshold=0.50, penalty=1, sampling_range=(-4, 4),
                 seed=0,
                 verbose=True):
        super().__init__(bayesian_network, features, chunks, likelihood_threshold=likelihood_threshold,
                         accuracy_threshold=accuracy_threshold, penalty=penalty, seed=seed, verbose=verbose)
        self.bayesian_network = bayesian_network
        self.n_vertex = n_vertex
        self.generations = generations
        self.population_size = pop_size
        self.sampling_range = sampling_range

    def run(self, instance: pd.DataFrame, parallelize=False, return_info=False):
        termination = DefaultSingleObjectiveTermination(
            ftol=0.025,
            period=10
        )
        # initialize the thread pool and create the runner
        if parallelize:
            n_processes = mp.cpu_count()
            pool = mp.Pool(n_processes)
            runner = StarmapParallelization(pool.starmap)
            problem = BestPathFinder(bayesian_network=self.bayesian_network, instance=instance, n_vertex=self.n_vertex,
                                     penalty=self.penalty, chunks=self.chunks,
                                     likelihood_threshold=self.likelihood_threshold,
                                     accuracy_threshold=self.accuracy_threshold, sampling_range=self.sampling_range,
                                     elementwise_runner=runner)
            algorithm = NSGA2(pop_size=self.population_size)

            res = minimize(problem,
                           algorithm,
                           termination=('n_gen', self.generations),
                           seed=self.seed,
                           verbose=self.verbose)
            pool.close()
        else :
            problem = BestPathFinder(bayesian_network=self.bayesian_network, instance=instance, n_vertex=self.n_vertex,
                                     penalty=self.penalty, chunks=self.chunks,
                                     likelihood_threshold=self.likelihood_threshold,
                                     accuracy_threshold=self.accuracy_threshold, sampling_range=self.sampling_range)
            algorithm = NSGA2(pop_size=self.population_size)

            res = minimize(problem,
                           algorithm,
                           termination=('n_gen', self.generations),
                           seed=self.seed,
                           verbose=self.verbose)
        if res.X is None:
            if return_info:
                return (ACEResult(None, instance.drop("class", axis=1), np.inf), res)
            return ACEResult(None, instance.drop("class", axis=1), np.inf)

        total_path = np.append(separate_dataset_and_class(instance)[0].values[0], res.X)
        path_to_ret = pd.DataFrame(data=np.resize(total_path, new_shape=(self.n_vertex + 2, self.n_features)),
                                   columns=self.features)
        counterfactual = path_to_ret.iloc[-1]
        distance = path_likelihood_length(path_to_ret, bayesian_network=self.bayesian_network, penalty=1)
        if return_info:
            return (ACEResult(counterfactual, path_to_ret, distance), res)
        return ACEResult(counterfactual, path_to_ret, distance)
