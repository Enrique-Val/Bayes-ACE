import time

import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy.stats import norm, truncnorm

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import StarmapParallelization
from pymoo.termination.default import DefaultSingleObjectiveTermination

from bayesace.models.utils import PybnesianParallelizationError
from bayesace.utils import *
from bayesace.algorithms.algorithm import ACE, ACEResult


class BestPathFinder(ElementwiseProblem):
    def __init__(self, bayesian_network, instance, n_vertex=1, penalty=1, chunks=2, likelihood_threshold=0.05,
                 accuracy_threshold=0.05, sampling_range=(-3, 3), **kwargs):
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
        assert len(np.append(self.x_og, x)) == (self.n_vertex + 2) * self.n_features
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
        g1 = posterior_probability(pd.DataFrame(x_cfx, columns=self.features), self.y_og,
                                   self.bayesian_network) - self.accuracy_threshold  # -likelihood(x_cfx, learned)+0.0000001
        g2 = -likelihood(pd.DataFrame(x_cfx, columns=self.features), self.bayesian_network) + self.likelihood_threshold
        out["G"] = np.column_stack([g1, g2])


class BayesACE(ACE):
    def get_initial_sample(self, instance):
        assert self.initialization == "default" or self.initialization == "guided"
        y_og = instance["class"].values[0]
        var_probs = {self.bayesian_network.cpd("class").variable_values()[i]:
                         self.bayesian_network.cpd("class").probabilities()[i] for i in
                     range(len(self.bayesian_network.cpd("class").variable_values()))}
        n_samples = int((self.population_size / (1 - var_probs[y_og])) * 2.5)
        initial_sample = self.bayesian_network.sample(n_samples, ordered=True, seed=self.seed).to_pandas()
        initial_sample = initial_sample[initial_sample["class"] != y_og].head(self.population_size).reset_index(
            drop=True)
        initial_sample = initial_sample.drop("class", axis=1)
        initial_sample = initial_sample.to_numpy()

        new_sample = self.bayesian_network.sample(self.n_vertex * self.population_size, ordered=True,
                                                  seed=self.seed).to_pandas()
        new_sample = new_sample[new_sample["class"] != y_og].head(self.population_size).reset_index(drop=True)
        new_sample = new_sample.drop("class", axis=1)
        unif_sample = np.resize(
            new_sample.to_numpy(),
            new_shape=(self.population_size, self.n_vertex * self.n_features))
        initial_sample_1 = np.hstack((unif_sample, initial_sample))
        # In semiparametrix networks, due to oversmoothing, samples my occur outside of the defined bounds. In this version, the bounds are fixed to -3 and 3.
        # In future versions, it will be a parameter
        initial_sample_1[initial_sample_1 <= -3] = -2.99999
        initial_sample_1[initial_sample_1 >= 3] = 2.99999
        if self.initialization == "default":
            return initial_sample_1
        else:

            initial_sample = self.bayesian_network.sample(n_samples, ordered=True, seed=self.seed + 1).to_pandas()
            initial_sample = initial_sample[initial_sample["class"] != y_og].head(self.population_size).reset_index(
                drop=True)
            initial_sample = initial_sample.drop("class", axis=1)
            initial_sample_2 = initial_sample.to_numpy()
            new_sample = []
            for i in initial_sample_2:
                new_sample.append(straight_path(instance.drop("class", axis=1).values, i, self.n_vertex + 2).flatten())
            initial_sample_2 = np.array(new_sample)
            initial_sample_2 = initial_sample_2[:, self.n_features:]
            noise = norm(0, 0.2).rvs(size=(initial_sample_2.shape[0], initial_sample_2.shape[1] - self.n_features))
            noise = np.hstack((noise, np.zeros(shape=(initial_sample_2.shape[0], self.n_features))))
            assert initial_sample_2.shape == noise.shape
            initial_sample_2 = initial_sample_2 + noise
            initial_sample_2[initial_sample_2 <= -3] = -2.99999
            initial_sample_2[initial_sample_2 >= 3] = 2.99999
            initial_sample = np.vstack((initial_sample_1, initial_sample_2))
            # initial_sample = initial_sample_1

            return initial_sample

    def __init__(self, bayesian_network, features, chunks, n_vertex, pop_size=100,
                 generations=10, likelihood_threshold=0.00, accuracy_threshold=0.50, penalty=1, sampling_range=(-4, 4),
                 initialization="default",
                 seed=0,
                 verbose=True):
        super().__init__(bayesian_network, features, chunks, likelihood_threshold=likelihood_threshold,
                         accuracy_threshold=accuracy_threshold, penalty=penalty, seed=seed, verbose=verbose)
        self.bayesian_network = bayesian_network
        self.n_vertex = n_vertex
        self.generations = generations
        self.population_size = pop_size
        self.sampling_range = sampling_range
        self.initialization = initialization

    def run(self, instance: pd.DataFrame, parallelize=False, return_info=False):
        termination = DefaultSingleObjectiveTermination(
            ftol=0.05 * self.n_features ** self.penalty,
            period=20
        )
        initial_sample = self.get_initial_sample(instance)
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
            algorithm = NSGA2(pop_size=self.population_size, sampling=initial_sample)

            res = minimize(problem,
                           algorithm,
                           termination=termination,
                           seed=self.seed,
                           verbose=self.verbose)
            pool.close()
        else:
            problem = BestPathFinder(bayesian_network=self.bayesian_network, instance=instance, n_vertex=self.n_vertex,
                                     penalty=self.penalty, chunks=self.chunks,
                                     likelihood_threshold=self.likelihood_threshold,
                                     accuracy_threshold=self.accuracy_threshold, sampling_range=self.sampling_range)
            algorithm = NSGA2(pop_size=self.population_size, sampling=initial_sample)

            res = minimize(problem,
                           algorithm,
                           termination=termination,  # ('n_gen', self.generations),
                           seed=self.seed,
                           verbose=self.verbose)
        if res.X is None:
            if return_info:
                return (ACEResult(None, instance.drop("class", axis=1), np.inf), res)
            return ACEResult(None, instance.drop("class", axis=1), np.inf)

        total_path = np.resize(np.append(separate_dataset_and_class(instance)[0].values[0], res.X),
                               new_shape=(self.n_vertex + 2, self.n_features))
        path_to_ret = pd.DataFrame(data=total_path,
                                   columns=self.features)
        counterfactual = path_to_ret.iloc[-1]
        path_to_compute = path(total_path, chunks=self.chunks)
        distance = 0
        for path_i in path_to_compute:
            distance += path_likelihood_length(pd.DataFrame(path_i, columns=self.features),
                                               bayesian_network=self.bayesian_network, penalty=self.penalty)
        if return_info:
            return (ACEResult(counterfactual, path_to_ret, distance), res)
        return ACEResult(counterfactual, path_to_ret, distance)
