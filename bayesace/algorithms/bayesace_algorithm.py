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

MAX_VALUE_FLOAT = 1e+307


class BestPathFinder(ElementwiseProblem):
    def __init__(self, bayesian_network, instance, target_label, n_vertex=1, penalty=1, chunks=2, likelihood_threshold=0.05,
                 accuracy_threshold=0.05, sampling_range=(-3, 3), **kwargs):
        n_features = (len(instance.columns) - 1)
        xl = None
        xu = None
        if sampling_range is None:
            xl = np.array([-3] * (n_features * (n_vertex + 1)))
            xu = np.array([3] * (n_features * (n_vertex + 1)))
        else:
            xl = np.repeat(sampling_range[0], n_vertex + 1)
            xu = np.repeat(sampling_range[1], n_vertex + 1)
        super().__init__(n_var=n_features * (n_vertex + 1),
                         n_obj=1,
                         n_ieq_constr=2,
                         xl=xl,
                         xu=xu, **kwargs)
        self.x_og = instance.drop("class", axis=1).values
        self.y_og = instance["class"].values[0]
        self.target_label = target_label
        assert self.y_og != self.target_label
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
        if not isinstance(self.bayesian_network, NormalizingFlowModel) and not self.bayesian_network.fitted():
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
        if f1 > MAX_VALUE_FLOAT:
            f1 = MAX_VALUE_FLOAT
        out["F"] = np.column_stack([f1])

        x_cfx = self.x_og.copy()
        x_cfx[:] = x[-self.n_features:]
        g1 = -posterior_probability(pd.DataFrame(x_cfx, columns=self.features), self.target_label,
                                   self.bayesian_network) + self.accuracy_threshold  # -likelihood(x_cfx, learned)+0.0000001
        g2 = -likelihood(pd.DataFrame(x_cfx, columns=self.features), self.bayesian_network) + self.likelihood_threshold
        out["G"] = np.column_stack([g1, g2])


class BayesACE(ACE):
    def get_initial_sample(self, instance, target_label):
        assert self.initialization == "default" or self.initialization == "guided"
        y_og = instance["class"].values[0]
        class_labels = None
        probabilities = None
        if isinstance(self.bayesian_network, NormalizingFlowModel):
            class_labels = self.bayesian_network.get_class_labels()
            probabilities = list(self.bayesian_network.get_class_distribution().values())

        else:
            class_cpd = self.bayesian_network.cpd("class")
            class_labels = class_cpd.variable_values()
            probabilities = self.bayesian_network.cpd("class").probabilities()
        var_probs = {class_labels[i]: probabilities[i] for i in
                     range(len(class_labels))}

        # This first bit of code give us the initial sample, where every counterfactual is above the likelihood and probability (TODO) threshold
        n_samples = int((self.population_size / var_probs[target_label]) * 2.5*2)
        completed = False
        initial_sample = pd.DataFrame(columns=self.features)
        count = 0
        while not completed :
            candidate_initial = self.bayesian_network.sample(n_samples, ordered=True, seed=self.seed+count).to_pandas()
            candidate_initial = candidate_initial[candidate_initial["class"] == target_label]

            # Get likelihood and probability of the class
            ll = likelihood(candidate_initial, self.bayesian_network)
            post_prob = posterior_probability(candidate_initial, target_label, self.bayesian_network)

            mask = (ll > self.likelihood_threshold) & (post_prob > self.accuracy_threshold)
            candidate_initial = candidate_initial[mask].reset_index(drop=True)
            candidate_initial = candidate_initial.drop("class", axis = 1)
            initial_sample = pd.concat([initial_sample, candidate_initial])
            count += 1
            if len(initial_sample) > self.population_size:
                completed = True
            print(count)

        '''initial_sample = self.bayesian_network.sample(n_samples, ordered=True, seed=self.seed).to_pandas()
        initial_sample = initial_sample[initial_sample["class"] != y_og].head(self.population_size).reset_index(
            drop=True)
        initial_sample = initial_sample.drop("class", axis=1)'''

        initial_sample = initial_sample.head(self.population_size).reset_index(drop = True)
        initial_sample = initial_sample.clip(self.sampling_range[0], self.sampling_range[1])
        initial_sample = initial_sample.to_numpy()

        if self.initialization == "default" :
            paths_sample = self.bayesian_network.sample(self.n_vertex * self.population_size, ordered=True,
                                                      seed=self.seed).to_pandas()
            paths_sample = paths_sample.drop("class", axis=1)
            paths_sample = paths_sample.clip(self.sampling_range[0], self.sampling_range[1])
            # new_sample = new_sample[new_sample["class"] != y_og].head(self.population_size).reset_index(drop=True)
            path_sample = np.resize(
                paths_sample.to_numpy(),
                new_shape=(self.population_size, self.n_vertex * self.n_features))
            return np.hstack((path_sample, initial_sample))
        elif self.initialization == "guided":
            paths_sample = []
            for i in initial_sample:
                paths_sample.append(straight_path(instance.drop("class", axis=1).values, i, self.n_vertex + 2).flatten())
            paths_sample = np.array(paths_sample)
            paths_sample = paths_sample[:, self.n_features:]
            # TODO optional, add a bit of noise to the paths
            return np.hstack((paths_sample, initial_sample))

    def __init__(self, bayesian_network, features, chunks, n_vertex, pop_size=100,
                 generations=10, likelihood_threshold=0.00, accuracy_threshold=0.50, penalty=1, sampling_range=None,
                 initialization="default",
                 seed=0,
                 verbose=True):
        super().__init__(bayesian_network, features, chunks, likelihood_threshold=likelihood_threshold,
                         accuracy_threshold=accuracy_threshold, penalty=penalty, seed=seed, verbose=verbose)
        self.bayesian_network = bayesian_network
        self.n_vertex = n_vertex
        self.generations = generations
        self.population_size = pop_size
        if sampling_range is None:
            self.sampling_range = (np.array([-3] * self.n_features), np.array([3] * self.n_features))
        else:
            self.sampling_range = sampling_range
        self.initialization = initialization

    def run(self, instance: pd.DataFrame, target_label, parallelize=False, return_info=False):
        termination = DefaultSingleObjectiveTermination(
            ftol=0.5 * self.n_features ** self.penalty,
            period=20
        )
        termination = ("n_gen",20)
        initial_sample = self.get_initial_sample(instance=instance, target_label=target_label)
        # initialize the thread pool and create the runner
        if parallelize:
            n_processes = mp.cpu_count()
            pool = mp.Pool(n_processes)
            runner = StarmapParallelization(pool.starmap)
            problem = BestPathFinder(bayesian_network=self.bayesian_network, instance=instance,
                                     target_label=target_label, n_vertex=self.n_vertex,
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
            problem = BestPathFinder(bayesian_network=self.bayesian_network, instance=instance,
                                     target_label=target_label, n_vertex=self.n_vertex,
                                     penalty=self.penalty, chunks=self.chunks,
                                     likelihood_threshold=self.likelihood_threshold,
                                     accuracy_threshold=self.accuracy_threshold, sampling_range=self.sampling_range)
            algorithm = NSGA2(pop_size=self.population_size, sampling=initial_sample)

            res = minimize(problem,
                           algorithm,
                           termination=termination,  # ('n_gen', self.generations),
                           seed=self.seed,
                           verbose=self.verbose)
        if res.X is None or res.F > MAX_VALUE_FLOAT:
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
