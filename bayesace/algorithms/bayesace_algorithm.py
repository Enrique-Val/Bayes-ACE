from typing import Type

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination

from bayesace.utils import *
from bayesace.algorithms.algorithm import ACE, ACEResult

MAX_VALUE_FLOAT = 1e+307
n_processes = np.max((1, int(mp.cpu_count()/1)))

# Helper function to encapsulate the parallelization logic
def process_x_i(x_i, x_og, n_vertex, n_features, chunks, features, density_estimator, penalty):
    vertex_array = np.resize(np.append(x_og, x_i), new_shape=(n_vertex + 2, n_features))
    paths = path(vertex_array, chunks=chunks)
    f1_i = path_likelihood_length(pd.DataFrame(paths, columns=features),
                                  density_estimator=density_estimator, penalty=penalty)
    if f1_i > MAX_VALUE_FLOAT:
        f1_i = MAX_VALUE_FLOAT
    return f1_i

class BestPathFinder(Problem):
    def __init__(self, density_estimator, instance, target_label, n_vertex=1, penalty=1, chunks=20, log_likelihood_threshold=-np.inf,
                 accuracy_threshold=0.05, sampling_range=(-3, 3), parallelize=False, **kwargs):
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
        self.density_estimator = density_estimator
        assert self.density_estimator.fitted()
        self.chunks = chunks
        self.log_likelihood_threshold = log_likelihood_threshold
        self.accuracy_threshold = accuracy_threshold
        self.parallelize = parallelize

    def _evaluate(self, x, out, *args, **kwargs):
        assert len(np.append(self.x_og, x[0])) == (self.n_vertex + 2) * self.n_features
        f1 = []
        if self.parallelize :
            with mp.Pool(n_processes) as pool:
                f1 = pool.starmap(process_x_i, [(x_i, self.x_og, self.n_vertex, self.n_features, self.chunks, self.features,
                                                 self.density_estimator, self.penalty) for x_i in x])
        else :
            for x_i in x:
                vertex_array = np.resize(np.append(self.x_og, x_i), new_shape=(self.n_vertex + 2, self.n_features))
                paths = path(vertex_array, chunks=self.chunks)
                f1_i = path_likelihood_length(pd.DataFrame(paths, columns=self.features),
                                              density_estimator=self.density_estimator, penalty=self.penalty)
                if f1_i > MAX_VALUE_FLOAT:
                    f1_i = MAX_VALUE_FLOAT
                f1.append(f1_i)
        out["F"] = np.column_stack(f1)

        x_cfx = pd.DataFrame(x[:,-self.n_features:], columns=self.features)
        g1 = -posterior_probability(pd.DataFrame(x_cfx, columns=self.features), self.target_label,
                                    self.density_estimator) + self.accuracy_threshold  # -likelihood(x_cfx, learned)+0.0000001
        g2 = -log_likelihood(pd.DataFrame(x_cfx, columns=self.features), self.density_estimator) + self.log_likelihood_threshold
        out["G"] = np.column_stack([g1, g2])


class BayesACE(ACE):
    def get_initial_sample(self, instance, target_label):
        assert self.initialization == "default" or self.initialization == "guided"
        y_og = instance["class"].values[0]
        class_labels = None
        probabilities = None
        if isinstance(self.density_estimator, ConditionalNF):
            class_labels = self.density_estimator.get_class_labels()
            probabilities = list(self.density_estimator.get_class_distribution().values())

        else:
            class_cpd = self.density_estimator.cpd("class")
            class_labels = class_cpd.variable_values()
            probabilities = self.density_estimator.cpd("class").probabilities()
        var_probs = {class_labels[i]: probabilities[i] for i in
                     range(len(class_labels))}

        # This first bit of code give us the initial sample, where every counterfactual is above the likelihood and
        # probability
        n_samples = min(int((self.population_size / var_probs[target_label]) * 10), 5000)
        completed = False
        initial_sample = pd.DataFrame(columns=self.features)
        count = 0
        while not completed :
            candidate_initial = self.density_estimator.sample(n_samples, ordered=True, seed=self.seed + count).to_pandas()
            candidate_initial = candidate_initial[candidate_initial["class"] == target_label]

            # Get likelihood and probability of the class
            logl = log_likelihood(candidate_initial, self.density_estimator)
            post_prob = posterior_probability(candidate_initial, target_label, self.density_estimator)

            mask = (logl > self.log_likelihood_threshold) & (post_prob > self.accuracy_threshold)
            # We also need to consider the sample range for the mask
            mask = mask & (candidate_initial[self.features] >= self.sampling_range[0]).all(axis=1)
            mask = mask & (candidate_initial[self.features] <= self.sampling_range[1]).all(axis=1)
            candidate_initial = candidate_initial[mask].reset_index(drop=True)
            candidate_initial = candidate_initial.drop("class", axis = 1)
            if candidate_initial.shape[0] > 0 :
                count = 0
            # We concatenate the new samples to the initial sample. We have to do a couple checks in case some of them is empty
            if not initial_sample.empty and not candidate_initial.empty:
                initial_sample = pd.concat([initial_sample, candidate_initial], axis=0)
            elif not initial_sample.empty:
                initial_sample = initial_sample.copy()
            elif not candidate_initial.empty:
                initial_sample = candidate_initial.copy()
            else:
                initial_sample = pd.DataFrame(columns=self.features)
            count += 1
            if initial_sample.shape[0] > self.population_size:
                completed = True
            count += 1
            if count > 100 and initial_sample.shape[0] < 1:
                warnings.warn("Could not find enough samples to start the optimization process. Please, try again "
                                "with a lower likelihood or probability threshold.")
                return None


        initial_sample = initial_sample.head(self.population_size).reset_index(drop = True)
        initial_sample = initial_sample.clip(self.sampling_range[0], self.sampling_range[1])
        initial_sample = initial_sample.to_numpy()

        if self.n_vertex == 0:
            return initial_sample

        if self.initialization == "random" :
            paths_sample = self.density_estimator.sample(self.n_vertex * self.population_size, ordered=True,
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
                paths_sample.append(np.linspace(instance.drop("class", axis=1).values, i, self.n_vertex + 2).flatten())
            paths_sample = np.array(paths_sample)
            paths_sample = paths_sample[:, self.n_features:]
            return paths_sample

    def __init__(self, density_estimator, features, n_vertex, chunks=20,
                 opt_algorithm: Type[GeneticAlgorithm] = NSGA2,
                 opt_algorithm_params:dict=None,
                 generations:int =1000,
                 log_likelihood_threshold=-np.inf, accuracy_threshold=0.50, penalty=1, sampling_range=None,
                 initialization="guided",
                 seed=0,
                 verbose=True, parallelize=False):
        super().__init__(density_estimator, features, chunks, log_likelihood_threshold=log_likelihood_threshold,
                         accuracy_threshold=accuracy_threshold, penalty=penalty, seed=seed, verbose=verbose,
                         parallelize=parallelize)
        if opt_algorithm_params is None:
            opt_algorithm_params = {"pop_size": 100}
        if "pop_size" not in opt_algorithm_params:
            opt_algorithm_params["pop_size"] = 100
        self.population_size = opt_algorithm_params["pop_size"]
        self.n_vertex = n_vertex
        self.opt_algorithm = opt_algorithm
        self.opt_algorithm_params = opt_algorithm_params
        self.generations = generations
        if sampling_range is None:
            self.sampling_range = (np.array([-3] * self.n_features), np.array([3] * self.n_features))
        else:
            self.sampling_range = sampling_range
        self.initialization = initialization

    def run(self, instance: pd.DataFrame, target_label, return_info=False):
        super().run(instance, target_label)
        termination = DefaultSingleObjectiveTermination(n_max_gen=self.generations)
        initial_sample = self.get_initial_sample(instance=instance, target_label=target_label)
        # Check if it was possible to even initialize the problem
        if initial_sample is None :
            return ACEResult(None, instance.drop("class", axis=1), np.inf)

        problem = BestPathFinder(density_estimator=self.density_estimator, instance=instance,
                                 target_label=target_label, n_vertex=self.n_vertex,
                                 penalty=self.penalty, chunks=self.chunks,
                                 log_likelihood_threshold=self.log_likelihood_threshold,
                                 accuracy_threshold=self.accuracy_threshold, sampling_range=self.sampling_range)
        # Create an algorithm of type self.opt_algorithm with the params self.opt_algorithm_params
        algorithm = self.opt_algorithm(sampling=initial_sample, **self.opt_algorithm_params)

        res = minimize(problem,
                       algorithm,
                       termination=termination,
                       seed=self.seed,
                       verbose=self.verbose)
        if len(res.F)>1 or res.X is None or res.F > MAX_VALUE_FLOAT:
            if return_info:
                return ACEResult(None, instance.drop("class", axis=1), np.inf), res
            return ACEResult(None, instance.drop("class", axis=1), np.inf)

        total_path = np.resize(np.append(separate_dataset_and_class(instance)[0].values[0], res.X),
                               new_shape=(self.n_vertex + 2, self.n_features))
        path_to_ret = pd.DataFrame(data=total_path,
                                   columns=self.features)
        counterfactual = path_to_ret.iloc[-1]
        path_to_compute = path(total_path, chunks=self.chunks)
        distance = path_likelihood_length(pd.DataFrame(path_to_compute, columns=self.features),
                                          density_estimator=self.density_estimator, penalty=self.penalty)
        if return_info:
            return ACEResult(counterfactual, path_to_ret, distance), res
        return ACEResult(counterfactual, path_to_ret, distance)
