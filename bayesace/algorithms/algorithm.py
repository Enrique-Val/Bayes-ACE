import random
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from bayesace import ConditionalDE


class ACEResult():
    def __init__(self, counterfactual: pd.Series, path: pd.DataFrame, distance: float):
        self.counterfactual = counterfactual
        self.path = path
        self.distance = distance

    def __eq__(self, other):
        if self is other:
            return True
        elif type(self) != type(other):
            return False
        else:
            return self.counterfactual == other.counterfactual and self.path == other.path and self.distance == other.distance

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return "BayesACEResult(counterfactual=" + str(
            self.counterfactual.to_numpy()) + ", path=pandas.DataFrame(), distance=" + str(self.distance) + ")"


class Algorithm(ABC):
    def __init__(self, density_estimator: ConditionalDE, features):
        self.density_estimator: ConditionalDE = density_estimator
        self.class_var_name = density_estimator.get_class_var_name()
        self.features = features
        self.n_features = len(self.features)


    @abstractmethod
    def run(self, instance: pd.DataFrame | pd.Series, target_label) -> ACEResult:
        assert (instance["class"].to_numpy()[0] != target_label)
        return ACEResult(None, pd.DataFrame(), np.nan)


class ACE(Algorithm):
    def __init__(self, density_estimator: ConditionalDE, features, chunks, log_likelihood_threshold=-np.inf,
                 posterior_probability_threshold=0.5,penalty=1,
                 seed=0, verbose=True, parallelize=False):
        super().__init__(density_estimator, features)
        self.penalty = penalty
        self.chunks = chunks
        self.log_likelihood_threshold = log_likelihood_threshold
        self.posterior_probability_threshold = posterior_probability_threshold
        self.seed = seed
        random.seed(self.seed)
        self.verbose = verbose
        self.parallelize = parallelize

    @abstractmethod
    def run(self, instance: pd.DataFrame | pd.Series, target_label) -> ACEResult:
        assert (instance[self.class_var_name].to_numpy()[0] != target_label)
        return ACEResult(None, pd.DataFrame(), np.nan)

    def set_log_likelihood_threshold(self, log_likelihood_threshold):
        old_log_likelihood_threshold = self.log_likelihood_threshold
        self.log_likelihood_threshold = log_likelihood_threshold
        return old_log_likelihood_threshold

    def set_posterior_probability_threshold(self, posterior_probability_threshold):
        old_posterior_probability_threshold = self.posterior_probability_threshold
        self.posterior_probability_threshold = posterior_probability_threshold
        return old_posterior_probability_threshold


