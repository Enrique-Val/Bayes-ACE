import random
from abc import ABC, abstractmethod

import pandas as pd


class ACEResult():
    def __init__(self, counterfactual, path, distance):
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


class ACE(ABC):
    def __init__(self, bayesian_network, features, chunks, likelihood_threshold=0, accuracy_threshold=0.5, penalty=1,
                 seed=0, verbose=True):
        self.bayesian_network = bayesian_network
        self.penalty = penalty
        self.features = features
        self.n_features = len(self.features)
        self.bayesian_network = bayesian_network
        self.chunks = chunks
        self.likelihood_threshold = likelihood_threshold
        self.accuracy_threshold = accuracy_threshold
        self.seed = seed
        random.seed(self.seed)
        self.verbose = verbose

    @abstractmethod
    def run(self, instance: pd.DataFrame | pd.Series) -> ACEResult:
        return