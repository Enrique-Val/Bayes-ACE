import random
from abc import ABC, abstractmethod

import pandas as pd


class ACEResult():
    def __init__(self, counterfactual, path, distance):
        self.counterfactual = counterfactual
        self.path = path
        self.distance = distance


class ACE(ABC):
    def __init__(self, bayesian_network, features, penalty, chunks, likelihood_threshold=0, accuracy_threshold=0.5,
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
