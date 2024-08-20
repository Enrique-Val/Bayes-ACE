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

    def __repr__(self):
        return "BayesACEResult(counterfactual="+str(self.counterfactual.values[0])+", path=pandas.DataFrame(), distance="+str(self.distance)+")"



class ACE(ABC):
    def __init__(self, density_estimator, features, chunks, likelihood_threshold=0, accuracy_threshold=0.5, penalty=1,
                 seed=0, verbose=True, parallelize=False):
        self.density_estimator = density_estimator
        self.penalty = penalty
        self.features = features
        self.n_features = len(self.features)
        self.density_estimator = density_estimator
        self.chunks = chunks
        self.likelihood_threshold = likelihood_threshold
        self.accuracy_threshold = accuracy_threshold
        self.seed = seed
        random.seed(self.seed)
        self.verbose = verbose
        self.parallelize = parallelize

    @abstractmethod
    def run(self, instance: pd.DataFrame | pd.Series, target_label) -> ACEResult:
        assert (instance["class"].values[0] != target_label)
        return