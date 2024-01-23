import random
from abc import ABC

import pandas as pd


class ACE(ABC):
    def __init__(self, bayesian_network, features, penalty, chunks, seed=0, verbose = True):
        self.bayesian_network = bayesian_network
        self.penalty = penalty
        self.features = features
        self.n_features = len(self.features)
        self.bayesian_network = bayesian_network
        self.chunks = chunks
        self.seed = seed
        random.seed(seed)
        self.verbose = verbose

    def run(self, instance: pd.DataFrame | pd.Series):
        return
