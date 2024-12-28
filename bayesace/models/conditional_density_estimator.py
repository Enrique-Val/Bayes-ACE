from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class ConditionalDE(ABC):
    def __init__(self):
        self.class_distribution = {}  # Generalized class priors
        self.columns = None  # Feature columns
        self.n_dims = 0  # Number of dimensions
        self.classes = []  # Ordered class labels
        self.trained = False

    def train(self, dataset: pd.DataFrame, class_var_name: str = "class"):
        """
        Abstract method for training the model.
        """
        raise NotImplementedError

    def get_class_labels(self):
        return list(self.class_distribution.keys()).copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the given data.
        """
        posterior_probs = self.predict_proba(X)
        predicted_indices = np.argmax(posterior_probs, axis=1)
        return np.array(self.classes)[predicted_indices]

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Abstract method for computing posterior probabilities P(Y|X).
        """
        pass

    @abstractmethod
    def sample(self, n_samples: int, ordered=True, seed=None):
        """
        Abstract method for generating samples.
        """
        pass

    def get_class_distribution(self):
        return self.class_distribution.copy()

    def fitted(self):
        return self.trained
