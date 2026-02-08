from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class ConditionalDE(ABC):
    def __init__(self):
        self.class_distribution = {}  # Generalized class priors
        self.columns = None  # Feature columns
        self.n_dims = 0  # Number of dimensions
        self.trained = False
        self.class_var_name = None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        """
        Abstract method for training the model.
        :param X:
        :param y:
        """
        self.columns = list(X.columns)
        self.n_dims = X.shape[1]

        # Estimate the class distribution with frequentist methods
        y_vals = y.to_numpy() if isinstance(y, pd.Series) else y
        class_labels = np.unique(y_vals)
        self.class_distribution = {str(label): len(y[y == label]) / len(y) for label in class_labels}

        self.class_var_name = y.name if isinstance(y, pd.Series) else "class"


    def get_class_labels(self):
        return list(self.class_distribution.keys()).copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the given data.
        """
        posterior_probs = self.predict_proba(X)
        predicted_indices = np.argmax(posterior_probs, axis=1)
        return np.array(self.get_class_labels())[predicted_indices]

    @abstractmethod
    def predict_proba(self, X: np.ndarray, output = "numpy") -> np.ndarray | pd.DataFrame:
        """
        Abstract method for computing posterior probabilities P(Y|X).
        """
        pass

    def posterior_probability(self, X: pd.DataFrame, y: str | np.ndarray):
        # Obtain the labels accesing either the MultiBNAF model or the cpd of the bn
        class_labels = self.get_class_labels()
        if isinstance(y, str):
            y = np.repeat(y, len(X.index))
        else:
            assert len(y) == len(X.index)
            assert isinstance(y, np.ndarray)
        # Get Log Likelihoods
        log_joint = self.logl(X, y)  # log P(X, y)
        log_marginal = self.logl(X)  # log P(X)

        to_ret = np.empty(shape=len(X.index))

        # Avoid division by zero and underflow by subtracting logs first
        # P(y|X) = exp( log P(X,y) - log P(X) )

        # Mask where marginal probability is effectively zero (-inf logl) or nan
        # to avoid NaN from (-inf) - (-inf)
        valid_mask = np.isfinite(log_marginal)

        # For valid points: exp(joint - marginal)
        to_ret[valid_mask] = np.exp(log_joint[valid_mask] - log_marginal[valid_mask])

        # For invalid points (where P(X)=0), fall back to uniform distribution
        # or 0, depending on your theoretical preference.
        # Your original code used uniform:
        to_ret[~valid_mask] = 1.0 / len(class_labels)

        return to_ret

    @abstractmethod
    def sample(self, n_samples: int, seed=None):
        """
        Abstract method for generating samples.
        """
        pass

    def get_class_distribution(self):
        return self.class_distribution.copy()

    def get_class_var_name(self) -> str:
        return self.class_var_name

    def fitted(self):
        return self.trained

    @abstractmethod
    def logl(self, X, y=None) -> np.ndarray:
        """
        Must return:
        - If y is provided: log P(X, y)  (Joint Log-Likelihood)
        - If y is None:     log P(X)     (Marginal Log-Likelihood)
        """
        pass

    def likelihood(self, X: pd.DataFrame, y: pd.Series | np.ndarray = None) -> np.ndarray:
        return np.exp(self.logl(X, y))

    def freeze(self):
        """
        Optional method to freeze the model parameters after training.
        """
        pass
