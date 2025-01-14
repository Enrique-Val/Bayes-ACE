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

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        """
        Abstract method for training the model.
        :param X:
        :param y:
        """
        self.columns = list(X.columns)
        self.n_dims = X.shape[1]

        # Estimate the class distribution with frequentist methods
        class_labels = np.unique(y.values)
        self.class_distribution = {label: len(y[y == label]) / len(y) for label in class_labels}

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
    def predict_proba(self, X: np.ndarray, output = "numpy") -> np.ndarray | pd.DataFrame:
        """
        Abstract method for computing posterior probabilities P(Y|X).
        """
        pass

    def posterior_probability(self, x_cfx: pd.DataFrame, y_og: str | list, class_var_name="class"):
        # Obtain the labels accesing either the MultiBNAF model or the cpd of the bn
        class_labels = self.get_class_labels()
        x_cfx = x_cfx.copy()
        if isinstance(y_og, str):
            x_cfx[class_var_name] = pd.Categorical([y_og] * len(x_cfx.index), categories=class_labels)
        else:
            assert len(y_og) == len(x_cfx.index)
            x_cfx[class_var_name] = pd.Categorical(y_og, categories=class_labels)
        prob = np.e ** self.logl(x_cfx)
        ll = self.likelihood(x_cfx, class_var_name)
        to_ret = np.empty(shape=len(x_cfx.index))
        # If the likelihood is 0, then classification is done with uniform probability
        to_ret[ll <= 0] = 1 / len(class_labels)
        to_ret[ll > 0] = prob[ll > 0] / ll[ll > 0]
        # if not (ll > 0).any():
        #    warnings("The instance with features "+str(x_cfx.iloc[0])+" and class " + str(y_og))
        return to_ret

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

    def logl(self, data) -> np.ndarray:
        pass

    def likelihood(self, data: pd.DataFrame, class_var_name="class") -> np.ndarray:
        pass

    def log_likelihood(self, data: pd.DataFrame, class_var_name="class") -> np.ndarray:
        ll = self.likelihood(data, class_var_name)
        logl = np.empty(shape=len(ll))
        logl[ll > 0] = np.log(ll[ll > 0])
        logl[ll <= 0] = -np.inf
        return logl
