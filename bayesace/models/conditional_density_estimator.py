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
        class_labels = np.unique(y.to_numpy())
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
        return np.array(self.classes)[predicted_indices]

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
        prob = np.e ** self.logl(X, y)
        ll = np.e ** self.logl(X)
        to_ret = np.empty(shape=len(X.index))
        # If the likelihood is 0, then classification is done with uniform probability
        to_ret[ll <= 0] = 1 / len(class_labels)
        to_ret[ll > 0] = prob[ll > 0] / ll[ll > 0]
        # if not (ll > 0).any():
        #    warnings("The instance with features "+str(x_cfx.iloc[0])+" and class " + str(y_og))
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
        pass

    def likelihood(self, X: pd.DataFrame, y: pd.Series | np.ndarray = None) -> np.ndarray:
        return np.e ** self.logl(X, y)


'''def logl_from_likelihood(likelihood):
    logl = np.empty(shape=len(likelihood))
    logl[likelihood > 0] = np.log(likelihood[likelihood > 0])
    logl[likelihood <= 0] = -np.inf
    return logl'''
