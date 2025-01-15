from sklearn.neighbors import KernelDensity
import numpy as np
import pandas as pd
import pyarrow as pa

from bayesace import ConditionalDE


class ConditionalKDE(ConditionalDE):
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        """
        A model that estimates conditional densities using scikit-learn KDEs.
        Each class has its own KDE, and priors are stored for posterior calculations.
        """
        super().__init__()
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.kdes = {}  # Maps each class to its KDE

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        """
        Train the KDE model.
        Parameters:
        - dataset: pd.DataFrame with the last column as the target/class variable.
        - class_var_name: name of the class column.
        """
        super().fit(X, y)

        X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Compute class priors
        unique_classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        self.class_distribution = {cls: count / total_samples for cls, count in zip(unique_classes, counts)}

        # Store the classes in a sorted order for consistent access
        self.classes = sorted(unique_classes)

        # Fit a KDE for each class
        for cls in unique_classes:
            cls_data = X[y == cls]
            kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
            kde.fit(cls_data)
            self.kdes[cls] = kde
        self.trained = True

    def logl(self, X: pd.DataFrame, y=None):
        """
        Compute the log-likelihood for the given data.
        Parameters:
        - data: Features and class labels.
        - class_var_name: name of the class column.
        Returns:
        - Log-likelihood
        :param y:
        """
        super().logl(X, y)
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values
            X = X.values
            class_labels = list(self.class_distribution.keys())

            log_likelihood = np.full(X.shape[0], -np.inf)
            present_labels = np.unique(y)
            for label in present_labels:
                if label in class_labels:
                    indices_w_label = y == label
                    log_likelihood[indices_w_label] = (
                            self.kdes[label].score_samples(X[indices_w_label]) +
                            np.log(self.class_distribution[label]))
                else :
                    raise ValueError(f"Class {label} not found in the training data.")
            return log_likelihood
        else:
            assert "Super not triggered"


    def likelihood(self, data: pd.DataFrame, class_var_name="class") -> np.ndarray:
        # If the class variable is passed, remove it
        if class_var_name in data.columns:
            data = data.drop(columns=class_var_name)

        lls = np.zeros(data.shape[0])
        for i in self.class_distribution.keys():
            lls += np.exp(self.logl(data, np.repeat(i, data.shape[0])))
        return lls

    def predict_proba(self, X: np.ndarray, output="numpy") -> np.ndarray | pd.DataFrame:
        """
        Compute posterior probabilities P(Y|X).
        Parameters:
        - X: Features of shape (n_samples, n_features).
        Returns:
        - Posterior probabilities as a 2D array of shape (n_samples, n_classes).
        """
        # Compute P(X|Y) for each class
        p_x_given_y = np.zeros((X.shape[0], len(self.classes)))
        for i, cls in enumerate(self.classes):
            if cls not in self.kdes:
                raise ValueError(f"Class {cls} not found in the training data.")
            p_x_given_y[:, i] = np.exp(self.kdes[cls].score_samples(X))

        # Compute P(X) as the sum of P(X|Y)P(Y)
        p_x = np.sum(p_x_given_y * np.array([self.class_distribution[cls] for cls in self.classes]), axis=1)
        p_x_given_y[p_x == 0] = 1 / len(self.classes)
        p_x[p_x == 0] = 1

        # Compute P(Y|X) = P(X|Y)P(Y) / P(X), but only if p_x>0. Else, uniform probability
        p_y_given_x = p_x_given_y * np.array([self.class_distribution[cls] for cls in self.classes])
        p_y_given_x = p_y_given_x / p_x[:, np.newaxis]
        if output == "pandas":
            return pd.DataFrame(p_y_given_x, columns=self.classes)
        return p_y_given_x

    def get_class_labels(self):
        return list(self.class_distribution.keys()).copy()

    def sample(self, n_samples, ordered=True, seed=None):
        np.random.seed(seed)
        samples = []
        samples_class_label = []
        for i, cls in enumerate(self.classes):
            n_samples_cls = int(n_samples * self.class_distribution[cls]) + 1
            samples_cls = self.kdes[cls].sample(n_samples_cls)
            samples.append(samples_cls)
            samples_class_label.append(np.repeat(cls, n_samples_cls))
        samples = np.concatenate(samples, axis=0)
        samples_class_label = np.concatenate(samples_class_label, axis=0)
        samples_class_label = samples_class_label[:n_samples]
        samples = pd.DataFrame(samples, columns=self.columns).head(n_samples)
        samples["class"] = samples_class_label
        # Convert to categorical
        samples["class"] = pd.Categorical(samples["class"], categories=self.classes)
        # Shuffle the samples
        samples = samples.sample(frac=1, random_state=seed)
        return pa.Table.from_pandas(samples)
        return samples

    def sample_given_class(self, n_samples, class_label, seed=None):
        if class_label not in self.class_distribution:
            raise ValueError(f"Class {class_label} not found in the training data.")
        kde = self.kdes[class_label]
        np.random.seed(seed)
        samples = kde.sample(n_samples)
        return samples
