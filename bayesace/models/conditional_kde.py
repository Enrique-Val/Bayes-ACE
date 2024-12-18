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

    def train(self, dataset: pd.DataFrame, class_var_name: str = "class"):
        """
        Train the KDE model.
        Parameters:
        - dataset: pd.DataFrame with the last column as the target/class variable.
        - class_var_name: name of the class column.
        """
        # New class attributes
        self.columns = dataset.columns[:-1]
        self.n_dims = len(self.columns)

        # Separate features and labels
        X = dataset.drop(columns=class_var_name).values
        y = dataset[class_var_name].values

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

    def logl(self, data: pd.DataFrame, class_var_name="class"):
        """
        Compute the log-likelihood for the given data.
        Parameters:
        - data: Features and class labels.
        - class_var_name: name of the class column.
        Returns:
        - Log-likelihood
        """
        class_labels = list(self.class_distribution.keys())

        log_likelihood = np.full(data.shape[0], -np.inf)

        # Transform dataset to numpy and cast class from string to numerical
        total_labels = np.unique(data[class_var_name])
        for i, label in enumerate(class_labels):
            if label in total_labels:
                log_likelihood[data[class_var_name] == label] = self.kdes[label].score_samples(data[data[class_var_name] == label].drop(columns=class_var_name).values)
        return log_likelihood


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute posterior probabilities P(Y|X).
        Parameters:
        - X: Features of shape (n_samples, n_features).
        Returns:
        - Posterior probabilities as a 2D array of shape (n_samples, n_classes).
        """
        # Compute P(X|Y) for each class
        p_x_given_y = np.array([
            np.exp(self.kdes[cls].score_samples(X)) if cls in self.kdes else np.zeros(X.shape[0])
            for cls in self.classes
        ])

        # Compute P(X) as the sum of P(X|Y)P(Y)
        p_x = np.sum(p_x_given_y * np.array([self.class_distribution[cls] for cls in self.classes])[:, None], axis=0)

        # Compute P(Y|X) = P(X|Y)P(Y) / P(X)
        p_y_given_x = (p_x_given_y.T * np.array([self.class_distribution[cls] for cls in self.classes])).T / p_x
        return p_y_given_x.T

    def get_class_labels(self):
        return list(self.class_distribution.keys()).copy()

    def sample(self, n_samples, seed=None):
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
