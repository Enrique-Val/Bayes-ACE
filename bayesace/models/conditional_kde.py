from sklearn.neighbors import KernelDensity
import numpy as np
import pandas as pd

class ConditionalKDE:
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        """
        A model that estimates conditional densities using scikit-learn KDEs.
        Each class has its own KDE, and priors are stored for posterior calculations.
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.kdes = {}  # Maps each class to its KDE
        self.class_priors = {}  # Maps each class to its prior
        self.classes = []  # Ordered list of class labels for easy access

    def train(self, dataset: pd.DataFrame, class_var_name: str = "class"):
        """
        Train the KDE model.
        Parameters:
        - dataset: pd.DataFrame with the last column as the target/class variable.
        - class_var_name: name of the class column.
        """
        # Separate features and labels
        X = dataset.drop(columns=class_var_name).values
        y = dataset[class_var_name].values

        # Compute class priors
        unique_classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        self.class_priors = {cls: count / total_samples for cls, count in zip(unique_classes, counts)}

        # Store the classes in a sorted order for consistent access
        self.classes = sorted(unique_classes)

        # Fit a KDE for each class
        for cls in unique_classes:
            cls_data = X[y == cls]
            kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
            kde.fit(cls_data)
            self.kdes[cls] = kde

    def logl(self, data: pd.DataFrame, class_var_name="class"):
        """
        Compute the log-likelihood for the given data.
        Parameters:
        - data: Features and class labels.
        - class_var_name: name of the class column.
        Returns:
        - Log-likelihood
        """
        class_labels = list(self.class_priors.keys())

        log_likelihood = np.full(data.shape[0], -np.inf)

        # Transform dataset to numpy and cast class from string to numerical
        total_labels = np.unique(data[class_var_name])
        for i, label in enumerate(class_labels):
            if label in total_labels:
                log_likelihood[data[class_var_name] == label] = self.kdes[label].score_samples(data.drop(columns=class_var_name).values)
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
        p_x = np.sum(p_x_given_y * np.array([self.class_priors[cls] for cls in self.classes])[:, None], axis=0)

        # Compute P(Y|X) = P(X|Y)P(Y) / P(X)
        p_y_given_x = (p_x_given_y.T * np.array([self.class_priors[cls] for cls in self.classes])).T / p_x
        return p_y_given_x.T

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the given data.
        Parameters:
        - X: Features of shape (n_samples, n_features).
        Returns:
        - Predicted class labels for each sample.
        """
        posterior_probs = self.predict_proba(X)
        predicted_indices = np.argmax(posterior_probs, axis=1)
        return np.array(self.classes)[predicted_indices]

    def get_class_labels(self):
        return list(self.class_priors.keys()).copy()
