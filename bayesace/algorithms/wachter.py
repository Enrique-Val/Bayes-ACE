import numpy as np
import pandas as pd

from bayesace import median_absolute_deviation, path_likelihood_length, path, ConditionalDE
from bayesace.algorithms.algorithm import Algorithm, ACEResult


class WachterCounterfactual(Algorithm):
    def __init__(self, density_estimator: ConditionalDE, features, dataset:pd.DataFrame,
                 target_proximity_weight=0.0, log_likelihood_threshold=-np.inf, posterior_probability_threshold=0.50):
        """
        Initialize with the density estimator, features, dataset, and weight for proximity in the loss.

        Args:
            density_estimator: The model used to evaluate counterfactual validity.
            features: List of feature names to be considered for counterfactual search.
            target_proximity_weight: Weight for proximity in the loss function.
        """
        super().__init__(density_estimator, features)
        self.target_proximity_weight = target_proximity_weight
        self.log_likelihood_threshold = log_likelihood_threshold
        self.posterior_probability_threshold = posterior_probability_threshold

        # Ensure that the dataset features match the expected features and their order
        columns_without_class = [col for col in dataset.columns if col != self.class_var_name]
        assert (columns_without_class == self.features).all()

        # Extract the features and labels from the dataset
        self.dataset_features = dataset[self.features].to_numpy()
        self.dataset_labels = dataset[self.class_var_name].to_numpy()

        # Compute the median absolute deviation for each feature
        self.feature_mad = median_absolute_deviation(self.dataset_features, axis=0)


    def _proximity_loss(self, candidates_cf, original_instance):
        """
        Compute the proximity loss for all candidates in a vectorized way.

        Args:
            candidates_cf: Candidate counterfactual instances as a NumPy array.
            original_instance: Original instance as a NumPy array.

        Returns:
            Array of weighted proximity losses.
        """

        # Calculate the absolute differences
        diff = np.abs(candidates_cf - original_instance)

        # Scale each feature difference by its standard deviation and square it
        scaled_diff = np.sum(diff / self.feature_mad, axis=1)

        if self.target_proximity_weight == 0:
            return scaled_diff
        else :
            raise NotImplementedError("Proximity weight is not 0")

    def run(self, instance: pd.DataFrame | pd.Series, target_label) -> ACEResult:
        """
        Find the best counterfactual from the dataset with the specified target label.

        Parameters
        ----------
        instance : pd.DataFrame | pd.Series
            Original instance as a pandas DataFrame or Series.
        target_label : str
            Desired target label for the counterfactual.
        dataset : pd.DataFrame
            Labeled dataset as a pandas DataFrame.

        Returns
        -------
        ACEResult
            An ACEResult containing the best counterfactual, path, and distance.
        """
        # Ensure that instance features match the expected features and its order
        columns_without_class = [col for col in instance.columns if col != self.class_var_name]
        assert (columns_without_class == self.features).all()

        # Ensure the instance class does not already match the target label
        assert (instance[self.class_var_name].to_numpy()[0] != target_label)

        # Convert the instance to a NumPy array for efficient calculation
        original_instance = instance[self.features].to_numpy().flatten()

        # Filter for instances that match the target label
        target_indices = np.where(self.dataset_labels == target_label)[0]

        if target_indices.size == 0:
            print("No instances in the dataset match the target label.")
            return ACEResult(counterfactual=pd.DataFrame(), path=pd.DataFrame(), distance=np.nan)
        # Retrieve the features of the matching instances
        candidate_cfs = self.dataset_features[target_indices]

        # Get likelihood and probability of the class
        logl = self.density_estimator.logl(pd.DataFrame(candidate_cfs,columns=self.features), target_label)
        post_prob = self.density_estimator.posterior_probability(pd.DataFrame(candidate_cfs,columns=self.features),
                                                                 target_label)

        # Filter for instances whose likelihood and posterior probability is above the threshold
        mask = (logl > self.log_likelihood_threshold) & (post_prob > self.posterior_probability_threshold)
        candidate_cfs = candidate_cfs[mask]

        if candidate_cfs.size == 0:
            print("No instances in the dataset match the likelihood and posterior probability thresholds.")
            return ACEResult(counterfactual=None, path=instance.drop(self.class_var_name,axis=1), distance=np.nan)

        # Compute the proximity loss in a vectorized way
        proximity_losses = self._proximity_loss(candidate_cfs, original_instance)

        # Identify the index of the best (minimum distance) counterfactual
        best_idx = np.argmin(proximity_losses)
        best_counterfactual = candidate_cfs[best_idx]

        # Convert the best counterfactual back to a DataFrame for consistency
        counterfactual_df = pd.Series(best_counterfactual, index=self.features)

        # The path will just contain the original and cfx instance
        vertices = pd.DataFrame([original_instance, best_counterfactual], columns=self.features)

        full_path = path(vertices.to_numpy(), chunks=20)
        path_df = pd.DataFrame(full_path, columns=vertices.columns)
        distance = path_likelihood_length(path_df, density_estimator=self.density_estimator, penalty=1)

        return ACEResult(counterfactual=counterfactual_df, path=vertices, distance=distance)
