import os

import openml as oml
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


def get_data(dataset_id: int, standardize=True):
    if dataset_id < 0:
        # Load a toy dataset, but first, change the syspath to access it easily
        print(os.getcwd())
        data = pd.read_csv("./bayesace/dataset/toy-3class.csv")
        data["class"] = data["z"].astype('str').astype('category')
        data = data.drop("z", axis=1)
    else:
        # Load the dataset
        data = oml.datasets.get_dataset(dataset_id, download_data=True, download_qualities=False,
                                        download_features_meta_data=False).get_data()[0]

    # Print warning if there are missing values
    if data.isnull().values.any():
        Warning("There are missing values in the dataset. They will be removed.")

    # Shuffle the dataset
    data = data.sample(frac=1, random_state=0)

    # Reset the index
    data = data.reset_index(drop=True)

    # Transform the class into a categorical variable
    class_var_name = data.columns[-1]
    class_processed = data[class_var_name].astype('string').astype('category')
    data = data.drop(class_var_name, axis=1)
    data[class_var_name] = class_processed

    if standardize:
        # Scale the rest of the dataset
        feature_columns = [i for i in data.columns if i != class_var_name]
        data[feature_columns] = StandardScaler().fit_transform(data[feature_columns].values)
    return data

def preprocess_data(data: pd.DataFrame | np.ndarray,  eliminate_outliers=np.inf, standardize=True,
                    min_unique_vals=20, max_cum_values=3, max_instances=100000):
    array_flag = False
    if isinstance(data, np.ndarray):
        # The following code but for an array instead of a dataframe:
        data = pd.DataFrame(data)
        array_flag = True
    # Separate the target column (last column) from the features
    data = data.head(max_instances)
    target_column = data.columns[-1]
    features = data.columns[:-1]

    feature_data = data[features]
    feature_data = feature_data.loc[:, feature_data.nunique() >= min_unique_vals]

    feature_data = feature_data.loc[:, feature_data.apply(lambda x: np.sort(np.histogram(x, bins=100)[0])[-max_cum_values:].sum() < len(data)*0.95, axis=0)]
    data = pd.concat([feature_data, data[target_column]], axis=1)

    means = data[data.columns[:-1]].mean()
    stds = data[data.columns[:-1]].std()
    data = data[(np.abs((data[data.columns[:-1]] - means) / stds) < eliminate_outliers).all(axis=1)]

    if standardize:
        data[data.columns[:-1]] = StandardScaler().fit_transform(data[data.columns[:-1]].values)
    # Assert that there are no missing values
    if data.isnull().values.any():
        raise ValueError("There are missing values in the post-processed dataset.")
    if array_flag:
        return data.values
    else:
        return data


def remove_outliers(data: pd.DataFrame, outlier_threshold: float, reset_index: bool=False):
    # Select columns of numeric data
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    # Calculate z-scores
    z_scores = (data[numeric_columns] - data[numeric_columns].mean()) / data[numeric_columns].std()
    # Remove rows with z-scores greater than threshold
    data = data[(z_scores.abs() < outlier_threshold).all(axis=1)]
    if reset_index:
        data = data.reset_index(drop=True)
    return data


def remove_outliers_median(data: pd.DataFrame, perc_outliers: float, reset_index: bool=False):
    # Select columns of numeric data
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    # Calculate the number of outliers
    n_outliers = int(len(data) * perc_outliers)
    # Calculate the median
    median = data[numeric_columns].median()
    # Delete the n_outliers. The ones that are furthest (euclidean distance)
    distances = np.sqrt(np.sum((data[numeric_columns] - median) ** 2, axis=1))
    data = data[distances.nsmallest(len(data) - n_outliers).index]
    if reset_index:
        data = data.reset_index(drop=True)
    return data
