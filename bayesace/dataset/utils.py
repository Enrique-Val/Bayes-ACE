import os
import sys

import openml as oml
import pandas as pd
from sklearn.preprocessing import StandardScaler


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
    class_processed = data[data.columns[-1]].astype('string').astype('category')
    data = data.drop(data.columns[-1], axis=1)
    data["class"] = class_processed

    if standardize:
        # Scale the rest of the dataset
        feature_columns = [i for i in data.columns if i != "class"]
        data[feature_columns] = StandardScaler().fit_transform(data[feature_columns].values)
    return data
