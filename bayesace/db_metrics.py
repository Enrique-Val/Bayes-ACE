import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler


def gower_distance(x: np.ndarray, x_cfx: np.ndarray, scaler: MinMaxScaler):
    x = scaler.transform(np.array([x]))[0]
    x_cfx = scaler.transform(np.array([x_cfx]))[0]
    return np.sum(np.abs(x - x_cfx)) / len(x)


def data_likelihood(x_cfx: np.ndarray, dataset: pd.DataFrame):
    # Exclude class, just in case
    if "class" in dataset.columns:
        dataset = dataset.drop("class", axis=1)

    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)
    x_cfx = scaler.transform(np.array([x_cfx]))[0]

    distances = np.sum(np.abs(dataset - x_cfx), axis=0) / len(x_cfx)

    return np.min(distances)
