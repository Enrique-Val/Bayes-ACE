import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler


def gower_distance(x_1: np.ndarray, x_2: np.ndarray, scaler: MinMaxScaler):
    x_1 = scaler.transform(np.array([x_1]))[0]
    x_2 = scaler.transform(np.array([x_2]))[0]
    return np.sum(np.abs(x_1 - x_2)) / len(x_1)


def average_L1_distance(x_1: np.ndarray, x_2: np.ndarray, axis=0) -> float:
    return np.sum(np.abs(x_1 - x_2), axis=axis) / len(x_1)


def average_L2_distance(x_1: np.ndarray, x_2: np.ndarray, axis=0) -> float:
    return np.sqrt(np.sum((x_1 - x_2) ** 2, axis=axis) / len(x_1))


def data_likelihood(x: np.ndarray, dataset: np.ndarray, distance, k:int=1) -> float:
    '''# Exclude class, just in case
    if "class" in dataset.columns:
        dataset = dataset.drop("class", axis=1)'''

    distances = distance(dataset, x, axis=1)
    distances = np.sort(distances)[:k]
    return np.sum(distances)

if __name__ == '__main__':
    ll = data_likelihood(np.array([[1,2,3],[4,5,6],[40,50,60],[10,20,30]]), np.array([1,1,1]), average_L1_distance, k=3)
