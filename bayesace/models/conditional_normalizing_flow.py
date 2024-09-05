from abc import ABC
import numpy as np
import pandas as pd
import torch
import pyro.distributions as dist

class NanLogProb(Exception):
    pass


class ConditionalNF(ABC):
    def __init__(self, gpu_acceleration=False):
        # Dict containing the marginal probability of tha target
        self.class_dist = {}

        # Auxiliary attributes filled the NF is trained. The columns exclude the class column
        self.columns = None
        self.n_dims = 0

        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() and gpu_acceleration else "cpu")
        self.trained = False

    def train(self, dataset):
        self.columns = dataset.columns[:-1]
        self.n_dims = len(self.columns)

        # Estimate the class distribution with frequentist methods
        class_labels = np.unique(dataset["class"].values)
        self.class_dist = {label: len(dataset[dataset["class"] == label]) / len(dataset) for label in class_labels}

    def get_loaders(self, dataset, batch_size):
        dataset = dataset.copy()
        # Transform dataset to numpy and cast class from string to numerical
        class_column = np.zeros(len(dataset))
        for i, label in enumerate(self.class_dist.keys()):
            class_column[dataset["class"] == label] = i
        dataset["class"] = class_column
        #dataset = dataset.astype(float)
        dataset_numpy = dataset.values

        # Train validation split
        train_dataset, val_dataset = np.split(dataset_numpy,
                                              [int(.8 * len(dataset))])
        train_dataset_tensor = torch.utils.data.TensorDataset(
            torch.from_numpy(train_dataset).to(self.device, dtype=torch.get_default_dtype())
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset_tensor, batch_size=batch_size, shuffle=True, num_workers=0
        )

        val_dataset_tensor = torch.utils.data.TensorDataset(
            torch.from_numpy(val_dataset).to(self.device, dtype=torch.get_default_dtype())
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset_tensor, batch_size=batch_size, shuffle=False, num_workers=0
        )

        return train_loader, val_loader

    def get_class_labels(self):
        return list(self.class_dist.keys()).copy()

    def get_class_distribution(self):
        return self.class_dist.copy()

    def fitted(self):
        return self.trained

    def sample(self, n_samples, ordered=True, seed=None):
        pass

    def logl_array(self, X: np.ndarray, y: np.ndarray):
        # To be implemented by specific classes, depending on the implementation of the conditional distribution
        pass

    def logl(self, data: pd.DataFrame, class_var_name="class"):
        class_labels = list(self.class_dist.keys())

        # Transform dataset to numpy and cast class from string to numerical
        class_column = np.zeros(data.shape[0], dtype=int)
        for i, label in enumerate(class_labels):
            class_column[data[class_var_name] == label] = i

        return self.logl_array(data.drop(columns=class_var_name).values, class_column)

    # The likelihood computed is just the likelihood of the data REGARDLESS of the class
    # Can also be understood as the sum of the likelihood for all classes
    def likelihood(self, data, class_var_name="class"):
        # If the class variable is passed, remove it
        if class_var_name in data.columns:
            data = data.values[:, :-1].astype(float)
        else :
            data = data.values
        lls = np.zeros(data.shape[0])
        for i in range(len(self.class_dist.keys())):
            lls = lls + np.e ** self.logl_array(data, np.repeat(i,data.shape[0]))
        return lls

    def log_likelihood(self, data, class_var_name="class"):
        return np.log(self.likelihood(data, class_var_name))
    '''
    # If the class variable is passed, remove it
    if class_var_name in data.columns:
        data = data.values[:, :-1].astype(float)
    else:
        data = data.values
    logl = self.logl_array(data, np.repeat(0, data.shape[0]))
    for i in range(len(self.class_dist.keys())-1):
        logl = logl + np.log(1+np.e ** (self.logl_array(data, np.repeat(i+1, data.shape[0]))-logl))
    return logl'''


    def predict_proba(self, data: np.ndarray, class_var_name="class") -> np.ndarray:
        # We want to get P(Y|x), which will be computed as P(Y|x) = P(x,Y) / P(x)
        p_xY = np.zeros((len(self.class_dist.keys()), data.shape[0]))
        p_x = np.zeros(data.shape[0])

        for i, _ in enumerate(self.class_dist.keys()):
            p_xY[i] = np.e ** self.logl_array(data, np.array([i] * len(data)))
            p_x = p_x + p_xY[i]
        zero_l = np.where(p_x == 0)
        p_x[p_x == 0] = 1
        for i in zero_l:
            p_xY[:, i] = 1 / len(self.class_dist.keys())

        p_Y_given_x = p_xY.transpose() / p_x[:, None]

        return p_Y_given_x

    def predict(self, data: np.ndarray) -> np.ndarray:
        # Get posterior probabilities
        posterior_probs = self.predict_proba(data)
        # Choose the label with the highest probability
        predicted_labels = np.argmax(posterior_probs, axis=1)
        return np.array([list(self.class_dist.keys())[i] for i in predicted_labels])