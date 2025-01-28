import numpy as np
import pandas as pd
import torch

from bayesace.models.conditional_density_estimator import ConditionalDE


class NanLogProb(Exception):
    pass


class ConditionalNF(ConditionalDE):
    def __init__(self, gpu_acceleration=False, verbose = False):
        super().__init__()

        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() and gpu_acceleration else "cpu")
        self.trained = False
        self.verbose = verbose

    def get_loaders(self, dataset, batch_size, proportion=0.8):
        dataset = dataset.copy()
        # Transform dataset to numpy and cast class from string to numerical
        class_column = np.zeros(len(dataset))
        for i, label in enumerate(self.class_distribution.keys()):
            class_column[dataset[self.class_var_name] == label] = i
        dataset[self.class_var_name] = class_column
        #dataset = dataset.astype(float)
        dataset_numpy = dataset.to_numpy()

        # Train validation split
        train_dataset, val_dataset = np.split(dataset_numpy,
                                              [int(proportion * len(dataset))])
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


    def logl_array(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # To be implemented by specific classes, depending on the implementation of the conditional distribution
        pass

    def logl(self, X: pd.DataFrame, y=None) -> np.ndarray:
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.to_numpy()
            data = X.copy()
            data[self.class_var_name] = y
            data[self.class_var_name] = data[self.class_var_name].astype('category')
            data[self.class_var_name] = data[self.class_var_name].cat.set_categories(self.get_class_labels())
            class_labels = list(self.class_distribution.keys())

            # Transform dataset to numpy and cast class from string to numerical
            class_column = np.zeros(data.shape[0], dtype=int)
            for i, label in enumerate(class_labels):
                class_column[data[self.class_var_name] == label] = i

            return self.logl_array(data.drop(columns=self.class_var_name).to_numpy(), class_column)
        else:
            '''X = X.values
            lls = np.zeros(X.shape[0])
            for i in range(len(self.class_distribution.keys())):
                lls = lls + np.e ** self.logl_array(X, np.repeat(i, X.shape[0]))
            return logl_from_likelihood(lls)'''
            X = X.to_numpy()
            log_likelihoods = []  # Store log-likelihoods for each class
            for i in range(len(self.class_distribution.keys())):
                log_likelihoods.append(self.logl_array(X, np.repeat(i, X.shape[0])))

            # Stack log-likelihoods and apply the log-sum-exp trick
            log_likelihoods = np.stack(log_likelihoods, axis=0)  # Shape: (num_classes, num_samples)
            max_log_likelihoods = np.max(log_likelihoods, axis=0)  # Shape: (num_samples,)

            # Log-sum-exp computation
            lls = max_log_likelihoods + np.log(np.sum(np.exp(log_likelihoods - max_log_likelihoods), axis=0))

            return lls

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

    def predict_proba(self, X: np.ndarray, output="numpy") -> np.ndarray | pd.DataFrame:
        # We want to get P(Y|x), which will be computed as P(Y|x) = P(x,Y) / P(x)
        p_xY = np.zeros((len(self.class_distribution.keys()), X.shape[0]))
        p_x = np.zeros(X.shape[0])

        for i, _ in enumerate(self.class_distribution.keys()):
            p_xY[i] = np.e ** self.logl_array(X, np.array([i] * len(X)))
            p_x = p_x + p_xY[i]
        zero_l = np.where(p_x == 0)
        p_x[p_x == 0] = 1
        for i in zero_l:
            p_xY[:, i] = 1 / len(self.class_distribution.keys())

        p_Y_given_x = p_xY.transpose() / p_x[:, None]
        if output == "pandas":
            return pd.DataFrame(p_Y_given_x, columns=self.class_distribution.keys())
        return p_Y_given_x
