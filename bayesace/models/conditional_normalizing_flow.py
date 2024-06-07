import numpy as np
import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from bayesace import get_data
from models.utils import preprocess_train_data, hill_climbing


class NormalizingFlowModel:
    def __init__(self, dataset_path, batch_size=1028):
        self.dist_base = None
        self.x1_dist = dist.Bernoulli(torch.tensor([0.5]))  # Initialize with a prior probability
        self.dist_x2_given_x1 = None
        self.dataset_path = dataset_path
        self.batch_size = batch_size

    def train(self, dataset, steps=1000, lr = 1e-2, weight_decay = 0, count_bins = 16, hidden_units = 10, hidden_layers = 1, n_flows = 1):
        class_labels = np.unique(dataset["class"].values)
        class_column = np.zeros(len(dataset))
        for i, label in enumerate(class_labels):
            class_column[dataset["class"] == label] = i
        dataset["class"] = class_column
        dataset = dataset.astype(float)
        dataset_numpy = dataset.values
        train_dataset, val_dataset, test_dataset = np.split(dataset_numpy,
                                              [int(.6 * len(dataset)),int(.8 * len(dataset))])
        train_dataset = preprocess_train_data(train_dataset)

        train_dataset_tensor = torch.utils.data.TensorDataset(
            torch.from_numpy(train_dataset).float()
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset_tensor, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        val_dataset_tensor = torch.utils.data.TensorDataset(
            torch.from_numpy(train_dataset).float()
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset_tensor, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        test_dataset_tensor = torch.utils.data.TensorDataset(
            torch.from_numpy(test_dataset).float()
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset_tensor, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        # Create conditional transformations
        n_dims = len(dataset.columns)-1
        x2_transform = T.conditional_spline(input_dim=n_dims, context_dim=1, count_bins=count_bins, hidden_dims=[hidden_units]*hidden_layers)
        self.dist_base = dist.MultivariateNormal(torch.zeros(n_dims), torch.eye(n_dims))
        self.dist_x2_given_x1 = dist.ConditionalTransformedDistribution(self.dist_base, [x2_transform])

        optimizer = torch.optim.Adam(x2_transform.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', cooldown=10, factor=0.5, patience=20, min_lr=5e-5)

        best_val_loss = float('inf')
        epochs_since_improvement = 0
        best_model_params = None
        patience_epochs = 100

        for step in range(steps):
            train_loss = 0.0
            for batch in train_loader:
                batch = batch[0]
                class_batch = torch.reshape(batch[:, -1], shape=(-1, 1))
                x_batch = batch[:, :-1]
                optimizer.zero_grad()
                ln_p_x2_given_x1 = self.dist_x2_given_x1.condition(class_batch).log_prob(x_batch)
                loss = -ln_p_x2_given_x1.mean()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(x_batch)
            train_loss /= len(train_loader.dataset)
            lr_scheduler.step(train_loss)

            if step % 10 == 0:
                print('step: {}, train_loss: {}'.format(step, train_loss))

            # Evaluate on validation set
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = val_batch[0]
                    class_val_batch = torch.reshape(val_batch[:, -1], shape=(-1, 1))
                    x_val_batch = val_batch[:, :-1]
                    optimizer.zero_grad()
                    ln_p_x2_given_x1_val = self.dist_x2_given_x1.condition(class_val_batch).log_prob(x_val_batch)
                    self.dist_x2_given_x1
                    val_loss += -ln_p_x2_given_x1_val.mean().item() * len(x_val_batch)
                val_loss /= len(val_loader.dataset)
                if step % 10 == 0:
                    print('Validation loss: {}'.format(val_loss))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_since_improvement = 0
                    best_model_params = self.dist_x2_given_x1.base_dist.base_dist.mean.detach().clone(), self.dist_x2_given_x1.base_dist.base_dist.stddev.detach().clone()
                    for transform in self.dist_x2_given_x1.transforms:
                        best_model_params += tuple(param.detach().clone() for param in transform.parameters())
                    #print("Updated!")
                else:
                    epochs_since_improvement += 1
                    if epochs_since_improvement >= patience_epochs:
                        print(
                            "Validation loss hasn't improved for {} epochs. Early stopping...".format(
                                patience_epochs))
                        break
                    else :
                        pass
                        '''if best_model_params:
                            self.dist_x2_given_x1.base_dist.base_dist.loc = best_model_params[0]
                            self.dist_x2_given_x1.base_dist.base_dist.scale = best_model_params[1]
                            current_param = 2
                            for transform in self.dist_x2_given_x1.transforms:
                                for param in transform.parameters():
                                    param.data = best_model_params[current_param]
                                    current_param += 1'''

            # Restore best model parameters
            if best_model_params:
                self.dist_x2_given_x1.base_dist.base_dist.loc = best_model_params[0]
                self.dist_x2_given_x1.base_dist.base_dist.scale = best_model_params[1]
                current_param = 2
                for transform in self.dist_x2_given_x1.transforms:
                    for param in transform.parameters():
                        param.data = best_model_params[current_param]
                        current_param += 1

        for val_batch in test_loader:
            val_batch = val_batch[0]
            class_val_batch = torch.reshape(val_batch[:, -1], shape=(-1, 1))
            x_val_batch = val_batch[:, :-1]
            ln_p_x2_given_x1_val = self.dist_x2_given_x1.condition(class_val_batch).log_prob(x_val_batch) + np.log(
                0.5)
            val_loss += -ln_p_x2_given_x1_val.mean().item() * len(x_val_batch)
        val_loss /= len(test_loader.dataset)
        print('Test loss: {}'.format(val_loss))

        n_samples = 2340
        y_sample = torch.reshape(torch.from_numpy(dataset_numpy[:n_samples, -1]).float(), (-1, 1))
        # y_sample = torch.from_numpy(dataset_numpy[:, -1]).float()
        #print(y_sample)
        #print(x_sample)
        new_sample = self.dist_x2_given_x1.condition(y_sample).sample((n_samples,)).reshape(-1, n_dims).float()

        for j,att in enumerate(dataset.columns[:-1]):
            plt.hist(dataset[att], bins=200, alpha=0.5, color="red", density=True)
            plt.hist(new_sample[:,j], bins=200, alpha=0.5, color="skyblue", density=True)
            plt.title(str(att))
            plt.show()
        '''plt.scatter(dataset[dataset.columns[0]],dataset[dataset.columns[1]], c="red", alpha=0.5)
        plt.scatter(new_sample[:,0],new_sample[:,1])
        plt.show()'''

def kde(data):
    data_train, data_test = np.split(data, [int(.8 * len(data))])

    data_train = preprocess_train_data(data_train)

    # Define the grid of bandwidths to search
    bandwidths = np.logspace(-1, 0, 20)

    # Use GridSearchCV to perform cross-validation for KDE
    params = {'bandwidth': bandwidths}
    grid_search = GridSearchCV(KernelDensity(), params, cv=10)

    # The scoring metric used here is the default log-likelihood
    grid_search.fit(data_train)

    # Get the best bandwidth
    best_bandwidth = grid_search.best_params_['bandwidth']
    print(f"Optimal bandwidth: {best_bandwidth}")

    kde = KernelDensity(bandwidth=best_bandwidth)
    kde.fit(data_train)

    print("kde loglikelihood: ", kde.score_samples(data_test).mean())

    sample = kde.sample(420)
    plt.scatter(dataset[dataset.columns[0]], dataset[dataset.columns[1]], c="red", alpha=0.5)
    plt.scatter(sample[:, 0], sample[:, 1])
    plt.show()

def check_hc(data):
    data_train = data.head(int(len(data)*0.8))
    data_test = data.tail(int(len(data) * 0.2))

    net = hill_climbing(data_train, "CLG")
    print("CLG loglikelihood: ", net.logl(data_test).mean())


# Example usage:
if __name__ == "__main__":
    data_path = "../../toy-3class.csv"
    dataset = pd.read_csv(data_path)
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(dataset[["x", "y"]])
    dataset[["x", "y"]] = dataset_scaled
    dataset["class"] = dataset["z"]
    dataset = dataset.drop(columns=["z"])

    dataset = get_data(44091)
    dataset = dataset.drop(columns = ["P1"])
    '''for i in [44091,44122,44123,44127,44130] :
        dataset = get_and_process_data(i)

        print("Dataset",i)
        kde(dataset.drop(columns=["class"]).values)
        print()'''

    check_hc(dataset)

    kde(dataset.drop(columns=["class"]).values)

    model = NormalizingFlowModel("../../toy-3class.csv")
    model.train(dataset)

    #kde(dataset.drop(columns=["class"]).values)

