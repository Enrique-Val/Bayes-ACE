import numpy as np
import torch
import pyro.distributions as dist
import pyro.distributions.transforms as T
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset


class NormalizingFlowModel:
    def __init__(self, gpu_acceleration=False):
        # Dict containing the marginal probability of tha target
        self.class_dist = {}
        # For our NF, we need to work with a conditional distribution
        self.dist_x_given_class: dist.ConditionalTransformedDistribution = None

        # Auxiliary attributes filled the NF is trained. The columns exclude the class column
        self.columns = None
        self.n_dims = 0

        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() and gpu_acceleration else "cpu")

    def train(self, dataset, steps=1000, batch_size=1028, lr=1e-3, weight_decay=1e-4, count_bins=6, hidden_units=150,
              layers=1,
              n_flows=1):
        self.columns = dataset.columns[:-1]
        self.n_dims = len(self.columns)

        # Estimate the class distribution with frequentist methods
        class_labels = np.unique(dataset["class"].values)
        self.class_dist = {label: len(dataset[dataset["class"] == label]) / len(dataset) for label in class_labels}

        # Transform dataset to numpy and cast class from string to numerical
        class_column = np.zeros(len(dataset))
        for i, label in enumerate(class_labels):
            class_column[dataset["class"] == label] = i
        dataset["class"] = class_column
        dataset = dataset.astype(float)
        dataset_numpy = dataset.values

        # Train validation split
        train_dataset, val_dataset = np.split(dataset_numpy,
                                              [int(.8 * len(dataset))])
        # train_dataset = preprocess_train_data(train_dataset)

        train_dataset_tensor = torch.utils.data.TensorDataset(
            torch.from_numpy(train_dataset).float().to(self.device)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset_tensor, batch_size=batch_size, shuffle=True, num_workers=0
        )

        val_dataset_tensor = torch.utils.data.TensorDataset(
            torch.from_numpy(val_dataset).float().to(self.device)
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset_tensor, batch_size=batch_size, shuffle=False, num_workers=0
        )

        # Create conditional transformations
        x2_transforms = [T.conditional_spline(input_dim=self.n_dims, context_dim=1, count_bins=count_bins,
                                            hidden_dims=[hidden_units] * layers).to(self.device) for _ in range(n_flows)]
        '''
        # Compute covariance matrix
        dataset_numpy_no_class = dataset_numpy[:, :-1]
        mu = dataset_numpy_no_class.mean(dim=0)

        # Compute the covariance matrix (sigma)
        # First, center the data by subtracting the mean
        centered_data = dataset_numpy_no_class - mu

        # Compute the covariance matrix
        sigma = (centered_data.T @ centered_data) / (centered_data.shape[0])

        self.dist_base = dist.MultivariateNormal(torch.zeros(self.n_dims), torch.tensor(sigma))
        '''
        dist_base = dist.MultivariateNormal(torch.zeros(self.n_dims), torch.eye(self.n_dims))
        self.dist_x_given_class = dist.ConditionalTransformedDistribution(dist_base, x2_transforms)

        modules = torch.nn.ModuleList(x2_transforms)

        optimizer = torch.optim.Adam(modules.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', cooldown=10, factor=0.5, patience=20, min_lr=5e-5)

        best_val_loss = float('inf')
        epochs_since_improvement = 0
        best_model_params = None
        patience_epochs = 100

        for step in range(steps):
            train_loss = 0.0
            instances_evaled = 0
            for batch in train_loader:
                batch = batch[0]
                class_batch = torch.reshape(batch[:, -1], shape=(-1, 1))
                x_batch = batch[:, :-1]
                optimizer.zero_grad()
                ln_p_x2_given_x1 = self.dist_x_given_class.condition(class_batch).log_prob(x_batch)
                loss = -ln_p_x2_given_x1.mean()
                if loss is not None:
                    loss.backward()
                    optimizer.step()
                    instances_evaled += len(x_batch)
                    train_loss += loss.item() * len(x_batch)
                else :
                    print("Loss is none at iteration {} and could not being backpropagated".format(step))
            train_loss /= instances_evaled
            lr_scheduler.step(train_loss)

            if step % 10 == 0 and False:
                print('step: {}, train_loss: {}'.format(step, train_loss))

            # Evaluate on validation set
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = val_batch[0]
                    class_val_batch = torch.reshape(val_batch[:, -1], shape=(-1, 1))
                    x_val_batch = val_batch[:, :-1]
                    optimizer.zero_grad()
                    ln_p_x2_given_x1_val = self.dist_x_given_class.condition(class_val_batch).log_prob(x_val_batch)
                    val_loss += -ln_p_x2_given_x1_val.mean().item() * len(x_val_batch)
                val_loss /= len(val_loader.dataset)
                if step % 10 == 0 and False:
                    print('Validation loss: {}'.format(val_loss))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_since_improvement = 0
                    best_model_params = self.dist_x_given_class.base_dist.base_dist.mean.detach().clone(), self.dist_x_given_class.base_dist.base_dist.stddev.detach().clone()
                    for transform in self.dist_x_given_class.transforms:
                        best_model_params += tuple(param.detach().clone() for param in transform.parameters())
                    # print("Updated!")
                else:
                    epochs_since_improvement += 1
                    if epochs_since_improvement >= patience_epochs:
                        #print("Validation loss hasn't improved for {} epochs. Early stopping...".format(patience_epochs))
                        break
            # Restore best model parameters
            if best_model_params:
                self.dist_x_given_class.base_dist.base_dist.loc = best_model_params[0]
                self.dist_x_given_class.base_dist.base_dist.scale = best_model_params[1]
                current_param = 2
                for transform in self.dist_x_given_class.transforms:
                    for param in transform.parameters():
                        param.data = best_model_params[current_param]
                        current_param += 1

        '''n_samples = 2340
        y_sample = torch.reshape(torch.from_numpy(dataset_numpy[:n_samples, -1]).float(), (-1, 1))
        # y_sample = torch.from_numpy(dataset_numpy[:, -1]).float()
        # print(y_sample)
        # print(x_sample)
        new_sample = self.dist_x_given_class.condition(y_sample).sample((n_samples,)).reshape(-1, n_dims).float()

        for j, att in enumerate(dataset.columns[:-1]):
            plt.hist(dataset[att], bins=200, alpha=0.5, color="red", density=True)
            plt.hist(new_sample[:, j], bins=200, alpha=0.5, color="skyblue", density=True)
            plt.title(str(att))
            plt.show()
        plt.scatter(dataset[dataset.columns[0]],dataset[dataset.columns[1]], c="red", alpha=0.5)
        plt.scatter(new_sample[:,0],new_sample[:,1])
        plt.show()'''

    def get_class_labels(self):
        return list(self.class_dist.keys()).copy()

    def get_class_distribution(self):
        return self.class_dist.copy()

    def sample(self, n_samples, ordered=True, seed=None):
        print(np.array(self.class_dist.values()))
        class_sampler = dist.Categorical(torch.tensor(list(self.class_dist.values())))
        classes = class_sampler.sample((n_samples,))
        # Reshape the class sampling
        classes_res = torch.reshape(classes.float(), (-1, 1)).to(self.device)
        X = self.dist_x_given_class.condition(classes_res).sample((n_samples,))  # .reshape(-1, self.n_dims).float()
        sample_df = pd.DataFrame(X, columns=self.columns)
        sample_df["class"] = pd.Categorical([list(self.class_dist.keys())[i] for i in classes], categories=self.get_class_labels())
        # return pa.Table.from_pandas(samples_df)
        return sample_df

    def logl_array(self, X: np.ndarray, y: np.ndarray):
        # Cast to tensor
        y_tensor = torch.reshape(torch.tensor(y).float(), (-1, 1)).to(self.device)
        X_tensor = torch.tensor(X).to(self.device)

        return (self.dist_x_given_class.condition(y_tensor).log_prob(X_tensor).cpu().detach().numpy() + np.log(
            np.array([list(self.class_dist.values())[i] for i in y])))

    def logl(self, data: pd.DataFrame, class_var_name="class"):
        class_labels = list(self.class_dist.keys())

        # Transform dataset to numpy and cast class from string to numerical
        class_column = np.zeros(len(data), dtype=int)
        for i, label in enumerate(class_labels):
            class_column[data[class_var_name] == label] = i

        return self.logl_array(data.drop(columns=class_var_name).values, class_column)




    # The likelihood computed is just the likelihood of the data REGARDLESS of the class
    # Can also be understood as the sum of the likelihood for all classes
    def likelihood(self, data, class_var_name="class"):
        # If the class variable is passed, remove it
        if class_var_name in data.columns:
            data = data.drop(class_var_name, axis=1)
        lls = np.zeros(len(data))
        dataset_test = data.copy()
        for i in self.class_dist.keys():
            dataset_test["class"] = i
            lls = lls + np.e ** self.logl(dataset_test)
        return lls

    def predict_proba(self, data: np.ndarray, class_var_name="class") -> np.ndarray:
        # We want to get P(Y|x), which will be computed as P(Y|x) = P(x,Y) / P(x)
        p_xY = np.zeros((len(self.class_dist.keys()), data.shape[0]))
        p_x = np.zeros(data.shape[0])

        for i,_ in enumerate(self.class_dist.keys()):
            p_xY[i] = np.e ** self.logl_array(data, np.array([i] * len(data)))
            p_x = p_x + p_xY[i]
        zero_l = np.where(p_x == 0)
        p_x[p_x ==0] = 1
        for i in zero_l :
            p_xY[:,i] = 1 / len(self.class_dist.keys())

        p_Y_given_x = p_xY.transpose() / p_x[:, None]

        return p_Y_given_x

    def predict(self, data: np.ndarray) -> np.ndarray:
        # Get posterior probabilities
        posterior_probs = self.predict_proba(data)
        # Choose the label with the highest probability
        predicted_labels = np.argmax(posterior_probs, axis=1)
        return np.array([list(self.class_dist.keys())[i] for i in predicted_labels])


