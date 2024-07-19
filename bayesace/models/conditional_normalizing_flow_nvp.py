import numpy as np
import torch
import pyro.distributions as dist
import pyro.distributions.transforms as T
import pandas as pd
from pyro.infer import Trace_ELBO, SVI
from torch.utils.data import Dataset
import pyro
from torch import nn
from pyro.nn.dense_nn import ConditionalDenseNN
from pyro.distributions.transforms import permute, BatchNorm
from pyro.distributions.transforms.affine_coupling import ConditionalAffineCoupling
import itertools

"""
The class ConditionalNormalizingFlow is mostly implement over original code
available in the repo https://github.com/DanieleGammelli/pyro-torch-normalizing-flows/

Some modifications to work with data with more than 2 dimensions were made
"""


class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, input_dim=2, split_dim=1, context_dim=1, hidden_dim=128, num_layers=1, flow_length=10,
                 use_cuda=False):
        print(input_dim, split_dim, context_dim, hidden_dim, num_layers, flow_length)
        super(ConditionalNormalizingFlow, self).__init__()
        self.base_dist = dist.Normal(torch.zeros(input_dim),
                                     torch.ones(input_dim))  # base distribution is Isotropic Gaussian
        self.param_dims = [input_dim - split_dim, input_dim - split_dim]
        # Define series of bijective transformations
        self.transforms = [ConditionalAffineCoupling(split_dim, ConditionalDenseNN(split_dim, context_dim,
                                                                                   [hidden_dim] * num_layers,
                                                                                   self.param_dims)) for _ in
                           range(flow_length)]
        # self.perms = [permute(2, torch.tensor([1, 0])) for _ in range(flow_length)]
        self.perms = [permute(input_dim, torch.randperm(input_dim)) for _ in range(flow_length)]
        self.bns = [BatchNorm(input_dim=input_dim) for _ in range(flow_length)]
        # Concatenate AffineCoupling layers with Permute and BatchNorm Layers
        self.generative_flows = list(itertools.chain(*zip(self.transforms, self.bns, self.perms)))[
                                :-2]  # generative direction (z-->x)
        self.normalizing_flows = self.generative_flows[::-1]  # normalizing direction (x-->z)

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()
            nn.ModuleList(self.transforms).cuda()
            self.base_dist = dist.Normal(torch.zeros(input_dim).cuda(),
                                         torch.ones(input_dim).cuda())

    def model(self, X=None, H=None):
        N = len(X) if X is not None else None
        pyro.module("nf", nn.ModuleList(self.transforms))
        with pyro.plate("data", N):
            self.cond_flow_dist = self._condition(H)
            obs = pyro.sample("obs", self.cond_flow_dist, obs=X)

    def guide(self, X=None, H=None):
        pass

    def forward(self, z, H):
        zs = [z]
        _ = self._condition(H)
        for flow in self.generative_flows:
            z_i = flow(zs[-1])
            zs.append(z_i)
        return zs, z_i

    def backward(self, x, H):
        zs = [x]
        _ = self._condition(H)
        for flow in self.normalizing_flows:
            z_i = flow._inverse(zs[-1])
            zs.append(z_i)
        return zs, z_i

    def sample(self, num_samples, H):
        z_0_samples = self.base_dist.sample([num_samples])
        zs, x = self.forward(z_0_samples, H)
        return x

    def log_prob(self, x, H):
        cond_flow_dist = self._condition(H)
        return cond_flow_dist.log_prob(x)

    def _condition(self, H):
        self.cond_transforms = [t.condition(H) for t in self.transforms]
        self.generative_flows = list(itertools.chain(*zip(self.cond_transforms, self.perms)))[:-1]
        self.normalizing_flows = self.generative_flows[::-1]
        return dist.TransformedDistribution(self.base_dist, self.generative_flows)


class NormalizingFlowModelNVP:
    def __init__(self, gpu_acceleration=False, graphics=True):
        # Dict containing the marginal probability of tha target
        self.class_dist = {}
        # For our NF, we need to work with a conditional distribution
        self.dist_x_given_class: ConditionalNormalizingFlow = None

        # Auxiliary attributes filled the NF is trained. The columns exclude the class column
        self.columns = None
        self.n_dims = 0

        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() and gpu_acceleration else "cpu")

        # Check if we can plot
        self.graphics = graphics

    def train(self, dataset, steps=1000, batch_size=1028, lr=1e-3, weight_decay=1e-4, split_dim=6, hidden_units=150,
              layers=1,
              n_flows=1):
        # If graphical, then import pyplot
        if self.graphics:
            import matplotlib.pyplot as plt

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
        self.dist_x_given_class = ConditionalNormalizingFlow(input_dim=self.n_dims, split_dim=1, context_dim=1,
                                                             hidden_dim=128, num_layers=1, flow_length=10,
                                                             use_cuda=False)

        # Build SVI object
        optimizer = pyro.optim.Adam({"lr": 0.0001})
        svi = SVI(self.dist_x_given_class.model, self.dist_x_given_class.guide, optimizer, Trace_ELBO(num_particles=1))

        num_epochs = 1000
        losses = []
        val_losses = []
        pyro.clear_param_store()

        for epoch in range(steps):
            try:
                running_loss = 0
                for batch in train_loader:
                    batch = batch[0]
                    y_batch = torch.reshape(batch[:, -1], shape=(-1, 1))
                    x_batch = batch[:, :-1]
                    if self.device == "cuda":
                        y_batch, x_batch = y_batch.cuda(), x_batch.cuda()
                    loss = svi.step(x_batch, y_batch)
                    running_loss += float(loss)
                    del x_batch, y_batch
                    del loss
                losses.append(running_loss / len(train_loader.dataset))
                del running_loss

                # Validation phase
                val_running_loss = 0
                for val_batch in val_loader:
                    val_batch = val_batch[0]
                    y_val_batch = torch.reshape(val_batch[:, -1], shape=(-1, 1))
                    x_val_batch = val_batch[:, :-1]
                    if self.device == "cuda":
                        y_val_batch, x_val_batch = y_val_batch.cuda(), x_val_batch.cuda()
                    # Assuming svi.evaluate_step is the method to compute the validation loss
                    val_loss = svi.evaluate_loss(x_val_batch, y_val_batch)
                    val_running_loss += float(val_loss)
                    del x_val_batch, y_val_batch
                    del val_loss
                val_losses.append(val_running_loss / len(val_loader.dataset))
                del val_running_loss

                # Checkpoint model
                #         if running_loss <= best_loss:
                #             torch.save(cnf, "cnf_torch_save_run")
            except KeyboardInterrupt:
                if self.graphics:
                    plt.plot(losses, label='Training Loss')
                    plt.plot(val_losses, label='Validation Loss')
                    plt.legend()
                    plt.show()
                break
        if self.graphics:
            plt.plot(losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.legend()
            plt.show()

    def get_class_labels(self):
        return list(self.class_dist.keys()).copy()

    def get_class_distribution(self):
        return self.class_dist.copy()

    def sample(self, n_samples, ordered=True, seed=None):
        class_sampler = dist.Categorical(torch.tensor(list(self.class_dist.values())))
        classes = class_sampler.sample((n_samples,))
        # Reshape the class sampling
        classes_res = torch.reshape(classes.float(), (-1, 1)).to(self.device)
        X = self.dist_x_given_class.sample(num_samples=n_samples, H=classes_res).cpu().detach()
        sample_df = pd.DataFrame(X, columns=self.columns)
        sample_df["class"] = pd.Categorical([list(self.class_dist.keys())[i] for i in classes],
                                            categories=self.get_class_labels())
        # return pa.Table.from_pandas(samples_df)
        return sample_df

    def logl_array(self, X: np.ndarray, y: np.ndarray):
        # Cast to tensor
        y_tensor = torch.reshape(torch.tensor(y).float(), (-1, 1)).to(self.device)
        X_tensor = torch.tensor(X).to(self.device)

        return (self.dist_x_given_class.log_prob(X_tensor, y_tensor).cpu().detach().numpy() + np.log(
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
