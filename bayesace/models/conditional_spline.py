import numpy as np
import torch
import pyro.distributions as dist
import pyro.distributions.transforms as T
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
import pyarrow as pa
import math

from bayesace.models.conditional_normalizing_flow import ConditionalNF

class ConditionalSpline(ConditionalNF):
    def __init__(self, gpu_acceleration=False):
        # Call the parent constructor
        super().__init__(gpu_acceleration)

        # For our NF, we need to work with a conditional distribution
        self.dist_x_given_class: dist.ConditionalTransformedDistribution = None

    def fit(self, X, y, batch_size=1028, steps=1000, lr=1e-3, weight_decay=1e-4, count_bins=6, hidden_units=150,
              layers=1,
              n_flows=1):
        super().fit(X,y)
        dataset = X.copy()
        if isinstance(y, pd.Series):
            y = y.values
        dataset[self.class_var_name] = y
        train_loader, val_loader = self.get_loaders(dataset, batch_size)

        # Create conditional transformations
        x2_transforms = [T.conditional_spline(input_dim=self.n_dims, context_dim=1, count_bins=count_bins,
                                              hidden_dims=[hidden_units] * layers).to(self.device) for _ in
                         range(n_flows)]

        dist_base = dist.MultivariateNormal(torch.zeros(self.n_dims).to(self.device),
                                            torch.eye(self.n_dims).to(self.device))
        self.dist_x_given_class = dist.ConditionalTransformedDistribution(dist_base, x2_transforms)

        modules = torch.nn.ModuleList(x2_transforms)  #.to(self.device)

        optimizer = torch.optim.Adam(modules.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', cooldown=10, factor=0.5, patience=20, min_lr=5e-5)

        best_val_loss = float('inf')
        epochs_since_improvement = 0
        best_model_params = None
        patience_epochs = 50

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

                if not math.isnan(loss):
                    loss.backward()
                    optimizer.step()
                    instances_evaled += len(x_batch)
                    train_loss += loss.item() * len(x_batch)
                else:
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
                        # print("Validation loss hasn't improved for {} epochs. Early stopping...".format(patience_epochs))
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
        self.trained = True

    def sample(self, n_samples, seed=None) -> pd.DataFrame:
        class_sampler = dist.Categorical(torch.tensor(list(self.class_distribution.values())))
        classes = class_sampler.sample((n_samples,))
        # Reshape the class sampling
        classes_res = torch.reshape(classes, (-1, 1)).to(self.device, dtype=torch.get_default_dtype())
        X = self.dist_x_given_class.condition(classes_res).sample(
            (n_samples,)).cpu()  # .reshape(-1, self.n_dims).float()
        sample_df = pd.DataFrame(X, columns=self.columns)
        sample_df[self.class_var_name] = pd.Categorical([list(self.class_distribution.keys())[i] for i in classes],
                                            categories=self.get_class_labels())
        return sample_df

    def logl_array(self, X: np.ndarray, y: np.ndarray):
        # Cast to tensor
        y_tensor = torch.reshape(torch.tensor(y), (-1, 1)).to(self.device, dtype=torch.get_default_dtype())
        X_tensor = torch.tensor(X).to(self.device, dtype=torch.get_default_dtype())

        return (self.dist_x_given_class.condition(y_tensor).log_prob(X_tensor).cpu().detach().numpy() + np.log(
            np.array([list(self.class_distribution.values())[i] for i in y])))

