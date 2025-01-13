import os
import warnings

import numpy as np
import torch
import pyro.distributions as dist
import pyarrow as pa
import pandas as pd
from pyro.infer import Trace_ELBO, SVI
from torch.utils.data import Dataset
import pyro
from torch import nn
from pyro.nn.dense_nn import ConditionalDenseNN
from pyro.distributions.transforms import permute, BatchNorm
from pyro.distributions.transforms.affine_coupling import ConditionalAffineCoupling
import itertools

from bayesace.models.conditional_normalizing_flow import ConditionalNF, NanLogProb

"""
The class ConditionalNormalizingFlow is mostly implement over original code
available in the repo https://github.com/DanieleGammelli/pyro-torch-normalizing-flows/

Some modifications to work with data with more than 2 dimensions were made
"""

def weights_init(m, scale=1.0):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=scale * (1/2) * nn.init.calculate_gain('relu'))
        #nn.init.xavier_normal_(m.weight, gain=sacle * (1/3) * nn.init.calculate_gain('linear'))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, input_dim=2, split_dim=1, context_dim=1, hidden_dim=128, num_layers=1, flow_length=10,
                 use_cuda=False, perms_instantiation = None):
        super(ConditionalNormalizingFlow, self).__init__()
        self.base_dist = dist.Normal(torch.zeros(input_dim),
                                     torch.ones(input_dim))  # base distribution is Isotropic Gaussian
        self.param_dims = [input_dim - split_dim, input_dim - split_dim]
        # Define series of bijective transformations
        self.transforms = nn.ModuleList([ConditionalAffineCoupling(split_dim, ConditionalDenseNN(split_dim, context_dim,
                                                                                   [hidden_dim] * num_layers,
                                                                                   self.param_dims, nonlinearity=torch.nn.ReLU())) for _ in
                           range(flow_length)])
        # self.perms = [permute(2, torch.tensor([1, 0])) for _ in range(flow_length)]
        self.perms_instantiation = perms_instantiation
        if self.perms_instantiation is None :
            self.perms_instantiation = [torch.randperm(input_dim) for _ in range(flow_length)]
        self.perms = [permute(input_dim, self.perms_instantiation[i]) for i in range(flow_length)]
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
        self.apply(weights_init)
        '''for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data)
        raise Exception("Stop here")'''

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
        try :
            cond_flow_dist = self._condition(H)
            return cond_flow_dist.log_prob(x)
        except ValueError:
            raise NanLogProb()

    def _condition(self, H):
        self.cond_transforms = [t.condition(H) for t in self.transforms]
        self.generative_flows = list(itertools.chain(*zip(self.cond_transforms, self.perms)))[:-1]
        self.normalizing_flows = self.generative_flows[::-1]
        return dist.TransformedDistribution(self.base_dist, self.generative_flows)


class ConditionalNVP(ConditionalNF):
    def __init__(self, gpu_acceleration=False, graphics=True):
        # Call the parent constructor
        super().__init__(gpu_acceleration)

        # For our NF, we need to work with a conditional distribution
        self.dist_x_given_class: ConditionalNormalizingFlow = None

        # Flag to enable graphics
        self.graphics = graphics

    def train(self, dataset, batch_size=1028, steps=1000, lr=1e-3, weight_decay=0, split_dim=1, hidden_units=150,
              layers=1,
              n_flows=1, sam_noise=0, perms_instantiation=None,patience=15, model_pth_name="model.pth", **kwargs):
        super().train(dataset)
        self.input_dim = len(dataset.columns)-1
        self.split_dim = split_dim
        self.hidden_units = hidden_units
        self.layers = layers
        self.n_flows = n_flows
        self.steps = steps
        train_loader, val_loader = self.get_loaders(dataset, batch_size)

        # If graphical, then import pyplot
        if self.graphics:
            import matplotlib.pyplot as plt

        # Create conditional transformations
        self.dist_x_given_class = ConditionalNormalizingFlow(input_dim=self.n_dims, split_dim=split_dim, context_dim=1,
                                                             hidden_dim=hidden_units, num_layers=layers,
                                                             flow_length=n_flows,
                                                             use_cuda=False, perms_instantiation=perms_instantiation)
        self.perms_instantiation = self.dist_x_given_class.perms_instantiation.copy()

        '''# Do 10 warm up steps
        optimizer_w = pyro.optim.ClippedAdam({"lr": lr/10, "weight_decay": weight_decay, "clip_norm": 5})
        svi_w = SVI(self.dist_x_given_class.model, self.dist_x_given_class.guide, optimizer_w, Trace_ELBO(num_particles=1))
        pyro.clear_param_store()

        for epoch in range(10):
            running_loss = 0
            for batch in train_loader:
                batch = batch[0]
                y_batch = torch.reshape(batch[:, -1], shape=(-1, 1))
                x_batch = batch[:, :-1]
                if self.device == "cuda":
                    y_batch, x_batch = y_batch.cuda(), x_batch.cuda()
                loss = svi_w.step(x_batch, y_batch)
                running_loss += float(loss)
                del x_batch, y_batch, loss
            del running_loss'''

        # Build SVI object for actual training
        optimizer = torch.optim.Adam([{'params': self.dist_x_given_class.parameters()}], lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=patience*2)

        losses = []
        val_losses = []

        err_scale = 0.9
        best_val_loss = np.inf
        patience = 15
        epochs_no_improve = 0
        torch.save(self.dist_x_given_class.state_dict(), model_pth_name)

        for epoch in range(steps):
            try:
                self.dist_x_given_class.train()
                running_loss = 0
                for batch in train_loader:
                    batch = batch[0]
                    y_batch = torch.reshape(batch[:, -1], shape=(-1, 1))
                    x_batch = batch[:, :-1]
                    # Sum a certain Gaussian noise to the batch
                    if sam_noise > 0:
                        x_batch += torch.randn_like(x_batch) * sam_noise
                    if self.device == "cuda":
                        y_batch, x_batch = y_batch.cuda(), x_batch.cuda()
                    loss = -self.dist_x_given_class.log_prob(x_batch, y_batch).sum()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += float(loss)
                    del x_batch, y_batch, loss
                losses.append(running_loss / len(train_loader.dataset))

                # Validation phase
                with torch.no_grad():
                    self.dist_x_given_class.eval()
                    val_running_loss = 0
                    for val_batch in val_loader:
                        val_batch = val_batch[0]
                        y_val_batch = torch.reshape(val_batch[:, -1], shape=(-1, 1))
                        x_val_batch = val_batch[:, :-1]
                        if self.device == "cuda":
                            y_val_batch, x_val_batch = y_val_batch.cuda(), x_val_batch.cuda()
                        val_loss = -self.dist_x_given_class.log_prob(x_val_batch, y_val_batch).sum()
                        val_running_loss += float(val_loss)
                        del x_val_batch, y_val_batch, val_loss
                    val_loss = val_running_loss / len(val_loader.dataset)
                    val_losses.append(val_loss)

            except KeyboardInterrupt:
                if self.graphics:
                    print(losses)
                    print(val_losses)
                    plt.plot(losses, label='Training Loss')
                    plt.plot(val_losses, label='Validation Loss')
                    plt.legend()
                    plt.show()
                break

            except NanLogProb as e:
                if err_scale < 0.2:
                    os.remove(model_pth_name)
                    raise e
                if len(val_losses) == 0:
                    self.dist_x_given_class.load_state_dict(torch.load(model_pth_name, weights_only=True))
                    self.dist_x_given_class.apply(lambda m : weights_init(m, scale=err_scale))
                    warnings.warn("Nan in first epoch. Restarting with smaller weights")
                    err_scale -= 0.1
                    losses = []
                    val_losses = []
                    continue

                if len(losses) > len(val_losses):
                    losses.pop(-1)
                val_losses.append(val_losses[-1])
                losses.append(losses[-1])
                self.dist_x_given_class.load_state_dict(torch.load(model_pth_name, weights_only=True))
                if self.verbose:
                    print("Nan in epoch", epoch)
                continue

            if val_losses[-1] >= best_val_loss:
                epochs_no_improve += 1
            else :
                best_val_loss = val_losses[-1]
                torch.save(self.dist_x_given_class.state_dict(), model_pth_name)
                epochs_no_improve = 0

            # Menor patience que que decay en el scheduler
            if epochs_no_improve > patience:
                epochs_no_improve = 0
                self.dist_x_given_class.load_state_dict(torch.load(model_pth_name, weights_only=True))


            scheduler.step(val_losses[-1])
            if self.verbose :
                print(f"Epoch {epoch + 1}/{steps} - Loss: {losses[-1]} - Val Loss: {val_losses[-1]}")
                print("Lr:", optimizer.param_groups[0]['lr'])
        if self.graphics:
            import matplotlib.pyplot as plt
            plt.plot(losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.legend()
            plt.show()
        # Select model with median validation loss
        '''median_model_idx = np.argsort(last_vals_logl)[len(last_vals_logl)//2]
        self.dist_x_given_class.load_state_dict(last_models[median_model_idx])'''
        # Delete the pth file and load best model a last time
        self.dist_x_given_class.load_state_dict(torch.load(model_pth_name))
        os.remove(model_pth_name)
        self.trained = True

    def sample(self, n_samples, ordered=True, seed=None):
        class_sampler = dist.Categorical(torch.tensor(list(self.class_distribution.values())))
        classes = class_sampler.sample((n_samples,))
        # Reshape the class sampling
        classes_res = torch.reshape(classes, (-1, 1)).to(self.device, dtype=torch.get_default_dtype())
        X = self.dist_x_given_class.sample(num_samples=n_samples, H=classes_res).cpu().detach()
        sample_df = pd.DataFrame(X, columns=self.columns)
        sample_df["class"] = pd.Categorical([list(self.class_distribution.keys())[i] for i in classes],
                                            categories=self.get_class_labels())
        return pa.Table.from_pandas(sample_df)
        return sample_df

    def logl_array(self, X: np.ndarray, y: np.ndarray):
        # Cast to tensor
        y_tensor = torch.reshape(torch.tensor(y), (-1, 1)).to(self.device, dtype=torch.get_default_dtype())
        X_tensor = torch.tensor(X).to(self.device, dtype=torch.get_default_dtype())

        return (self.dist_x_given_class.log_prob(X_tensor, y_tensor).cpu().detach().numpy() + np.log(
            np.array([list(self.class_distribution.values())[i] for i in y])))

    def __getstate__(self):
        state = self.__dict__.copy()
        # Save the module's state_dict separately
        state['module_state_dict'] = self.dist_x_given_class.state_dict()
        # Remove the actual module from the state
        del state['dist_x_given_class']
        return state

    def __setstate__(self, state):
        # Restore the module
        state_dict = state['module_state_dict']
        del state['module_state_dict']
        self.__dict__.update(state)
        self.dist_x_given_class = ConditionalNormalizingFlow(input_dim=self.input_dim, split_dim=self.split_dim, hidden_dim=self.hidden_units, num_layers=self.layers, flow_length=self.n_flows, perms_instantiation=self.perms_instantiation)
        self.dist_x_given_class.load_state_dict(state_dict)
        # Restore the other attributes



