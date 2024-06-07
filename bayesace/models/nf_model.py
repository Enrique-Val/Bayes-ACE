import numpy as np
import torch
import pyro
import pandas as pd
import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.distributions.transforms import AffineCoupling
from pyro.nn import DenseNN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#import seaborn as sns
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import TensorDataset, DataLoader

from bayesace import get_data

data = get_data(44123).drop("class", axis=1)
data = pd.read_csv("../../toy-3class.csv").drop("z", axis=1)
data = data.sample(frac =1)



n_dims = len(data.columns)

X = data.values

X = StandardScaler().fit_transform(X)
X = get_data(44127).drop("class", axis=1).values
n_dims = X.shape[1]

X_train, X_val, X_test = np.split(X, [int(.6 * len(X)),int(.8 * len(X))])

# Convert your dataset to a PyTorch TensorDataset
batch_size = 200
X_train_dataloader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float)), batch_size=batch_size, shuffle=True)
X_val_dataloader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float)), batch_size=batch_size, shuffle=True)

base_dist = dist.Normal(torch.zeros(n_dims), torch.ones(n_dims))
spline_transform = T.spline_coupling(n_dims, hidden_dims=[10,10])
spline_transform_2 = T.spline_coupling(n_dims)
aff = [AffineCoupling(1, DenseNN(1, [1280]*10, [1,1])) for _ in range(5)]
flow_dist = dist.TransformedDistribution(base_dist, [spline_transform])


steps = 1000
dataset = torch.tensor(X, dtype=torch.float)
optimizer = torch.optim.Adam(spline_transform.parameters(), lr=1e-2)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', cooldown=10, factor=0.5, patience=20, min_lr=5e-5, verbose=True)

best_model_params = None
best_loss = float('inf')
early_stopping_patience = 100
early_stopping_counter = 0

for epoch in range(steps + 1):
    total_loss = 0
    for batch_data in X_train_dataloader:
        optimizer.zero_grad()
        loss = -flow_dist.log_prob(batch_data[0]).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        flow_dist.clear_cache()

    if True : #epoch % 500 == 0:
        print('step: {}, loss: {}'.format(epoch, total_loss / len(X_train_dataloader)))

    # Validation loop
    validation_loss = -torch.stack(
        [
            flow_dist.log_prob(batch_data[0]).mean()
            for batch_data in X_val_dataloader
        ],
        -1,
    ).mean()

    print(f'Epoch {epoch}, Validation Loss: {validation_loss}')

    # Update learning rate scheduler
    lr_scheduler.step(validation_loss)

    # Early stopping check
    if validation_loss < best_loss:
        best_loss = validation_loss
        early_stopping_counter = 0
        best_model_params = flow_dist.base_dist.mean.detach().clone(), flow_dist.base_dist.stddev.detach().clone()
        for transform in flow_dist.transforms:
            best_model_params += tuple(param.detach().clone() for param in transform.parameters())
        print("Updated!")
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print(
                "Validation loss hasn't improved for {} epochs. Early stopping...".format(early_stopping_patience))
            break


# Restore best model parameters
if best_model_params:
    flow_dist.base_dist.loc = best_model_params[0]
    flow_dist.base_dist.scale = best_model_params[1]
    current_param = 2
    for transform in flow_dist.transforms:
        for param in transform.parameters():
            param.data = best_model_params[current_param]
            current_param += 1

validation_loss = -torch.stack(
        [
            flow_dist.log_prob(batch_data[0]).mean()
            for batch_data in X_val_dataloader
        ],
        -1,
    ).mean()

test_loss = flow_dist.log_prob(torch.from_numpy(X_test).to(torch.float)).mean()

print(f'Final {1000}, Validation Loss: {validation_loss}')
print(f'Final {1000}, Test Loss: {test_loss}')

new_samples = flow_dist.sample((1000,))
plt.scatter(new_samples[:, 0], new_samples[:, 1], alpha=0.5)
plt.show()

print()

