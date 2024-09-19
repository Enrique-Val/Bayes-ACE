import pickle

from bayesace.models.utils import get_data
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

data = get_data(44091)

data_train, data_test = np.split(data, [int(.8 * len(data))])

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
print("loglikelihood: ", grid_search.best_score_)

kde = KernelDensity(bandwidth=best_bandwidth)
kde.fit(data_train)

print("kde loglikelihood: ", kde.score_samples(data_test).mean())