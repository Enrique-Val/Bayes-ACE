from itertools import product

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

from bayesace import hill_climbing, get_data
from bayesace.models.conditional_spline import ConditionalSpline
from Multiprocessing import Pool, cpu_count


def kde(data):
    data_train, data_test = np.split(data, [int(.8 * len(data))])

    # data_train = preprocess_train_data(data_train)

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

    sample = kde.sample(420)
    plt.scatter(dataset[dataset.columns[0]], dataset[dataset.columns[1]], c="red", alpha=0.5)
    plt.scatter(sample[:, 0], sample[:, 1])
    # plt.show()


def check_hc(data):
    data_train = data.head(int(len(data) * 0.8))
    data_test = data.tail(int(len(data) * 0.2))

    net = hill_climbing(data_train, "CLG")
    print("CLG loglikelihood: ", net.logl(data_test).mean())


def train_and_evaluate(dataset, dataset_test, lr, wd, bins, hu, layers, n_flows):
    model = ConditionalSpline()
    model.train(dataset, lr=lr, weight_decay=wd, count_bins=bins, hidden_units=hu, hidden_layers=layers, n_flows=n_flows)
    res = np.log(model.likelihood(dataset_test)).mean()
    print("lr", lr, "   weight_decay", wd, "   bins", bins, "   hidden_units", hu, "   layers",
          layers, "   n_flows", n_flows)
    print(res)
    print()
    return res, {"lr": lr, "weight_decay": wd, "bins": bins, "hidden_u": hu, "layers": layers, "n_flows": n_flows}

# Example usage:
if __name__ == "__main__":
    for i in [44091,44122,44123,44127,44130] :
        dataset = get_data(i)

        dataset_test = dataset.tail(int(len(dataset) * 0.2))
        dataset_train = dataset.head(int(len(dataset) * 0.8))

        check_hc(dataset)

        # GRID SEARCH
        import collections

        d = len(dataset.columns)
        # Define the parameter grid
        param_grid = {
            "lr": [1e-2, 1e-3, 1e-4],
            "weight_decay": [0, 1e-4, 1e-3],
            "bins": [2, 4, 6],
            "hidden_u": [2*d,5*d,10*d],
            "layers": [1,2],
            "n_flows": [1,2,4]
        }

        # Create a list of all parameter combinations
        param_combinations = list(
            product(param_grid["lr"], param_grid["weight_decay"], param_grid["bins"], param_grid["hidden_u"],
                    param_grid["layers"],param_grid["n_flows"]))

        '''for i in param_combinations:
            train_and_evaluate(dataset, dataset_test, *i)'''

        # Use multiprocessing to speed up the grid search
        print(cpu_count()-2)
        with Pool(cpu_count()) as pool:
            results = pool.starmap(train_and_evaluate, [(dataset, dataset_test, lr, wd, bins, hu, layers, n_flows) for lr, wd, bins, hu, layers, n_flows in param_combinations])

        # Find the best result
        best_res, best_params = max(results, key=lambda x: x[0])

        print(f"Best result: {best_res}")
        print(f"Best parameters: {best_params}")
