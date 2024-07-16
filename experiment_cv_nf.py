import os
import random
import sys
from itertools import product

from matplotlib import pyplot as plt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_evaluations
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from bayesace.models.utils import get_data, preprocess_train_data

import pickle

sys.path.append(os.getcwd())
import argparse

from bayesace.utils import *

import time


def kfold_indices(data, k):
    fold_size = len(data) // k
    indices = np.arange(len(data))
    folds = []
    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        folds.append((train_indices, test_indices))
    return folds


# Define the number of folds (K)
k = 5
steps = 1000

# Define how the preprocessing will be done
JIT_COEF = 0.2
ELIM_OUTL = False

# Define a time limit (in hours) for execution
TIME_LIMIT = np.inf

# Define the possible parameter values IF using grid search
# The number of hidden units will be multiplied by number of features
param_grid = {
    "lr": [1e-2, 1e-3, 1e-4],
    "weight_decay": [0, 1e-4, 1e-3],
    "count_bins": [2, 4, 6],
    "hidden_units": [2, 5, 10],
    "layers": [1, 2],
    "n_flows": [1, 2, 4]
}

# Define the parameter value range IF using Bayesian optimization
param_space = [
    Real(1e-4, 1e-2, name='lr', prior='log-uniform'),
    Real(0, 1e-3, name='weight_decay'),
    Integer(2, 16, name='count_bins'),
    Integer(2, 10, name='hidden_units'),
    Integer(1, 2, name='layers'),
    Integer(1, 5, name='n_flows')
]


def cross_validate_nf(dataset, fold_indices=None, lr=None, weight_decay=None, count_bins=None, hidden_units=None,
                      layers=None,
                      n_flows=None):
    logl = []
    brier = []
    auc_list = []
    times = []
    for train_index, test_index in fold_indices:
        df_train = dataset.iloc[train_index].reset_index(drop=True)
        # df_train = preprocess_train_data(df_train, jit_coef=JIT_COEF, eliminate_outliers=ELIM_OUTL)
        df_test = dataset.iloc[test_index].reset_index(drop=True)
        t0 = time.time()
        model = NormalizingFlowModel()
        model.train(df_train, lr=lr, weight_decay=weight_decay, count_bins=count_bins,
                    hidden_units=hidden_units * (len(df_train.columns) - 1), layers=layers,
                    n_flows=n_flows, steps=steps)
        pickle.dump(model, open("model.pkl", "wb"))
        it_time = time.time() - t0
        times.append(it_time)
        logl.append(model.logl(df_test).mean())
        predictions = predict_class(df_test.drop("class", axis=1), model)
        brier.append(brier_score(df_test["class"].values, predictions))
        auc_list.append(auc(df_test["class"].values, predictions))
    print(str({"lr": lr, "weight_decay": weight_decay, "bins": count_bins, "hidden_u": hidden_units, "layers": layers,
               "n_flows": n_flows}), "normalizing flow learned")
    print(str({"Logl": np.mean(logl), "Brier": np.mean(brier), "AUC": np.mean(auc_list), "Time": np.mean(times)}))
    print()
    return np.mean(logl), np.std(logl), np.mean(brier), np.std(brier), np.mean(auc_list), np.std(auc_list), np.mean(
        times), np.std(times), {"lr": lr, "weight_decay": weight_decay, "bins": count_bins, "hidden_u": hidden_units,
                                "layers": layers,
                                "n_flows": n_flows}


def get_best_normalizing_flow(dataset, fold_indices):
    # Bayesian optimization
    # Define the objective function
    @use_named_args(param_space)
    def objective(**params):
        result = cross_validate_nf(dataset, fold_indices, **params)
        nf_logl_means = result[0]
        return -nf_logl_means  # Assuming we want to maximize loglikelihood

    # Perform Bayesian optimization
    result = gp_minimize(objective, param_space, n_calls=n_iter, random_state=0)

    # Get the best parameters
    best_params = result.x
    gt_params = {param_space[i].name: best_params[i] for i in range(len(param_space))}
    # gt_params_str = str(gt_params)
    print("Best parameters: ", best_params)
    print("Gt params:", gt_params)

    metrics = cross_validate_nf(dataset, fold_indices, **gt_params)

    # Train once again to return the object
    model = NormalizingFlowModel()
    params = {param_space[i].name: best_params[i] for i in range(len(param_space))}
    params["hidden_units"] = d * gt_params["hidden_units"]
    model.train(dataset, **gt_params, steps=steps)

    return model, metrics, result


if __name__ == "__main__":
    t_init = time.time()
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset_id", nargs='?', default=-1, type=int)
    parser.add_argument('--no_graphics', action=argparse.BooleanOptionalAction)
    parser.add_argument('--search', choices=['grid', 'bayesian'], default='bayesian')
    parser.add_argument('--n_iter', nargs='?', default=10, type=int)
    args = parser.parse_args()

    dataset_id = args.dataset_id
    GRAPHIC = not args.no_graphics
    BAYESIAN = args.search == 'bayesian'
    n_iter = args.n_iter

    print("Cross validation dataset: ", dataset_id)

    random.seed(0)

    # Load the dataset
    dataset = get_data(dataset_id)
    d = len(dataset.columns) - 1

    # Preprocess the dataset
    dataset = preprocess_train_data(dataset, jit_coef=JIT_COEF, eliminate_outliers=ELIM_OUTL)

    # Get the fold indices
    fold_indices = kfold_indices(dataset, k)

    # Storage of results
    result_metrics = ["Logl", "Brier", "AUC", "Time"]
    cartesian_product = list(product(result_metrics, ["_mean", "_std"]))

    # Flattening the list of tuples into a single list
    cartesian_product = [word1 + word2 for word1, word2 in cartesian_product]
    results_df = pd.DataFrame(
        index=cartesian_product+["params"])

    # First, learn a normalizing now and sample new synthetic data
    gt_model, metrics, result = get_best_normalizing_flow(dataset, fold_indices)
    results_df["GT_RD"] = metrics

    pickle.dump(gt_model, open("gt_nf_" + str(dataset_id) + ".pkl", "wb"))
    resampled_dataset = gt_model.sample(len(dataset))

    # Check the metrics of the model given the resampled data
    resampled_dataset_metrics =np.zeros(len(results_df)-1)
    resampled_dataset_metrics[0]=gt_model.logl(resampled_dataset).mean()
    predictions = predict_class(resampled_dataset.drop("class", axis=1), gt_model)
    resampled_dataset_metrics[2] = brier_score(resampled_dataset["class"].values, predictions)
    resampled_dataset_metrics[4] = auc(resampled_dataset["class"].values, predictions)
    resampled_dataset_metrics = list(resampled_dataset_metrics)
    resampled_dataset_metrics.append(metrics[-1])
    results_df["GT_SD"] = resampled_dataset_metrics

    # Validate Gaussian network
    bn_results = np.zeros((k, len(result_metrics)))
    slogl = []
    brier = []
    aucs = []
    times = []
    for i, (train_index, test_index) in enumerate(fold_indices):
        bn_results_i = []
        df_train = resampled_dataset.iloc[train_index].reset_index(drop=True)
        # df_train = preprocess_train_data(df_train, jit_coef=JIT_COEF, eliminate_outliers=ELIM_OUTL)
        df_test = resampled_dataset.iloc[test_index].reset_index(drop=True)
        t0 = time.time()
        network = hill_climbing(data=df_train, bn_type="CLG")
        time_i = time.time() - t0
        logl_i = network.logl(df_test).mean()
        bn_results_i.append(logl_i)
        predictions = predict_class(df_test.drop("class", axis=1), network)
        brier_i = brier_score(df_test["class"].values, predictions)
        bn_results_i.append(brier_i)
        auc_i = auc(df_test["class"].values, predictions)
        bn_results_i.append(auc_i)
        bn_results_i.append(time_i)
        bn_results[i] = bn_results_i

    bn_results_mean = np.mean(bn_results, axis=0)
    bn_results_std = np.std(bn_results, axis=0)
    bn_results = list(np.vstack((np.mean(bn_results, axis=0), np.std(bn_results, axis=0))).ravel('F'))
    bn_results.append("BIC")
    results_df["CLG"] = bn_results

    print("Bayesian network learned")
    dict_print = {"Logl": bn_results_mean[0], "Brier": bn_results_mean[1], "AUC": bn_results_mean[2],
                  "Time": bn_results_mean[3]}
    print(str(dict_print))

    # Validate normalizing flow with different params
    if not BAYESIAN:
        # Exhaustive grid search
        # Create a list of all parameter combinations
        param_combinations = list(
            product(param_grid["lr"], param_grid["weight_decay"], param_grid["bins"], param_grid["hidden_u"],
                    param_grid["layers"], param_grid["n_flows"]))

        results = []
        for i in param_combinations:
            results.append(cross_validate_nf(resampled_dataset, fold_indices, *i))
            if TIME_LIMIT * 60 * 60 - 3600 < (time.time() - t_init):
                break
        for metrics in results:
            nf_params = metrics[-1]
            results_df["NF" + str(nf_params)] = metrics[:-1]

    else:
        nf_model, metrics, result = get_best_normalizing_flow(resampled_dataset, fold_indices)
        results_df["NF"] = metrics

        pickle.dump(nf_model, open("nf_" + str(dataset_id) + ".pkl", "wb"))

        if GRAPHIC:
            # Visualize the convergence of the objective function
            plot_convergence(result)
            plt.show()

            # Visualize the convergence of the parameters
            plot_evaluations(result)
            plt.show()

        # Train neural net using the best parameters to get metrics
    print(results_df.drop("params"))
    results_df.to_csv('./results/exp_cv_2/data' + str(dataset_id) + '.csv')
