import os
import random
import sys
from itertools import product

import pandas as pd
import torch
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_evaluations
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from bayesace.models.conditional_nvp import ConditionalNVP
from bayesace.models.conditional_spline import ConditionalSpline
from bayesace.models.utils import get_data, preprocess_data

import pickle

sys.path.append(os.getcwd())
import argparse

from bayesace.utils import *
import multiprocessing as mp

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
k = 10
steps = 1000
n_batches = 10

# Define the number of iterations for Bayesian optimization
default_opt_iter = 50

# Define how the preprocessing will be done
JIT_COEF = 1
ELIM_OUTL = True
min_unique_vals = 50
max_unique_vals_to_jit = 0.05
max_cum_values = 3
minimum_spike_jitter = 0.0

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
    Real(5e-5, 1e-3, name='lr', prior='log-uniform'),
    Real(1e-4, 5e-2, name='weight_decay'),
    Integer(2, 16, name='count_bins'),
    Integer(2, 10, name='hidden_units'),
    Integer(1, 5, name='layers'),
    Integer(1, 10, name='n_flows')
]


def cross_validate_bn(dataset, fold_indices=None):
    # Validate Gaussian network
    bn_results = np.zeros((k, len(result_metrics)))
    for i, (train_index, test_index) in enumerate(fold_indices):
        bn_results_i = []
        df_train = dataset.iloc[train_index].reset_index(drop=True)
        df_test = dataset.iloc[test_index].reset_index(drop=True)
        t0 = time.time()
        network = hill_climbing(data=df_train, bn_type="CLG")
        time_i = time.time() - t0
        tmp = network.logl(df_test)
        logl_i = tmp.mean()
        logl_std_i = tmp.std()
        bn_results_i.append(logl_i)
        bn_results_i.append(logl_std_i)
        predictions = predict_class(df_test.drop("class", axis=1), network)
        brier_i = brier_score(df_test["class"].values, predictions)
        bn_results_i.append(brier_i)
        auc_i = auc(df_test["class"].values, predictions)
        bn_results_i.append(auc_i)
        bn_results_i.append(time_i)
        bn_results[i] = bn_results_i

    bn_results_mean = np.mean(bn_results, axis=0)
    bn_results = list(np.vstack((np.mean(bn_results, axis=0), np.std(bn_results, axis=0))).ravel('F'))
    bn_results.append("BIC")

    print("Bayesian network learned")
    dict_print = {result_metrics[i]: bn_results_mean[i] for i in range(len(result_metrics))}
    print(str(dict_print))
    print()

    return bn_results

def train_nf_and_get_results(df_train, df_test, model_type="NVP", batch_size=64, lr=None, weight_decay=None,
                             count_bins=None, hidden_units=None, layers=None, n_flows=None, perms_instantiation=None) :
    d = df_train.shape[1] - 1
    t0 = time.time()
    model = None
    if model_type == "NVP":
        model = ConditionalNVP(graphics=False)
        model.train(df_train, lr=lr, weight_decay=weight_decay, split_dim=d // 2,
                    hidden_units=hidden_units * d, layers=layers,
                    n_flows=n_flows, steps=steps, batch_size=batch_size, perms_instantiation=perms_instantiation)
    elif model_type == "Spline":
        model = ConditionalSpline()
        model.train(df_train, lr=lr, weight_decay=weight_decay, count_bins=count_bins,
                    hidden_units=hidden_units * d, layers=layers,
                    steps=steps, batch_size=batch_size)
    it_time = time.time() - t0
    logl_data = model.logl(df_test)
    logl = logl_data.mean()
    logl_std = logl_data.std()
    predictions = predict_class(df_test.drop("class", axis=1), model)
    brier = brier_score(df_test["class"].values, predictions)
    auc_res = auc(df_test["class"].values, predictions)
    return {"Logl": np.mean(logl), "LoglStd": np.mean(logl_std), "Brier": np.mean(brier), "AUC": np.mean(auc_res),
            "Time": np.mean(it_time)}

def cross_validate_nf(dataset, fold_indices=None, model_type="NVP", batch_size=64, lr=None, weight_decay=None,
                      count_bins=None, hidden_units=None,
                      layers=None,
                      n_flows=None, perms_instantiation=None, parallelize = False):
    param_dict = None
    if model_type == "NVP":
        param_dict = {"lr": lr, "weight_decay": weight_decay, "hidden_u": hidden_units,
                      "layers": layers, "n_flows": n_flows}
        if perms_instantiation is None :
            perms_instantiation = [torch.randperm(d) for _ in range(n_flows)]
    elif model_type == "Spline":
        param_dict = {"lr": lr, "weight_decay": weight_decay, "count_bins": count_bins, "hidden_u": hidden_units,
                      "layers": layers}

    cv_iter_results = []
    if not parallelize :
        for train_index, test_index in fold_indices:
            df_train = dataset.iloc[train_index].reset_index(drop=True)
            df_test = dataset.iloc[test_index].reset_index(drop=True)
            cv_iter_results.append(train_nf_and_get_results(df_train, df_test, model_type=model_type, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                                     count_bins=count_bins, hidden_units=hidden_units, layers=layers, n_flows=n_flows,
                                     perms_instantiation=perms_instantiation))
    elif parallelize :
        pool = mp.Pool(k)
        cv_iter_results = pool.starmap(train_nf_and_get_results, [(dataset.iloc[train_index].reset_index(drop=True), dataset.iloc[test_index].reset_index(drop=True), model_type, batch_size, lr, weight_decay, count_bins, hidden_units, layers, n_flows, perms_instantiation) for train_index, test_index in fold_indices])
        pool.close()
    cv_results = {"Logl": [], "LoglStd": [], "Brier": [], "AUC": [], "Time": []}
    for cv_iter_result in cv_iter_results:
        for key in cv_results.keys():
            cv_results[key].append(cv_iter_result[key])


    print(str(param_dict), "normalizing flow learned")
    cv_results_summary = {"Logl_mean": np.mean(cv_results["Logl"]), "Logl_std": np.std(cv_results["Logl"]),
                          "LoglStd_mean": np.mean(cv_results["LoglStd"]), "LoglStd_std": np.std(cv_results["LoglStd"]),
                          "Brier_mean": np.mean(cv_results["Brier"]), "Brier_std": np.std(cv_results["Brier"]),
                          "AUC_mean": np.mean(cv_results["AUC"]), "AUC_std": np.std(cv_results["AUC"]),
                          "Time_mean": np.mean(cv_results["Time"]), "Time_std": np.std(cv_results["Time"])}
    print(cv_results_summary)
    print()
    return [cv_results_summary[i] for i in cv_results_summary.keys()] + [param_dict]


def get_best_normalizing_flow(dataset, fold_indices, model_type="NVP", parallelize = False):
    # Bayesian optimization

    # List to store the random permutations
    perms_list = []

    # Define the objective function
    @use_named_args(param_space)
    def objective(**params):
        # First, if using NVP, define a random permutation and store it in a global list
        perms_instantiation = None
        if model_type == "NVP":
            perms_instantiation = [torch.randperm(d) for _ in range(params["n_flows"])]
            perms_list.append(perms_instantiation)
        result = None
        try:
            result = cross_validate_nf(dataset, fold_indices, model_type=model_type, batch_size=batch_size,
                                       perms_instantiation=perms_instantiation,parallelize=parallelize,**params)
            nf_logl_means = result[0]
            return -nf_logl_means  # Assuming we want to maximize loglikelihood
        except ValueError as e:
            if e.args[0][:30] == "Error while computing log_prob":
                to_print = ("Error while computing log_prob. Returning a high value for the objective function. "
                            "Consider smoothing data, decreasing the value of the lr or the complexity of the "
                            "network.") + str(params)
                warnings.warn(to_print, RuntimeWarning)
                print()
                return 3e+38
            else:
                raise e

    # Get number of features
    d = len(dataset.columns) - 1

    # Perform Bayesian optimization
    result = gp_minimize(objective, param_space, n_calls=n_iter, random_state=0)

    # Get the permutations from temporary file

    # Get the best permutation (optimized at random)
    best_iter = np.argmin(result.func_vals)
    best_perm = perms_list[best_iter]

    # Get the best parameters
    best_params = result.x
    gt_params = {param_space[i].name: best_params[i] for i in range(len(param_space))}
    # gt_params_str = str(gt_params)
    print("Best parameters: ", best_params)
    print("Gt params:", gt_params)

    # Cross validate again to get the rest of the metrics
    metrics = cross_validate_nf(dataset, fold_indices, batch_size=batch_size,
                                perms_instantiation=best_perm, **gt_params)
    params = {param_space[i].name: best_params[i] for i in range(len(param_space))}
    params["hidden_units"] = d * gt_params["hidden_units"]

    # Train once again to return the object
    model = None
    if model_type == "NVP":
        model = ConditionalNVP(graphics=False)
        params["split_dim"] = d // 2
    elif model_type == "Splines":
        model = ConditionalSpline()
    model.train(dataset, **params, steps=steps, batch_size=batch_size, perms_instantiation=best_perm)
    return model, metrics, result


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    t_init = time.time()
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset_id", nargs='?', default=-1, type=int)
    parser.add_argument('--no_graphics', action=argparse.BooleanOptionalAction)
    parser.add_argument("--type", choices=["NVP", "Spline"], default="NVP")
    parser.add_argument('--search', choices=['grid', 'bayesian'], default='bayesian')
    parser.add_argument('--n_iter', nargs='?', default=default_opt_iter, type=int)
    parser.add_argument('--part', choices=['full', 'rd', 'sd'], default='full')
    parser.add_argument('--parallelize', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    dataset_id = args.dataset_id
    GRAPHIC = not args.no_graphics
    BAYESIAN = args.search == 'bayesian'
    n_iter = args.n_iter
    parallelize = args.parallelize

    directory_path = "./results/exp_cv_2/" + str(dataset_id) + "/"
    if not os.path.exists(directory_path):
        # If the directory does not exist, create it
        os.makedirs(directory_path)

    print("Cross validation dataset: ", dataset_id)

    # Set the seed
    random.seed(0)

    # Set the metrics to evaluate
    result_metrics = ["Logl", "LoglStd", "Brier", "AUC", "Time"]

    if args.part == 'rd' or args.part == 'full':
        # Load the dataset and preprocess it
        dataset = get_data(dataset_id, standardize=True)
        dataset = preprocess_data(dataset, standardize=True, eliminate_outliers=ELIM_OUTL, jit_coef=JIT_COEF,
                                  min_unique_vals=min_unique_vals,
                                  max_unique_vals_to_jit=max_unique_vals_to_jit * len(dataset), max_instances=50000,
                                  minimum_spike_jitter=minimum_spike_jitter, max_cum_values=max_cum_values)
        d = len(dataset.columns) - 1
        split_dim = d // 2
        n_instances = dataset.shape[0]
        global batch_size
        batch_size = int((n_instances / n_batches)+1)

        # In case we use NVP, we need to add the split_dim parameter
        if args.type == "NVP":
            param_space.pop(2)

        # Get the fold indices
        fold_indices = kfold_indices(dataset, k)

        # Storage of results
        cartesian_product = list(product(result_metrics, ["_mean", "_std"]))
        # Flattening the list of tuples into a single list
        cartesian_product = [word1 + word2 for word1, word2 in cartesian_product]
        results_df = pd.DataFrame(
            index=cartesian_product + ["params"])
        results_df.index.name = str(dataset_id)

        # Validate Gaussian network
        bn_results = cross_validate_bn(dataset, fold_indices)
        results_df["CLG_RD"] = bn_results

        # First, learn a normalizing now and sample new synthetic data
        gt_model, metrics, result = get_best_normalizing_flow(dataset, fold_indices, model_type=args.type, parallelize = parallelize)
        results_df["GT_RD"] = metrics
        if GRAPHIC:
            # Visualize the convergence of the objective function
            plot_convergence(result)
            plt.savefig(directory_path + "convergence_RD_" + str(dataset_id) + ".png")
            plt.clf()

            # Visualize the convergence of the parameters
            plot_evaluations(result)
            plt.savefig(directory_path + "evaluations_RD_" + str(dataset_id) + ".png")
            plt.clf()

        pickle.dump(gt_model, open(directory_path + "gt_nf_" + str(dataset_id) + ".pkl", "wb"))
        resampled_dataset = gt_model.sample(np.min((len(dataset), 50000))).to_pandas()
        resampled_dataset.to_csv(directory_path + "resampled_data" + str(dataset_id) + ".csv")

        if args.part == 'rd':
            print(results_df.drop("params"))
        results_df.to_csv(directory_path + 'data_' + str(dataset_id) + '.csv')

    if args.part == 'sd' or args.part == 'full':
        gt_model = pickle.load(open(directory_path + "gt_nf_" + str(dataset_id) + ".pkl", "rb"))
        resampled_dataset = pd.read_csv(directory_path + "resampled_data" + str(dataset_id) + ".csv", index_col=0)
        resampled_dataset["class"] = resampled_dataset["class"].astype('str').astype('category')
        results_df = pd.read_csv(directory_path + 'data_' + str(dataset_id) + '.csv', index_col=0)
        if len(results_df.columns) > 2:
            results_df = results_df.drop("CLG", axis=1)
            results_df = results_df.drop("GT_SD", axis=1)
            results_df = results_df.drop("NF", axis=1)

        # New fold indices
        fold_indices = kfold_indices(resampled_dataset, k)

        # If we use NVP, we need to add the split_dim parameter
        d = len(resampled_dataset.columns) - 1
        if args.type == "NVP" and args.part == 'sd':
            param_space.pop(2)

        # Check the metrics of the model given the resampled data
        resampled_dataset_metrics = np.zeros(len(results_df) - 1)
        tmp = gt_model.logl(resampled_dataset)
        resampled_dataset_metrics[0] = tmp.mean()
        resampled_dataset_metrics[2] = tmp.std()
        predictions = predict_class(resampled_dataset.drop("class", axis=1), gt_model)
        resampled_dataset_metrics[4] = brier_score(resampled_dataset["class"].values, predictions)
        resampled_dataset_metrics[6] = auc(resampled_dataset["class"].values, predictions)
        resampled_dataset_metrics = list(resampled_dataset_metrics)
        resampled_dataset_metrics.append(results_df["GT_RD"].values[-1])
        results_df["GT_SD"] = resampled_dataset_metrics

        # Validate Gaussian network
        bn_results = cross_validate_bn(resampled_dataset, fold_indices)
        results_df["CLG"] = bn_results

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
            nf_model, metrics, result = get_best_normalizing_flow(resampled_dataset, fold_indices, model_type=args.type, parallelize=True)
            results_df["NF"] = metrics

            pickle.dump(nf_model, open(directory_path + "nf_" + str(dataset_id) + ".pkl", "wb"))

            if GRAPHIC:
                # Visualize the convergence of the objective function
                plot_convergence(result)
                plt.savefig(directory_path + "convergence_SD_" + str(dataset_id) + ".png")
                plt.clf()

                # Visualize the convergence of the parameters
                plot_evaluations(result)
                plt.savefig(directory_path + "evaluations_SD_" + str(dataset_id) + ".png")
                plt.clf()

            # Train neural net using the best parameters to get metrics
        print(results_df.drop("params"))
        results_df.to_csv(directory_path + 'data_' + str(dataset_id) + '.csv')
