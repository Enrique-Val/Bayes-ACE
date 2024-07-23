import os
import random
import sys
from itertools import product

from matplotlib import pyplot as plt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_evaluations
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from bayesace.models.conditional_nvp import ConditionalNVP
from bayesace.models.conditional_spline import ConditionalSpline
from bayesace.models.utils import get_data, preprocess_data

import dill

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
k = 10
steps = 1000
batch_size = 512

# Define how the preprocessing will be done
JIT_COEF = 0.1
ELIM_OUTL = True

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

    return bn_results

def cross_validate_nf(dataset, fold_indices=None, model_type="NVP", batch_size=64, lr=None, weight_decay=None, count_bins=None, hidden_units=None,
                      layers=None, split_dim=None,
                      n_flows=None):
    logl = []
    logl_std = []
    brier = []
    auc_list = []
    times = []
    param_dict = None
    if model_type == "NVP":
        param_dict = {"lr": lr, "weight_decay": weight_decay, "split_dim": split_dim, "hidden_u": hidden_units, "layers": layers, "n_flows": n_flows}
    elif model_type == "Spline":
        param_dict = {"lr": lr, "weight_decay": weight_decay, "count_bins": count_bins, "hidden_u": hidden_units, "layers": layers}
    for train_index, test_index in fold_indices:
        df_train = dataset.iloc[train_index].reset_index(drop=True)
        df_test = dataset.iloc[test_index].reset_index(drop=True)
        t0 = time.time()
        model = None
        if model_type == "NVP":
            model = ConditionalNVP(graphics=False)
            model.train(df_train, lr=lr, weight_decay=weight_decay, split_dim=split_dim,
                        hidden_units=hidden_units * (len(df_train.columns) - 1), layers=layers,
                        n_flows=n_flows, steps=steps, batch_size=batch_size)
        elif model_type == "Spline":
            model = ConditionalSpline()
            model.train(df_train, lr=lr, weight_decay=weight_decay, count_bins=count_bins,
                        hidden_units=hidden_units * (len(df_train.columns) - 1), layers=layers,
                        steps=steps, batch_size=batch_size)
        it_time = time.time() - t0
        times.append(it_time)
        tmp = model.logl(df_test)
        logl.append(tmp.mean())
        logl_std.append(tmp.std())
        predictions = predict_class(df_test.drop("class", axis=1), model)
        brier.append(brier_score(df_test["class"].values, predictions))
        auc_list.append(auc(df_test["class"].values, predictions))
    print(str(param_dict), "normalizing flow learned")
    print(str({"Logl": np.mean(logl), "LoglStd":np.mean(logl_std) , "Brier": np.mean(brier), "AUC": np.mean(auc_list), "Time": np.mean(times)}))
    print()
    return np.mean(logl), np.std(logl), np.mean(logl_std), np.std(logl_std), np.mean(brier), np.std(brier), np.mean(auc_list), np.std(auc_list), np.mean(
        times), np.std(times), param_dict

def get_best_normalizing_flow(dataset, fold_indices, model_type = "NVP"):
    # Bayesian optimization
    # Define the objective function
    @use_named_args(param_space)
    def objective(**params):
        result = cross_validate_nf(dataset, fold_indices, model_type=model_type, batch_size=batch_size, **params)
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

    metrics = cross_validate_nf(dataset, fold_indices, batch_size=batch_size, **gt_params)

    # Train once again to return the object
    model = None
    if model_type == "NVP":
        model = ConditionalNVP(graphics=False)
    elif model_type == "Splines":
        model = ConditionalSpline()
    params = {param_space[i].name: best_params[i] for i in range(len(param_space))}
    params["hidden_units"] = d * gt_params["hidden_units"]
    model.train(dataset, **gt_params, steps=steps)

    return model, metrics, result


if __name__ == "__main__":
    t_init = time.time()
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset_id", nargs='?', default=-1, type=int)
    parser.add_argument('--no_graphics', action=argparse.BooleanOptionalAction)
    parser.add_argument("--type", choices=["NVP", "Spline"], default="NVP")
    parser.add_argument('--search', choices=['grid', 'bayesian'], default='bayesian')
    parser.add_argument('--n_iter', nargs='?', default=100, type=int)
    args = parser.parse_args()

    dataset_id = args.dataset_id
    GRAPHIC = not args.no_graphics
    BAYESIAN = args.search == 'bayesian'
    n_iter = args.n_iter

    directory_path = "./results/exp_cv_2/"+str(dataset_id)+"/"
    if not os.path.exists(directory_path):
        # If the directory does not exist, create it
        os.makedirs(directory_path)

    print("Cross validation dataset: ", dataset_id)

    random.seed(0)

    # Load the dataset and preprocess it
    dataset = get_data(dataset_id, standardize=False)
    dataset = preprocess_data(dataset, jit_coef=JIT_COEF, eliminate_outliers=ELIM_OUTL)
    d = len(dataset.columns) - 1

    if args.type == "NVP" :
        param_space[2] = Integer(0, int(d/2), name='split_dim')

    # Get the fold indices
    fold_indices = kfold_indices(dataset, k)

    # Storage of results
    result_metrics = ["Logl", "LoglStd", "Brier", "AUC", "Time"]
    cartesian_product = list(product(result_metrics, ["_mean", "_std"]))
    # Flattening the list of tuples into a single list
    cartesian_product = [word1 + word2 for word1, word2 in cartesian_product]
    results_df = pd.DataFrame(
        index=cartesian_product+["params"])

    # Validate Gaussian network
    bn_results = cross_validate_bn(dataset, fold_indices)
    results_df["CLG_RD"] = bn_results

    # First, learn a normalizing now and sample new synthetic data
    gt_model, metrics, result = get_best_normalizing_flow(dataset, fold_indices, model_type=args.type)
    results_df["GT_RD"] = metrics
    if GRAPHIC:
        # Visualize the convergence of the objective function
        plot_convergence(result)
        plt.savefig(directory_path+"convergence_RD_"+str(dataset_id)+".png")
        plt.clf()

        # Visualize the convergence of the parameters
        plot_evaluations(result)
        plt.savefig(directory_path+"evaluations_RD_"+str(dataset_id)+".png")
        plt.clf()

    dill.dump(gt_model, open(directory_path+"gt_nf_" + str(dataset_id) + ".pkl", "wb"))
    resampled_dataset = gt_model.sample(np.min((len(dataset),100000))).to_pandas()
    resampled_dataset.to_csv(directory_path+"resampled_data"+str(dataset_id)+".csv")

    # New fold indices
    fold_indices = kfold_indices(resampled_dataset, k)

    # Check the metrics of the model given the resampled data
    resampled_dataset_metrics = np.zeros(len(results_df)-1)
    tmp = gt_model.logl(resampled_dataset)
    resampled_dataset_metrics[0] = tmp.mean()
    resampled_dataset_metrics[2] = tmp.std()
    predictions = predict_class(resampled_dataset.drop("class", axis=1), gt_model)
    resampled_dataset_metrics[4] = brier_score(resampled_dataset["class"].values, predictions)
    resampled_dataset_metrics[6] = auc(resampled_dataset["class"].values, predictions)
    resampled_dataset_metrics = list(resampled_dataset_metrics)
    resampled_dataset_metrics.append(metrics[-1])
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
        nf_model, metrics, result = get_best_normalizing_flow(resampled_dataset, fold_indices, model_type=args.type)
        results_df["NF"] = metrics

        dill.dump(nf_model, open(directory_path+"nf_" + str(dataset_id) + ".pkl", "wb"))

        if GRAPHIC:
            # Visualize the convergence of the objective function
            plot_convergence(result)
            plt.savefig(directory_path+"convergence_SD_"+str(dataset_id)+".png")
            plt.clf()

            # Visualize the convergence of the parameters
            plot_evaluations(result)
            plt.savefig(directory_path+"evaluations_SD_"+str(dataset_id)+".png")
            plt.clf()

        # Train neural net using the best parameters to get metrics
    print(results_df.drop("params"))
    results_df.to_csv(directory_path+'data' + str(dataset_id) + '_bis.csv')
