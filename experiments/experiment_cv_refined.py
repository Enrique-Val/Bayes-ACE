import os
import random
from itertools import product
from multiprocessing import shared_memory

import torch
from sklearn.model_selection import KFold
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_evaluations
from skopt.space import Real, Integer, Dimension
from skopt.utils import use_named_args

from bayesace.models.bayesian_network_classifier import BayesianNetworkClassifier
from bayesace.models.conditional_normalizing_flow import NanLogProb
from bayesace.models.conditional_nvp import ConditionalNVP
from bayesace.models.conditional_spline import ConditionalSpline
from bayesace.models.conditional_kde import ConditionalKDE
from bayesace.dataset.utils import get_data, preprocess_data, remove_outliers

import pickle

import argparse

from bayesace.utils import *
import multiprocessing as mp

import time


def cross_validate_bn(dataset: pd.DataFrame, kfold_object: KFold, outliers: float = np.inf, training_params: dict = None) -> list | None:
    # Validate Gaussian network
    if training_params is None:
        training_params = {}
    bn_results = []
    for i, (train_index, test_index) in enumerate(kfold_object.split(dataset)):
        bn_results_i = []
        df_train = dataset.iloc[train_index].reset_index(drop=True)
        df_train = remove_outliers(df_train, outliers)
        df_test = dataset.iloc[test_index].reset_index(drop=True)
        t0 = time.time()

        network = BayesianNetworkClassifier(network_type="CLG")
        network.fit(df_train[df_train.columns[:-1]], df_train[df_train.columns[-1]], training_params=training_params)
        time_i = time.time() - t0
        X_test = df_test.drop(network.get_class_var_name(), axis=1)
        y_test = df_test[network.get_class_var_name()]
        tmp = network.logl(X_test, y_test)
        logl_i = tmp.mean()
        logl_std_i = tmp.std()
        bn_results_i.append(logl_i)
        bn_results_i.append(logl_std_i)
        predictions = network.predict_proba(X_test.to_numpy(), output="pandas")
        brier_i = brier_score(y_test.to_numpy(), predictions)
        bn_results_i.append(brier_i)
        auc_i = auc(y_test.to_numpy(), predictions)
        bn_results_i.append(auc_i)
        bn_results_i.append(time_i)
        bn_results.append(bn_results_i)

    bn_results = np.array(bn_results)
    bn_results = list(np.vstack((np.mean(bn_results, axis=0), np.std(bn_results, axis=0))).ravel('F'))
    bn_results.append("BIC")

    return bn_results


#########################
# PARALLELIZATION FUNCS #
#########################

# Convert the dataset to a NumPy array for shared memory usage
def to_numpy_shared(df: pd.DataFrame) -> tuple[shared_memory.SharedMemory, np.ndarray, dict]:
    class_var_name = df.columns[-1]
    unique_values = df[class_var_name].unique()
    ordinal_mapping = {value: idx for idx, value in enumerate(unique_values)}
    # Convert DataFrame to NumPy array
    df_numpy = df.drop(class_var_name, axis=1).to_numpy()
    df_numpy = np.hstack((df_numpy, np.array([ordinal_mapping[value] for value in df[class_var_name]]).reshape(-1, 1)))
    shm = shared_memory.SharedMemory(create=True, size=df_numpy.nbytes)
    shared_array = np.ndarray(df_numpy.shape, dtype=df_numpy.dtype, buffer=shm.buf)
    np.copyto(shared_array, df_numpy)
    return shm, shared_array, ordinal_mapping


def prep_worker(shm_name: str, shape: tuple, dtype: np.dtype, column_names: list, ordinal_mapping: dict,
                i_fold: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_index, test_index = i_fold
    # Reconstruct the DataFrame using the shared memory array
    # Access shared memory by name
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    shared_array = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    # Create a DataFrame from the shared memory array
    df_shared = pd.DataFrame(shared_array, columns=column_names)
    # Recodify the str of the class using the ordinal mapping
    class_var_name = column_names[-1]
    df_shared[class_var_name] = df_shared[class_var_name].apply(
        lambda x: list(ordinal_mapping.keys())[list(ordinal_mapping.values()).index(x)])
    # Create train and test DataFrames
    df_train = df_shared.iloc[train_index].reset_index(drop=True)
    df_test = df_shared.iloc[test_index].reset_index(drop=True)
    return df_train, df_test


def worker_ckde(shm_name: str, shape: tuple, dtype: np.dtype, column_names: list, ordinal_mapping: dict,
                i_fold: tuple, bandwidth=1.0, kernel="gaussian", outliers = np.inf):
    torch.set_num_threads(1)
    df_train, df_test = prep_worker(shm_name, shape, dtype, column_names, ordinal_mapping, i_fold)
    # Proceed with training
    return train_ckde_and_get_results(df_train, df_test, bandwidth=bandwidth, kernel=kernel, outliers=outliers)


################
# CV FUNCTIONS #
################
def get_metrics(model: ConditionalDE, df_test: pd.DataFrame):
    X_test = df_test.drop(model.get_class_var_name(), axis=1)
    y_test = df_test[model.get_class_var_name()]
    logl_data = model.logl(X_test, y_test)
    logl = logl_data.mean()
    logl_std = logl_data.std()
    predictions = model.predict_proba(X_test.to_numpy(), output="pandas")
    brier = brier_score(y_test.to_numpy(), predictions)
    auc_res = auc(y_test.to_numpy(), predictions)
    return {"Logl": logl, "LoglStd": logl_std, "Brier": brier, "AUC": auc_res}


def train_ckde_and_get_results(df_train: pd.DataFrame, df_test: pd.DataFrame, bandwidth: float = 1.0,
                               kernel="gaussian", outliers: float = np.inf):
    # Remove outliers in training
    df_train = remove_outliers(df_train, outliers)
    df_train = df_train.head(10000)
    X_train = df_train[df_train.columns[:-1]]
    y_train = df_train[df_train.columns[-1]]
    t0 = time.time()
    model = ConditionalKDE(bandwidth=bandwidth)
    model.fit(X_train, y_train)
    it_time = time.time() - t0
    metrics = get_metrics(model, df_test)
    metrics["Time"] = it_time
    return metrics


def cross_validate_ckde(dataset: pd.DataFrame, kfold_object: KFold, bandwidth: float = 1.0, kernel="gaussian",
                        outliers: float = np.inf, parallelize: bool = False) -> list | None:
    cv_iter_results = []
    if not parallelize:
        for train_index, test_index in kfold_object.split(dataset):
            df_train = dataset.iloc[train_index].reset_index(drop=True)
            df_test = dataset.iloc[test_index].reset_index(drop=True)
            cv_iter_results.append(
                train_ckde_and_get_results(df_train, df_test, bandwidth=bandwidth, kernel=kernel, outliers=outliers))
    elif parallelize:
        shm, shared_array, ordinal_mapping = to_numpy_shared(dataset)
        column_names = dataset.columns.tolist()
        pool = mp.Pool(min(mp.cpu_count() - 1, kfold_object.n_splits))
        cv_iter_results = pool.starmap(worker_ckde,
                                       [(shm.name, shared_array.shape, shared_array.dtype, column_names,
                                         ordinal_mapping, i_fold, bandwidth, kernel, outliers)
                                        for i_fold in kfold_object.split(dataset)])
        pool.close()
        pool.join()

        shm.close()
        shm.unlink()
    cv_results = {"Logl": [], "LoglStd": [], "Brier": [], "AUC": [], "Time": []}
    for cv_iter_result in cv_iter_results:
        for key in cv_results.keys():
            cv_results[key].append(cv_iter_result[key])

    print("CKDE learned.   Params:", {"bandwidth": bandwidth, "kernel": kernel})
    cv_results_summary = {"Logl_mean": np.mean(cv_results["Logl"]), "Logl_std": np.std(cv_results["Logl"]),
                          "LoglStd_mean": np.mean(cv_results["LoglStd"]), "LoglStd_std": np.std(cv_results["LoglStd"]),
                          "Brier_mean": np.mean(cv_results["Brier"]), "Brier_std": np.std(cv_results["Brier"]),
                          "AUC_mean": np.mean(cv_results["AUC"]), "AUC_std": np.std(cv_results["AUC"]),
                          "Time_mean": np.mean(cv_results["Time"]), "Time_std": np.std(cv_results["Time"])}
    print(cv_results_summary)
    print()
    return [cv_results_summary[i] for i in cv_results_summary.keys()] + [{"bandwidth": bandwidth, "kernel": kernel}]


def grid_search_ckde(dataset: pd.DataFrame, kfold_object: KFold, param_space: dict, previous_best=None, parallelize=False):
    best_bandwidth = None
    best_kernel = None
    best_logl = -np.inf
    if previous_best is not None:
        best_logl = previous_best["logl"]
        best_bandwidth = previous_best["bandwidth"]
        best_kernel = previous_best["kernel"]
    for bandwidth, kernel in product(param_space["bandwidth"], param_space["kernel"]):
        metrics = cross_validate_ckde(dataset, kfold_object, bandwidth=bandwidth, kernel=kernel, parallelize=parallelize)
        # Get the mean_logl
        mean_logl = metrics[0]
        if mean_logl > best_logl:
            best_logl = mean_logl
            best_bandwidth = bandwidth
            best_kernel = kernel
    return best_logl, best_bandwidth, best_kernel


def get_best_ckde(dataset: pd.DataFrame, kfold_object: KFold, param_space: dict = None, parallelize=False):
    # Param space is a grid of parameters. Instead of Bayesian optimization, we will use a grid search
    if param_space is None:
        param_space_gauss = {"bandwidth": np.logspace(-1, 0, num=10),
                             "kernel": ["gaussian"]}
        param_space_linear = {"bandwidth": np.logspace(0, 1, num=10),
                              "kernel": ["epanechnikov", "linear"]}
        best_logl, best_bandwidth, best_kernel = grid_search_ckde(dataset, kfold_object,
                                                                  param_space_gauss, parallelize=parallelize)
        '''_, best_bandwidth, best_kernel = grid_search_ckde(dataset, kf,
                                                          param_space_linear,
                                                          previous_best={
                                                              "logl": best_logl,
                                                              "bandwidth": best_bandwidth,
                                                              "kernel": best_kernel})'''
    else:
        _, best_bandwidth, best_kernel = grid_search_ckde(dataset, kfold_object, param_space)

    # Cross validate again to get the rest of the metrics
    metrics = cross_validate_ckde(dataset, kfold_object, bandwidth=best_bandwidth, kernel=best_kernel)

    # Train once again to return the object
    model = ConditionalKDE(bandwidth=best_bandwidth)
    X = dataset[dataset.columns[:-1]].head(10000)
    y = dataset[dataset.columns[-1]].head(10000)
    model.fit(X, y)
    return model, metrics, best_bandwidth


if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.set_default_dtype(torch.float32)
    t_init = time.time()
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset_id", nargs='?', default=44090, type=int)
    parser.add_argument('--graphics', action=argparse.BooleanOptionalAction)
    parser.add_argument("--type", choices=["NVP", "Spline"], default="NVP")
    parser.add_argument('--n_iter', nargs='?', default=50, type=int)
    parser.add_argument('--parallelize', action=argparse.BooleanOptionalAction)
    parser.add_argument('--dir_name', nargs='?', default="./results/exp_cv_2/", type=str)
    parser.add_argument('--gpu', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    # Hard code some parameters
    # Define the number of folds (K)
    k = 10

    # Define how the preprocessing will be done
    ELIM_OUTL = np.inf
    min_unique_vals = 20
    max_cum_values = 3

    dataset_id = args.dataset_id
    GRAPHIC = args.graphics

    directory_path = os.path.join(args.dir_name, str(dataset_id))
    if not os.path.exists(directory_path):
        # If the directory does not exist, create it
        os.makedirs(directory_path)

    # File naming
    results_file = os.path.join(directory_path, "results_" + str(dataset_id) + ".csv")
    resampled_data_file = os.path.join(directory_path, "resampleddata_" + str(dataset_id) + ".csv")
    clg_pkl = os.path.join(directory_path, "clg_" + str(dataset_id) + ".pkl")
    gt_pkl = os.path.join(directory_path, "gt_" + str(dataset_id) + ".pkl")
    print("Cross validation dataset: ", dataset_id)

    # Set the seed
    random.seed(0)

    # Set the metrics to evaluate
    result_metrics = ["Logl", "LoglStd", "Brier", "AUC", "Time"]

    # Create a k-fold object
    kf = KFold(n_splits=k, shuffle=True, random_state=0)

    # Load the dataset and preprocess it
    dataset_oml = get_data(dataset_id)
    dataset_oml, scaler = preprocess_data(dataset_oml, standardize=True, eliminate_outliers=ELIM_OUTL,
                                          min_unique_vals=min_unique_vals,
                                          max_instances=50000, max_cum_values=max_cum_values)

    d = len(dataset_oml.columns) - 1
    n_instances = dataset_oml.shape[0]

    training_params_bn = {"score" : "bic"}

    # Storage of results
    cartesian_product = list(product(result_metrics, ["_mean", "_std"]))
    # Flattening the list of tuples into a single list
    cartesian_product = [word1 + word2 for word1, word2 in cartesian_product]
    results_df = pd.read_csv(results_file)

    # Validate Gaussian network for preliminary comparisons
    bn_results = cross_validate_bn(dataset_oml, kf, training_params=training_params_bn)
    results_df["CLG_RD"] = bn_results

    # Print results
    print("Bayesian network learned")
    dict_print = {result_metrics[i]: bn_results[i * 2] for i in range(len(result_metrics))}
    print(str(dict_print))
    print()

    # First, learn a KDE serving as ground truth flow and sample new synthetic data
    gt_model, metrics_ckde, result = get_best_ckde(dataset_oml, kf, parallelize=args.parallelize)
    results_df["GT_RD"] = metrics_ckde

    # Save net GT model (torch)
    pickle.dump(gt_model, open(gt_pkl, "wb"))

    resampled_dataset = pd.read_csv(resampled_data_file, index_col=0)
    resampled_dataset[gt_model.get_class_var_name()] = resampled_dataset[gt_model.get_class_var_name()].astype('str').astype('category')

    # Check the metrics of the model given the resampled data
    resampled_dataset_metrics = np.zeros(len(results_df) - 1)
    resampled_X = resampled_dataset.drop(gt_model.get_class_var_name(), axis=1)
    resampled_y = resampled_dataset[gt_model.get_class_var_name()]
    tmp = gt_model.logl(resampled_X, resampled_y)
    resampled_dataset_metrics[0] = tmp.mean()
    resampled_dataset_metrics[2] = tmp.std()
    predictions = gt_model.predict_proba(resampled_X.to_numpy(), output="pandas")
    resampled_dataset_metrics[4] = brier_score(resampled_y.to_numpy(), predictions)
    resampled_dataset_metrics[6] = auc(resampled_y.to_numpy(), predictions)
    resampled_dataset_metrics = list(resampled_dataset_metrics)
    resampled_dataset_metrics.append(results_df["GT_RD"].to_numpy()[-1])
    results_df["GT_SD"] = resampled_dataset_metrics

    # Validate Gaussian network
    bn_results = cross_validate_bn(resampled_dataset, kf, training_params=training_params_bn)
    results_df["CLG"] = bn_results

    # Print results
    print("Bayesian network learned")
    dict_print = {result_metrics[i]: bn_results[i * 2] for i in range(len(result_metrics))}
    print(str(dict_print))
    print()

    # Train a and pickle the Bayesian network
    bn = BayesianNetworkClassifier(network_type="CLG")
    X_resampled = resampled_dataset[resampled_dataset.columns[:-1]]
    y_resampled = resampled_dataset[resampled_dataset.columns[-1]]
    bn.fit(X_resampled, y_resampled, training_params=training_params_bn)
    pickle.dump(bn, open(clg_pkl, "wb"))

    #print(results_df.drop("params"))
    print(results_df.to_string())
    results_df.to_csv(results_file)
