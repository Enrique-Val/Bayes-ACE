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

from bayesace.models.conditional_normalizing_flow import NanLogProb
from bayesace.models.conditional_nvp import ConditionalNVP
from bayesace.models.conditional_spline import ConditionalSpline
from bayesace.models.conditional_kde import ConditionalKDE
from bayesace.models.utils import preprocess_data
from bayesace.dataset.utils import get_data

import pickle

import argparse

from bayesace.utils import *
import multiprocessing as mp

import time

# Define the number of folds (K)
k = 10
steps = 500
n_batches = 20

# Define the number of iterations for Bayesian optimization
default_opt_iter = 100

# Define how the preprocessing will be done
ELIM_OUTL = np.inf
min_unique_vals = 20
max_cum_values = 3

def cross_validate_bn(dataset: pd.DataFrame, kf: KFold, outliers:float = np.inf):
    # Validate Gaussian network
    bn_results = []
    for i, (train_index, test_index) in enumerate(kf.split(dataset)):
        bn_results_i = []
        df_train = dataset.iloc[train_index].reset_index(drop=True)
        means_train = df_train[df_train.columns[:-1]].mean()
        stds_train = df_train[df_train.columns[:-1]].std()
        df_train = df_train[(np.abs((df_train[df_train.columns[:-1]] - means_train) / stds_train) < outliers).all(axis=1)]
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
        bn_results.append(bn_results_i)

    bn_results = np.array(bn_results)
    bn_results_mean = np.mean(bn_results, axis=0)
    bn_results = list(np.vstack((np.mean(bn_results, axis=0), np.std(bn_results, axis=0))).ravel('F'))
    bn_results.append("BIC")

    return bn_results


#########################
# PARALLELIZATION FUNCS #
#########################

# Convert the dataset to a NumPy array for shared memory usage
def to_numpy_shared(df: pd.DataFrame) -> tuple[shared_memory.SharedMemory, np.ndarray, dict]:
    unique_values = df["class"].unique()
    ordinal_mapping = {value: idx for idx, value in enumerate(unique_values)}
    # Convert DataFrame to NumPy array
    df_numpy = df.drop("class", axis=1).to_numpy()
    df_numpy = np.hstack((df_numpy, np.array([ordinal_mapping[value] for value in df["class"]]).reshape(-1, 1)))
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
    df_shared["class"] = df_shared["class"].apply(
        lambda x: list(ordinal_mapping.keys())[list(ordinal_mapping.values()).index(x)])
    # Create train and test DataFrames
    df_train = df_shared.iloc[train_index].reset_index(drop=True)
    df_test = df_shared.iloc[test_index].reset_index(drop=True)
    return df_train, df_test


# Worker function that accesses shared memory
def worker_nf(shm_name: str, shape: tuple, dtype: np.dtype, column_names: list, ordinal_mapping: dict,
              i_fold: tuple, model_type="NVP", nn_params: dict = None, directory_path="./"):
    torch.set_num_threads(1)
    df_train, df_test = prep_worker(shm_name, shape, dtype, column_names, ordinal_mapping, i_fold)
    return train_nf_and_get_results(df_train, df_test, model_type=model_type, nn_params=nn_params,
                                    directory_path=directory_path)


def worker_ckde(shm_name: str, shape: tuple, dtype: np.dtype, column_names: list, ordinal_mapping: dict,
                i_fold: tuple, bandwidth=1.0, kernel="gaussian"):
    torch.set_num_threads(1)
    df_train, df_test = prep_worker(shm_name, shape, dtype, column_names, ordinal_mapping, i_fold)
    # Proceed with training
    return train_ckde_and_get_results(df_train, df_test, bandwidth=bandwidth, kernel=kernel)


################
# CV FUNCTIONS #
################

def train_nf_and_get_results(df_train: pd.DataFrame, df_test: pd.DataFrame, model_type="NVP", nn_params: dict = None,
                             directory_path="./"):
    d = df_train.shape[1] - 1

    # We make a copy, since the hidden units that we specify are RELATIVE to the inputs
    nn_params_copy = nn_params.copy()
    nn_params_copy["hidden_units"] = d * nn_params["hidden_units"]
    outliers = nn_params_copy.pop("outliers", np.inf)

    # Remove outliers in training
    means_train = df_train[df_train.columns[:-1]].mean()
    stds_train = df_train[df_train.columns[:-1]].std()
    df_train = df_train[(np.abs((df_train[df_train.columns[:-1]] - means_train) / stds_train) < outliers).all(axis=1)]

    t0 = time.time()
    model = None
    if model_type == "NVP":
        model = ConditionalNVP(graphics=False)
        model.train(df_train, split_dim=d // 2, model_pth_name=directory_path + "model-" + str(os.getpid()) + ".pkl",
                    **nn_params_copy)
    elif model_type == "Spline":
        model = ConditionalSpline()
        model.train(df_train, **nn_params_copy)
    it_time = time.time() - t0
    logl_data = model.logl(df_test)
    logl = logl_data.mean()
    logl_std = logl_data.std()
    predictions = predict_class(df_test.drop("class", axis=1), model)
    brier = brier_score(df_test["class"].values, predictions)
    auc_res = auc(df_test["class"].values, predictions)
    return {"Logl": np.mean(logl), "LoglStd": np.mean(logl_std), "Brier": np.mean(brier), "AUC": np.mean(auc_res),
            "Time": np.mean(it_time)}


def cross_validate_nf(dataset: pd.DataFrame, kf: KFold, model_type="NVP", parallelize=False, working_dir="./",
                      nn_params: dict = None) -> list | None:
    cv_iter_results = []
    if not parallelize:
        try:
            for train_index, test_index in kf.split(dataset):
                df_train = dataset.iloc[train_index].reset_index(drop=True)
                df_test = dataset.iloc[test_index].reset_index(drop=True)
                cv_iter_results.append(
                    train_nf_and_get_results(df_train, df_test, model_type=model_type, nn_params=nn_params,
                                             directory_path=working_dir))
        except NanLogProb as e:
            cv_iter_results = None
            to_print = ("Error while computing log_prob. Returning a high value for the objective function. "
                        "Consider smoothing data, decreasing the value of the lr or the complexity of the "
                        "network.") + str(nn_params)
            warnings.warn(to_print, RuntimeWarning)
            print()
    elif parallelize:
        shm, shared_array, ordinal_mapping = to_numpy_shared(dataset)
        column_names = dataset.columns.tolist()
        pool = mp.Pool(min(mp.cpu_count() - 1, k))
        try:
            # Use starmap with the shared memory array and other needed parameters
            cv_iter_results = pool.starmap(worker_nf,
                                           [(shm.name, shared_array.shape, shared_array.dtype, column_names,
                                             ordinal_mapping, i_fold, model_type, nn_params,
                                             working_dir)
                                            for i_fold in kf.split(dataset)])
        except NanLogProb as e:
            cv_iter_results = None
            to_print = ("Error while computing log_prob. Returning a high value for the objective function. "
                        "Consider smoothing data, decreasing the value of the lr or the complexity of the "
                        "network.") + str(nn_params)
            warnings.warn(to_print, RuntimeWarning)
            print()

        pool.close()
        pool.join()

        shm.close()
        shm.unlink()
    if cv_iter_results is None:
        return None
    cv_results = {"Logl": [], "LoglStd": [], "Brier": [], "AUC": [], "Time": []}
    for cv_iter_result in cv_iter_results:
        for key in cv_results.keys():
            cv_results[key].append(cv_iter_result[key])
    nn_params_print = nn_params.copy()
    # Drop the working directory and permutations
    nn_params_print.pop("perms_instantiation", None)
    nn_params_print.pop("working_dir", None)

    print(str(nn_params_print), "normalizing flow learned")
    cv_results_summary = {"Logl_mean": np.mean(cv_results["Logl"]), "Logl_std": np.std(cv_results["Logl"]),
                          "LoglStd_mean": np.mean(cv_results["LoglStd"]), "LoglStd_std": np.std(cv_results["LoglStd"]),
                          "Brier_mean": np.mean(cv_results["Brier"]), "Brier_std": np.std(cv_results["Brier"]),
                          "AUC_mean": np.mean(cv_results["AUC"]), "AUC_std": np.std(cv_results["AUC"]),
                          "Time_mean": np.mean(cv_results["Time"]), "Time_std": np.std(cv_results["Time"])}
    print(cv_results_summary)
    print()
    return [cv_results_summary[i] for i in cv_results_summary.keys()] + [nn_params]


def get_best_normalizing_flow(dataset: pd.DataFrame, kf: KFold, n_iter: int = 50, nn_params_fixed: dict = None,
                              model_type="NVP",
                              parallelize: bool = False,
                              param_space: list[Dimension] = None, working_dir="./"):
    # Bayesian optimization

    # List to store the random permutations
    if param_space is None and model_type == "NVP":
        param_space = [
            Real(1e-4, 5e-3, name='lr'),
            Real(1e-4, 1e-2, name='weight_decay'),
            Integer(2, 5, name='hidden_units'),
            Integer(1, 3, name='layers'),
            Integer(1, 8, name='n_flows')
        ]
    elif param_space is None:
        raise ValueError("param_space must be defined for the model type")
    perms_list = []

    # Define the objective function
    @use_named_args(param_space)
    def objective(**params):
        # First, if using NVP, define a random permutation and store it in a global list
        print("Minimize")
        perms_instantiation = None
        if model_type == "NVP":
            perms_instantiation = [torch.randperm(d) for _ in range(params["n_flows"])]
            perms_list.append(perms_instantiation)
            params["perms_instantiation"] = perms_instantiation
        # Apend the params and fixed params
        if nn_params_fixed is not None:
            params.update(nn_params_fixed)
        # Cross validate
        result_cv = cross_validate_nf(dataset, kf, model_type=model_type, parallelize=parallelize,
                                      working_dir=working_dir, nn_params=params)
        if result_cv is None or result_cv[0] < -3e11:
            # If the logl is too low, return a high value for the objective function. This allows to not overpenalize
            # regions of the space
            return 3e11
        return -result_cv[0]  # Assuming we want to maximize loglikelihood

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
    best_params = {param_space[i].name: best_params[i] for i in range(len(param_space))}
    # gt_params_str = str(gt_params)
    print("Best parameters: ", best_params)

    # Add to the best parameters the fixed parameters and the best permutation
    if nn_params_fixed is not None:
        best_params.update(nn_params_fixed)
    best_params["perms_instantiation"] = best_perm

    # Cross validate again to get the rest of the metrics
    metrics = cross_validate_nf(dataset, kf, nn_params=best_params, working_dir=working_dir)

    # Train once again to return the object
    model = None
    if model_type == "NVP":
        model = ConditionalNVP(graphics=False)
        best_params["split_dim"] = d // 2
    elif model_type == "Splines":
        model = ConditionalSpline()

    # Prior to training, we DERELATIVIZE the hidden units
    best_params["hidden_units"] = d * best_params["hidden_units"]
    model.train(dataset, **best_params, model_pth_name=working_dir + "best_model.pkl")
    return model, metrics, result


def train_ckde_and_get_results(df_train: pd.DataFrame, df_test: pd.DataFrame, bandwidth: float = 1.0,
                               kernel="gaussian", outliers: float = np.inf):
    t0 = time.time()
    model = ConditionalKDE(bandwidth=bandwidth, kernel=kernel)
    means_train = df_train[df_train.columns[:-1]].mean()
    stds_train = df_train[df_train.columns[:-1]].std()
    df_train = df_train[(np.abs((df_train[df_train.columns[:-1]] - means_train) / stds_train) < outliers).all(axis=1)]
    model.train(df_train.head(15000))
    it_time = time.time() - t0
    logl_data = model.logl(df_test)
    logl = logl_data.mean()
    logl_std = logl_data.std()
    predictions = predict_class(df_test.drop("class", axis=1), model)
    brier = brier_score(df_test["class"].values, predictions)
    auc_res = auc(df_test["class"].values, predictions)
    return {"Logl": np.mean(logl), "LoglStd": np.mean(logl_std), "Brier": np.mean(brier), "AUC": np.mean(auc_res),
            "Time": np.mean(it_time)}


def cross_validate_ckde(dataset: pd.DataFrame, kf: KFold, bandwidth: float = 1.0, kernel="gaussian",
                        outliers: float = np.inf, parallelize: bool = False) -> list | None:
    cv_iter_results = []
    if not parallelize:
        for train_index, test_index in kf.split(dataset):
            df_train = dataset.iloc[train_index].reset_index(drop=True)
            df_test = dataset.iloc[test_index].reset_index(drop=True)
            cv_iter_results.append(
                train_ckde_and_get_results(df_train, df_test, bandwidth=bandwidth, kernel=kernel, outliers=outliers))
    elif parallelize:
        shm, shared_array, ordinal_mapping = to_numpy_shared(dataset)
        column_names = dataset.columns.tolist()
        pool = mp.Pool(min(mp.cpu_count() - 1, k))
        cv_iter_results = pool.starmap(worker_ckde,
                                       [(shm.name, shared_array.shape, shared_array.dtype, column_names,
                                         ordinal_mapping, i_fold, bandwidth, kernel, outliers)
                                        for i_fold in kf.split(dataset)])
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


def grid_search_ckde(dataset: pd.DataFrame, kf: KFold, param_space: dict, previous_best=None):
    best_bandwidth = None
    best_kernel = None
    best_logl = -np.inf
    if previous_best is not None:
        best_logl = previous_best["logl"]
        best_bandwidth = previous_best["bandwidth"]
        best_kernel = previous_best["kernel"]
    for bandwidth, kernel in product(param_space["bandwidth"], param_space["kernel"]):
        metrics = cross_validate_ckde(dataset, kf, bandwidth=bandwidth, kernel=kernel)
        # Get the mean_logl
        mean_logl = metrics[0]
        if mean_logl > best_logl:
            best_logl = mean_logl
            best_bandwidth = bandwidth
            best_kernel = kernel
    return best_logl, best_bandwidth, best_kernel

def get_best_ckde(dataset: pd.DataFrame, kf: KFold, param_space: dict = None):
    # Param space is a grid of parameters. Instead of Bayesian optimization, we will use a grid search
    if param_space is None:
        param_space_gauss = {"bandwidth": np.logspace(-1, 0, num=10),
                             "kernel": ["gaussian"]}
        param_space_linear = {"bandwidth": np.logspace(0, 1, num=10),
                              "kernel": ["epanechnikov", "linear"]}
        best_logl, best_bandwidth, best_kernel = grid_search_ckde(dataset, kf,
                                                                  param_space_gauss)
        '''_, best_bandwidth, best_kernel = grid_search_ckde(dataset, kf,
                                                          param_space_linear,
                                                          previous_best={
                                                              "logl": best_logl,
                                                              "bandwidth": best_bandwidth,
                                                              "kernel": best_kernel})'''
    else:
        _, best_bandwidth, best_kernel = grid_search_ckde(dataset, kf, param_space)

    # Cross validate again to get the rest of the metrics
    metrics = cross_validate_ckde(dataset, kf, bandwidth=best_bandwidth, kernel=best_kernel)

    # Train once again to return the object
    model = ConditionalKDE(bandwidth=best_bandwidth, kernel=best_kernel)
    model.train(dataset)
    return model, metrics, best_bandwidth


if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.set_default_dtype(torch.float32)
    t_init = time.time()
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset_id", nargs='?', default=44127, type=int)
    parser.add_argument('--graphics', action=argparse.BooleanOptionalAction)
    parser.add_argument("--type", choices=["NVP", "Spline"], default="NVP")
    parser.add_argument('--n_iter', nargs='?', default=default_opt_iter, type=int)
    parser.add_argument('--part', choices=['full', 'rd', 'sd'], default='full')
    parser.add_argument('--parallelize', action=argparse.BooleanOptionalAction)
    parser.add_argument('--dir_name', nargs='?', default="./results/exp_cv_2_kde/", type=str)
    args = parser.parse_args()

    dataset_id = args.dataset_id
    GRAPHIC = args.graphics
    n_iter = args.n_iter
    parallelize = args.parallelize

    # Hard code parameter space
    param_space = [
        Real(1e-4, 5e-3, name='lr'),
        Real(1e-4, 1e-2, name='weight_decay'),
        Integer(2, 16, name='count_bins'),
        Integer(2, 10, name='hidden_units'),
        Integer(1, 5, name='layers'),
        Integer(1, 10, name='n_flows'),
        Real(0.01, 0.3, name='sam_noise', prior='log-uniform')
    ]

    directory_path = args.dir_name + str(dataset_id) + "/"
    if not os.path.exists(directory_path):
        # If the directory does not exist, create it
        os.makedirs(directory_path)

    print("Cross validation dataset: ", dataset_id)

    # Set the seed
    random.seed(0)

    # Set the metrics to evaluate
    result_metrics = ["Logl", "LoglStd", "Brier", "AUC", "Time"]

    # Create a k-fold object
    kf = KFold(n_splits=k, shuffle=True, random_state=0)

    if args.part == 'rd' or args.part == 'full':
        # Load the dataset and preprocess it
        dataset = get_data(dataset_id, standardize=True)
        dataset = preprocess_data(dataset, standardize=True, eliminate_outliers=ELIM_OUTL,
                              min_unique_vals=min_unique_vals,
                              max_instances=50000, max_cum_values=max_cum_values)
        d = len(dataset.columns) - 1
        n_instances = dataset.shape[0]
        batch_size = int((n_instances / n_batches) * 0.8 + 1)

        if args.type == "NVP":
            param_space.pop(2)

        # Storage of results
        cartesian_product = list(product(result_metrics, ["_mean", "_std"]))
        # Flattening the list of tuples into a single list
        cartesian_product = [word1 + word2 for word1, word2 in cartesian_product]
        results_df = pd.DataFrame(
            index=cartesian_product + ["params"])
        results_df.index.name = str(dataset_id)

        # Validate Gaussian network for preliminary comparisons
        bn_results = cross_validate_bn(dataset, kf)
        results_df["CLG_RD"] = bn_results

        # Print results
        print("Bayesian network learned")
        dict_print = {result_metrics[i]: bn_results[i * 2] for i in range(len(result_metrics))}
        print(str(dict_print))
        print()

        # First, learn a KDE serving as grouns truth flow and sample new synthetic data
        gt_model, metrics, result = get_best_ckde(dataset, kf)
        results_df["GT_RD"] = metrics

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

        d = resampled_dataset.shape[1] - 1
        n_instances = resampled_dataset.shape[0]
        batch_size = int((n_instances / n_batches) + 1)
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
        bn_results = cross_validate_bn(resampled_dataset, kf)
        results_df["CLG"] = bn_results

        # Print results
        print("Bayesian network learned")
        dict_print = {result_metrics[i]: bn_results[i * 2] for i in range(len(result_metrics))}
        print(str(dict_print))
        print()

        # Train a and pickle the Bayesian network
        bn = hill_climbing(data=resampled_dataset, bn_type="CLG")
        pickle.dump(bn, open(directory_path + "clg_" + str(dataset_id) + ".pkl", "wb"))

        # Validate normalizing flow with different params. Specify also the fixed params
        nn_params_fixed = {"batch_size": batch_size, "steps": steps, "working_dir": directory_path}

        nf_model, metrics, result = get_best_normalizing_flow(resampled_dataset, kf, model_type=args.type,
                                                              n_iter=n_iter, nn_params_fixed=nn_params_fixed,
                                                              parallelize=parallelize, param_space=param_space)
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
