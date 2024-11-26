import random
import os
import sys

import pickle
from itertools import product

import numpy as np
import pandas as pd

import argparse

import torch
from pymoo.algorithms.moo.nsga2 import NSGA2, binary_tournament
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.selection.tournament import TournamentSelection

from bayesace.algorithms.wachter import WachterCounterfactual
from bayesace.utils import *
from bayesace.algorithms.bayesace_algorithm import BayesACE
from bayesace.algorithms.face import FACE

import time

import multiprocessing as mp

from experiments.utils import setup_experiment, get_constraints, get_counterfactual_from_algorithm


# Constant string
FACE_BASELINE = "face_baseline"
FACE_KDE = "face_kde"
FACE_EPS = "face_eps"
WACHTER = "wachter"
BAYESACE = "bayesace"


def worker(instance, algorithm_path, density_estimator_path, gt_estimator_path, penalty, chunks):
    torch.set_num_threads(1)
    algorithm = pickle.load(open(algorithm_path, 'rb'))
    density_estimator = pickle.load(open(density_estimator_path, 'rb'))
    algorithm.density_estimator = density_estimator
    gt_estimator = pickle.load(open(gt_estimator_path, 'rb'))
    return get_counterfactual_from_algorithm(instance, algorithm, gt_estimator, penalty, chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset_id", nargs='?', default=-1, type=int)
    parser.add_argument('--parallelize', action=argparse.BooleanOptionalAction)
    parser.add_argument('--cv_dir', nargs='?', default='./results/exp_cv_2/', type=str)
    parser.add_argument('--results_dir', nargs='?', default='./results/exp_2/', type=str)
    parser.add_argument('--cv_opt_dir', nargs='?', default='./results/exp_opt2/', type=str)
    parser.add_argument('--multiobjective', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    dataset_id = args.dataset_id
    parallelize = args.parallelize
    verbose = False

    # ALGORITHM PARAMETERS The likelihood parameter is relative. I.e. the likelihood threshold will be the mean logl
    # for that class plus "likelihood_threshold_sigma" sigmas of the logl std
    n_vertices = [0,1]
    penalty = 1
    likelihood_dev_list = [-1, -0.5, 0, 0.5]
    accuracy_threshold_list = [-1, -0.5, 0, 0.5]
    # Number of points for approximating integrals:
    chunks = 20
    # Number of counterfactuals
    n_counterfactuals = 20
    eps = np.inf
    n_train_size = 3000
    n_generations = 1000

    # Activate for multiple objectives
    multi_objective = args.multiobjective

    dummy = False
    if dummy:
        chunks = 5
        n_counterfactuals = 2
        likelihood_dev_list = likelihood_dev_list[:1]
        accuracy_threshold_list = accuracy_threshold_list[:1]
        n_train_size = 10
        n_vertices = n_vertices[:1]
        n_generations = 10
        verbose = True
        parallelize = False

    # Folder for storing the results
    results_dir = args.results_dir + str(dataset_id) + '/'

    random.seed(0)

    # Split the dataset into train and test. Test only contains the n_counterfactuals counterfactuals to be evaluated
    results_cv_dir = args.cv_dir + str(dataset_id) + '/'
    results_opt_cv_dir = args.cv_opt_dir
    df_train, df_counterfactuals, gt_estimator, gt_estimator_path, clg_network, clg_network_path, normalizing_flow, nf_path = setup_experiment(
        results_cv_dir, dataset_id, n_counterfactuals)
    sampling_range, mu_gt, std_gt, mae_gt, std_mae_gt = get_constraints(df_train, df_counterfactuals, gt_estimator, eps = -1)
    df_train = df_train.head(n_train_size)

    # Load the best parameters for the NSGA
    best_params = pd.read_csv(results_opt_cv_dir + "best_params.csv", index_col=0)
    try:
        eta_c = int(best_params.loc[dataset_id, "eta_crossover"])
        eta_m = int(best_params.loc[dataset_id, "eta_mutation"])
        selection_type = best_params.loc[dataset_id, "selection_type"]
    except KeyError:
        eta_c = int(best_params.loc["default", "eta_crossover"])
        eta_m = int(best_params.loc["default", "eta_mutation"])
        selection_type = best_params.loc["default", "selection_type"]
    selection_type = TournamentSelection(
        func_comp=binary_tournament) if selection_type == "tourn" else RandomSelection()

    # Names of the models
    models = [normalizing_flow, clg_network]
    models_str = ["nf", "clg"]
    faces_str = [FACE_BASELINE, FACE_KDE, FACE_EPS, WACHTER]
    algorithm_str_list = faces_str + [BAYESACE + "_" + model_str + "_v" + str(n_vertex) for model_str, n_vertex in
                                      product(models_str, n_vertices)]

    # List for storing the models
    algorithms = []

    # List for storing the density estimator path for each algorithm
    density_estimator_paths = []

    # I want to store the times of building the algorithms
    construction_time_df = pd.DataFrame(columns=["construction_time"],
                                        index=algorithm_str_list)

    t0 = time.time()
    alg = FACE(density_estimator=gt_estimator, features=df_train.columns[:-1], chunks=chunks,
               dataset=df_train.drop("class", axis=1),
               distance_threshold=eps, graph_type="integral", f_tilde=None, seed=0, verbose=verbose,
               log_likelihood_threshold=0.00, accuracy_threshold=0.00, penalty=1, parallelize=parallelize)
    tf = time.time() - t0
    algorithms.append(alg)
    density_estimator_paths.append(gt_estimator_path)
    construction_time_df.loc[FACE_BASELINE, "construction_time"] = tf

    t0 = time.time()
    alg = FACE(density_estimator=normalizing_flow, features=df_train.columns[:-1], chunks=chunks,
               dataset=df_train.drop("class", axis=1),
               distance_threshold=eps, graph_type="kde", f_tilde=None, seed=0, verbose=verbose,
               log_likelihood_threshold=0.00, accuracy_threshold=0.00, penalty=1, parallelize=parallelize)
    tf = time.time() - t0
    algorithms.append(alg)
    density_estimator_paths.append(nf_path)
    construction_time_df.loc[FACE_KDE, "construction_time"] = tf

    t0 = time.time()
    alg = FACE(density_estimator=normalizing_flow, features=df_train.columns[:-1], chunks=chunks,
               dataset=df_train.drop("class", axis=1),
               distance_threshold=eps, graph_type="epsilon", f_tilde="identity", seed=0, verbose=verbose,
               log_likelihood_threshold=0.00, accuracy_threshold=0.00, penalty=1, parallelize=parallelize)
    tf = time.time() - t0
    algorithms.append(alg)
    density_estimator_paths.append(nf_path)
    construction_time_df.loc[FACE_EPS, "construction_time"] = tf

    t0 = time.time()
    alg = WachterCounterfactual(density_estimator=gt_estimator, features=df_train.columns[:-1],
               log_likelihood_threshold=0.00, accuracy_threshold=0.00, dataset=df_train)
    tf = time.time() - t0
    algorithms.append(alg)
    density_estimator_paths.append(gt_estimator_path)
    construction_time_df.loc[WACHTER, "construction_time"] = tf

    # Create as many BayesACE (both with normalizing flow and CLG) as vertices
    for algorithm_str, model, model_path in zip(["nf", "clg"], [normalizing_flow, clg_network], [nf_path, clg_network_path]):
        for n_vertex in n_vertices:
            t0 = time.time()
            alg = BayesACE(density_estimator=model, features=df_train.columns[:-1],
                           n_vertex=n_vertex,
                           accuracy_threshold=0.00, log_likelihood_threshold=0.00,
                           chunks=chunks, penalty=penalty, sampling_range=sampling_range,
                           initialization="guided",
                           seed=0, verbose=verbose, opt_algorithm=NSGA2,
                           opt_algorithm_params={"pop_size": 100, "crossover": SBX(eta=eta_c, prob=0.9),
                                             "mutation": PM(eta=eta_m), "selection": selection_type},
                           generations=n_generations,
                           parallelize=parallelize)
            tf = time.time() - t0
            algorithms.append(alg)
            density_estimator_paths.append(model_path)
            construction_time_df.loc[BAYESACE + "_" + algorithm_str + "_v" + str(n_vertex), "construction_time"] = tf
    # Set parallelism to False for each algorithm
    for alg in algorithms:
        alg.parallelize = False

    # Store the construction time. The dataset need to be identified.
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not dummy :
        construction_time_df.to_csv(results_dir + 'construction_time_data' + str(dataset_id) + '.csv')

    metrics = ["distance", "counterfactual", "time", "time_w_construct", "distance_to_face_baseline", "real_logl", "real_pp"]

    # Folder in case we want to store every result:
    if not os.path.exists(results_dir + 'paths/'):
        os.makedirs(results_dir + 'paths/')

    for likelihood_dev in likelihood_dev_list:
        for accuracy_threshold in accuracy_threshold_list:
            # Result storage
            results_dfs = {i: pd.DataFrame(columns=algorithm_str_list, index=range(n_counterfactuals)) for i in metrics}
            # Name the index column with the dataset id
            for i in metrics:
                results_dfs[i].index.name = dataset_id
            for algorithm, algorithm_str, density_estimator_path in zip(algorithms, algorithm_str_list, density_estimator_paths):
                # Set the proper likelihood  and accuracy thresholds
                algorithm.log_likelihood_threshold = mu_gt + likelihood_dev * std_gt
                algorithm.accuracy_threshold = min(mae_gt + std_mae_gt * accuracy_threshold, 0.99)
                if parallelize :
                    # Pickle the algorithm to avoid I/O in every worker. The file will later be deleted
                    de = algorithm.density_estimator
                    algorithm.density_estimator = None
                    tmp_file_str = results_dir + 'algorithm_' + algorithm_str + '.pkl'
                    pickle.dump(algorithm, open(tmp_file_str, 'wb'))
                    pool = mp.Pool(min(mp.cpu_count()-1, n_counterfactuals))
                    results = pool.starmap(worker, [(df_counterfactuals.iloc[[i]], tmp_file_str, density_estimator_path, gt_estimator_path,
                                                    penalty, chunks) for i in range(n_counterfactuals)])
                    pool.close()
                    pool.join()
                    os.remove(tmp_file_str)
                    algorithm.density_estimator = de
                else :
                    results = []
                    for i in range(n_counterfactuals):
                        instance = df_counterfactuals.iloc[[i]]
                        results = get_counterfactual_from_algorithm(instance, algorithm, gt_estimator, penalty,
                                                                                chunks)
                for i in range(n_counterfactuals):
                    path_length_gt, tf, counterfactual, real_logl, real_pp = results
                    # Check if we are dealing with multiobjective BayesACE by checking the number of outputs
                    if isinstance(path_length_gt, np.ndarray) and len(path_length_gt) > 1:
                        # First we try to select the counterfactuals that surpasses in likelihood and posterior prob
                        # to FACE baseline
                        logl_baseline = -results_dfs["real_logl"].loc[i, FACE_BASELINE]
                        pp_baseline = results_dfs["real_pp"].loc[i, FACE_BASELINE]
                        distance_baseline = results_dfs["distance"].loc[i, FACE_BASELINE]

                        mask: np.ndarray = real_logl > logl_baseline & real_pp > pp_baseline

                        if mask.any():
                            path_length_gt[mask] = 0
                            index = np.argmax(path_length_gt)
                            path_length_gt = path_length_gt[index]
                            counterfactual = counterfactual[index]
                            real_logl = real_logl[index]
                            real_pp = real_pp[index]
                        # If none surpasses it take the one that is closer in terms of likelihood and posterior prob
                        else:
                            # Normalize the logl between 0 and 1 (take percentiles instead of max and min)
                            normalized_real_logl = (real_logl-np.quantile(real_logl,0.05)) / (np.quantile(real_logl,0.95)-np.quantile(real_logl,0.05))
                            normalized_logl_baseline = (logl_baseline-np.quantile(real_logl,0.05)) / (np.quantile(real_logl,0.95)-np.quantile(real_logl,0.05))
                            total_diff = np.abs(normalized_logl_baseline-normalized_real_logl) + np.abs(pp_baseline-real_pp)
                            index = np.argmin(total_diff)
                            path_length_gt = path_length_gt[index]
                            counterfactual = counterfactual[index]
                            real_logl = real_logl[index]
                            real_pp = real_pp[index]
                    results_dfs["distance"].loc[i, algorithm_str] = path_length_gt
                    results_dfs["counterfactual"].loc[i, algorithm_str] = counterfactual
                    results_dfs["time"].loc[i, algorithm_str] = tf
                    results_dfs["time_w_construct"].loc[i, algorithm_str] = tf + construction_time_df.loc[algorithm_str, "construction_time"]
                    results_dfs["real_logl"].loc[i, algorithm_str] = -real_logl
                    results_dfs["real_pp"].loc[i, algorithm_str] = real_pp

            # Prior to save the result, compute the distance between the counterfactual found by the first
            # FACE and the ones found by the other algorithms
            for i in range(n_counterfactuals):
                for algorithm_str in algorithm_str_list:
                    if results_dfs["counterfactual"].loc[i, FACE_BASELINE] is not np.nan and results_dfs["counterfactual"].loc[i, algorithm_str] is not np.nan:
                        results_dfs["distance_to_face_baseline"].loc[i, algorithm_str] = np.linalg.norm(
                            results_dfs["counterfactual"].loc[i, FACE_BASELINE] - results_dfs["counterfactual"].loc[i, algorithm_str])
                    elif results_dfs["counterfactual"].loc[i, FACE_BASELINE] is np.nan and results_dfs["counterfactual"].loc[i, algorithm_str] is not np.nan:
                        results_dfs["distance_to_face_baseline"].loc[i, algorithm_str] = 0
                    else :
                        results_dfs["distance_to_face_baseline"].loc[i, algorithm_str] = np.inf



            if not dummy :
                # Save the results
                for i in metrics:
                    if not os.path.exists(results_dir + i + '/'):
                        os.makedirs(results_dir + i + '/')
                    results_dfs[i].to_csv(
                        results_dir + i + '/likelihood' + str(likelihood_dev) + '_pp' + str(
                            accuracy_threshold) + '.csv')
            else :
                for i in metrics:
                    print(i)
                    print(results_dfs[i].to_string())
                    print()
