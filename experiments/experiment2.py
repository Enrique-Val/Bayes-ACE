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


def worker(instance, algorithm_path, gt_estimator_path, penalty, chunks):
    torch.set_num_threads(1)
    algorithm = pickle.load(open(algorithm_path, 'rb'))
    gt_estimator = pickle.load(open(gt_estimator_path, 'rb'))
    return get_counterfactual_from_algorithm(instance, algorithm, gt_estimator, penalty, chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset_id", nargs='?', default=-1, type=int)
    parser.add_argument('--parallelize', action=argparse.BooleanOptionalAction)
    parser.add_argument('--cv_dir', nargs='?', default='./results/exp_cv_2/', type=str)
    parser.add_argument('--results_dir', nargs='?', default='./results/exp_2/', type=str)
    parser.add_argument('--cv_opt_dir', nargs='?', default='./results/exp_opt2/', type=str)
    args = parser.parse_args()

    dataset_id = args.dataset_id
    parallelize = args.parallelize
    verbose = False

    # ALGORITHM PARAMETERS The likelihood parameter is relative. I.e. the likelihood threshold will be the mean logl
    # for that class plus "likelihood_threshold_sigma" sigmas of the logl std
    n_vertices = [0,1]
    penalty = 1
    likelihood_dev_list = [-1, -0.5, 0]
    accuracy_threshold_list = [-1, -0.5, 0]
    # Number of points for approximating integrals:
    chunks = 20
    # Number of counterfactuals
    n_counterfactuals = 20
    eps = np.inf

    # Folder for storing the results
    results_dir = args.results_dir + str(dataset_id) + '/'

    random.seed(0)

    # Split the dataset into train and test. Test only contains the n_counterfactuals counterfactuals to be evaluated
    results_cv_dir = args.cv_dir + str(dataset_id) + '/'
    results_opt_cv_dir = args.cv_opt_dir
    df_train, df_counterfactuals, gt_estimator, gt_estimator_path, clg_network, clg_network_path, normalizing_flow, nf_path = setup_experiment(
        results_cv_dir, dataset_id, n_counterfactuals)
    sampling_range, mu_gt, std_gt, mae_gt, std_mae_gt = get_constraints(df_train, df_counterfactuals, gt_estimator, eps = -1)
    df_train = df_train.head(1000)

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
    faces_str = ["face_baseline", "face_kde", "face_eps", "wachter"]
    algorithm_str_list = faces_str + ["bayesace_" + model_str + "_v" + str(n_vertex) for model_str, n_vertex in
                                      product(models_str, n_vertices)]

    # List for storing the models
    algorithms = []

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
    construction_time_df.loc["face_baseline", "construction_time"] = tf

    t0 = time.time()
    alg = FACE(density_estimator=normalizing_flow, features=df_train.columns[:-1], chunks=chunks,
               dataset=df_train.drop("class", axis=1),
               distance_threshold=eps, graph_type="kde", f_tilde=None, seed=0, verbose=verbose,
               log_likelihood_threshold=0.00, accuracy_threshold=0.00, penalty=1, parallelize=parallelize)
    tf = time.time() - t0
    algorithms.append(alg)
    construction_time_df.loc["face_kde", "construction_time"] = tf

    t0 = time.time()
    alg = FACE(density_estimator=normalizing_flow, features=df_train.columns[:-1], chunks=chunks,
               dataset=df_train.drop("class", axis=1),
               distance_threshold=eps, graph_type="epsilon", f_tilde="identity", seed=0, verbose=verbose,
               log_likelihood_threshold=0.00, accuracy_threshold=0.00, penalty=1, parallelize=parallelize)
    tf = time.time() - t0
    algorithms.append(alg)
    construction_time_df.loc["face_eps", "construction_time"] = tf

    t0 = time.time()
    alg = WachterCounterfactual(density_estimator=clg_network, features=df_train.columns[:-1],
               log_likelihood_threshold=0.00, accuracy_threshold=0.00)
    tf = time.time() - t0
    algorithms.append(alg)
    construction_time_df.loc["wachter", "construction_time"] = tf

    # I need as many BayesACE (both with normalizing flow and CLG) as vertices
    for algorithm_str, model in zip(["nf", "clg"], [normalizing_flow, clg_network]):
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
                           generations=1000,
                           parallelize=parallelize)
            tf = time.time() - t0
            algorithms.append(alg)
            construction_time_df.loc["bayesace_" + algorithm_str + "_v" + str(n_vertex), "construction_time"] = tf
    # Set parallelism to False for each algorithm
    for alg in algorithms:
        alg.parallelize = False

    # Store the construction time. The dataset need to be identified.
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    construction_time_df.to_csv(results_dir + 'construction_time_data' + str(dataset_id) + '.csv')

    metrics = ["distance", "counterfactual", "time", "distance_to_face_baseline"]

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
            for algorithm, algorithm_str in zip(algorithms, algorithm_str_list):
                algorithm.parallelize = False
                # Set the proper likelihood  and accuracy thresholds
                algorithm.log_likelihood_threshold = mu_gt + likelihood_dev * std_gt
                algorithm.accuracy_threshold = min(mae_gt + std_mae_gt * accuracy_threshold, 0.99)
                if parallelize :
                    # Pickle the algorithm to avoid I/O in every worker. The file will later be deleted
                    tmp_file_str = results_dir + 'algorithm_' + algorithm_str + '.pkl'
                    pickle.dump(algorithm, open(tmp_file_str, 'wb'))
                    pool = mp.Pool(min(mp.cpu_count()-1, n_counterfactuals))
                    results = pool.starmap(worker, [(df_counterfactuals.iloc[[i]], tmp_file_str, gt_estimator_path,
                                                    penalty, chunks) for i in range(n_counterfactuals)])
                    pool.close()
                    pool.join()
                    os.remove(tmp_file_str)
                    for i in range(n_counterfactuals):
                        results_dfs["distance"].loc[i, algorithm_str] = results[i][0]
                        results_dfs["counterfactual"].loc[i, algorithm_str] = results[i][2]
                        results_dfs["time"].loc[i, algorithm_str] = results[i][1]

                else :
                    for i in range(n_counterfactuals):
                        instance = df_counterfactuals.iloc[[i]]
                        path_length_gt, tf, counterfactual = get_counterfactual_from_algorithm(instance, algorithm, gt_estimator, penalty,
                                                                                chunks)
                        results_dfs["distance"].loc[i, algorithm_str] = path_length_gt
                        results_dfs["counterfactual"].loc[i, algorithm_str] = counterfactual
                        results_dfs["time"].loc[i, algorithm_str] = tf

            # Prior to save the result, compute the distance between the counterfactual found by the first
            # FACE and the ones found by the other algorithms
            for i in range(n_counterfactuals):
                for algorithm_str in algorithm_str_list:
                    if results_dfs["counterfactual"].loc[i, "face_baseline"] is not np.nan and results_dfs["counterfactual"].loc[i, algorithm_str] is not np.nan:
                        results_dfs["distance_to_face_baseline"].loc[i, algorithm_str] = np.linalg.norm(
                            results_dfs["counterfactual"].loc[i, "face_baseline"] - results_dfs["counterfactual"].loc[i, algorithm_str])
                    else:
                        results_dfs["distance_to_face_baseline"].loc[i, algorithm_str] = 0



            # Save the results
            for i in metrics:
                if not os.path.exists(results_dir + i + '/'):
                    os.makedirs(results_dir + i + '/')
                results_dfs[i].to_csv(
                    results_dir + i + '/likelihood' + str(likelihood_dev) + '_pp' + str(
                        accuracy_threshold) + '.csv')
