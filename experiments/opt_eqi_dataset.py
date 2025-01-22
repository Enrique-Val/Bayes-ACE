import random
import os
import multiprocessing as mp

import pickle

import argparse

import time

import pandas as pd
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2, binary_tournament
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.selection.tournament import TournamentSelection
from sklearn.model_selection import ParameterGrid

from bayesace.utils import *
from bayesace.algorithms.bayesace_algorithm import BayesACE
from experiments.utils import setup_experiment, get_constraints, check_enough_instances


# Worker function for parallelization
def worker(instance: pd.DataFrame, density_estimator_path: str, penalty: int,
           n_vertices: int, ace_params: dict):
    torch.set_num_threads(1)
    density_estimator = pickle.load(open(density_estimator_path, 'rb'))
    return get_counterfactuals(instance, density_estimator, penalty, n_vertices, ace_params)


def get_counterfactuals(instance: pd.DataFrame, density_estimator: ConditionalDE,
                        penalty: int,
                        n_vertices: int, ace_params: dict):
    distances = np.zeros(n_vertices)
    times = np.zeros(n_vertices)
    for n_vertex in range(n_vertices):
        target_label = get_other_class(instance["class"].cat.categories, instance["class"].values[0])
        t0 = time.time()
        print("Vertices:", n_vertex)
        alg = BayesACE(density_estimator=density_estimator, features=instance.columns[:-1],
                       n_vertices=n_vertex + 1,
                       **ace_params,
                       generations=1000, parallelize=False)
        result = alg.run(instance, target_label=target_label)
        tf = time.time() - t0
        distances[n_vertex] = result.distance
        times[n_vertex] = tf
    return distances, times


if __name__ == "__main__":
    # ALGORITHM PARAMETERS The likelihood parameter is relative. I.e. the likelihood threshold will be the mean logl
    # for that class plus "likelihood_threshold_sigma" sigmas of the logl std
    likelihood_threshold_sigma = 0.0
    post_prob_threshold_sigma = 0.0
    n_vertices = 1
    penalties = [1,5]
    # Number of points for approximating integrals:
    chunks = 20
    # Number of counterfactuals
    n_counterfactuals = 20

    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('--parallelize', action=argparse.BooleanOptionalAction)
    parser.add_argument('--dir_name', nargs='?', default="./results/exp_cv_eqi/", type=str)
    parser.add_argument('--model', nargs='?', default="all", choices=["all","bn_restricted",
                                                                      "bn_restricted_lim_arcs","bn_unrestricted"])
    args = parser.parse_args()

    model_str: str = args.model
    parallelize = args.parallelize

    results_dir = args.dir_name + "data_processed/"
    model_dir = args.dir_name + "models/"

    random.seed(0)

    # Load some train data and the different estimators to fine tune a genetic algorithm
    df_train = pd.read_csv(os.path.join(results_dir, "data_train.csv"), index_col=0)
    models: [str, ConditionalDE] = {}
    models_path: [str, ConditionalDE] = {}
    for model in os.listdir(model_dir):
        if ".pkl" in model :
            model_path = os.path.join(model_dir, model)
            with open(model_path, "rb") as f:
                models[model[:-4]] = pickle.load(f)
                models_path[model[:-4]] = model_path

    df_counterfactuals = df_train[int(df_train["class"]) < 5].head(n_counterfactuals)

    sampling_range, mu_gt, std_gt, mae_gt, std_mae_gt = get_constraints(df_train, df_counterfactuals, models["nf"])
    log_likelihood_threshold = mu_gt + likelihood_threshold_sigma * std_gt
    post_prob_threshold = min(mae_gt + post_prob_threshold_sigma * std_mae_gt, 0.99)
    # Check if there are instances with this threshold in the training set
    for model_name, model in models :
        check_enough_instances(df_train, model, log_likelihood_threshold, post_prob_threshold)

    # Reduce the models dict in specified in args
    if args.model != "all" :
        models = {args.model : models[args.model]}
        models_str = {args.model: models_path[args.model]}

    param_grid = {
        'eta_crossover': [10, 15, 20],  # Example range for crossover eta
        'eta_mutation': [10, 20, 30],  # Example range for mutation eta
        'selection_type': ["tourn", "ran"]  # Example range for selection type
        # Types of selection methods
    }
    param_combinations = ParameterGrid(param_grid)

    for model_str in models.keys():
        model_path = models_path[model_str]
        model = models[model_str]

        results_df = pd.DataFrame(columns=[str(params) for params in param_combinations],
                                  index=range(n_counterfactuals * len(penalties) * n_vertices))

        for params in param_combinations:
            print("Running with parameters: " + str(params) + " in model " + model_str)
            # Create dictionary of ace parameters
            ace_params = {"posterior_probability_threshold": post_prob_threshold,
                          "log_likelihood_threshold": log_likelihood_threshold, "chunks": chunks,
                          "sampling_range": sampling_range, "opt_algorithm_params": {
                    "pop_size": 100,
                    "crossover": SBX(eta=params["eta_crossover"]),
                    "mutation": PM(eta=params["eta_mutation"]),
                    "selection": TournamentSelection(func_comp=binary_tournament) if params["selection_type"] == "tourn" else RandomSelection()}}
            distances = []
            for penalty in penalties:
                print("Running with parameters: " + str(params) + "      Penalty " + str(penalty))
                # Result storage
                distances_mat = np.zeros((n_counterfactuals, n_vertices))
                if parallelize:
                    pool = mp.Pool(min(mp.cpu_count() - 1, n_counterfactuals))
                    results = pool.starmap(worker,
                                           [(df_counterfactuals.iloc[[i]], model_path,
                                             penalty, n_vertices, ace_params) for i in
                                            range(n_counterfactuals)])
                    pool.close()
                    pool.join()

                    for i in range(n_counterfactuals):
                        distances_mat[i], _ = results[i]
                else:
                    for i in range(n_counterfactuals):
                        instance = df_counterfactuals.iloc[[i]]
                        distances_mat[i], _ = get_counterfactuals(instance, model,
                                                                  penalty, n_vertices, ace_params)
                distances_pen = distances_mat.flatten()
                distances.append(distances_pen)
            distances = np.concatenate(distances)
            results_df[str(params)] = distances

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        results_df.to_csv(os.path.join(results_dir, 'results_dataEQI_' + model_str + '.csv'))
