import random
import os
import multiprocessing as mp

import pickle

import argparse

import time
from itertools import product

import torch
from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.selection.tournament import TournamentSelection
from sklearn.model_selection import ParameterGrid

from bayesace.utils import *
from bayesace.algorithms.bayesace_algorithm import BayesACE
from experiments.utils import setup_experiment, get_constraints, check_enough_instances


# Worker function for parallelization
def worker(instance: pd.DataFrame, density_estimator_path: str, penalty: float,
           n_vertices: int, ace_params: dict):
    torch.set_num_threads(1)
    density_estimator = pickle.load(open(density_estimator_path, 'rb'))
    return get_counterfactual(instance, density_estimator, penalty, n_vertices, ace_params)


def get_counterfactual(instance: pd.DataFrame, density_estimator: ConditionalDE,
                        penalty: float,
                        n_vertices: int, ace_params: dict):
    class_var_name = density_estimator.get_class_var_name()
    target_label = get_other_class(instance[class_var_name].cat.categories, instance[class_var_name].values[0])
    t0 = time.time()
    alg = BayesACE(density_estimator=density_estimator, features=instance.columns[:-1],
                   n_vertices=n_vertices,
                   **ace_params, parallelize=False, penalty=penalty)
    result = alg.run(instance, target_label=target_label)
    tf = time.time() - t0
    if result.counterfactual is None:
        return np.inf, tf
    else:
        if np.isnan(result.distance):
            raise ValueError("Distance is not a number")
        return result.distance, tf


if __name__ == "__main__":
    # ALGORITHM PARAMETERS The likelihood parameter is relative. I.e. the likelihood threshold will be the mean logl
    # for that class plus "likelihood_threshold_sigma" sigmas of the logl std
    likelihood_threshold_sigma = -0.5
    post_prob_threshold_sigma = -0.5
    vertices_list = [0, 1, 2, 3]
    penalty_range = (1, 5, 10)
    # Number of points for approximating integrals:
    chunks = 20
    # Number of counterfactuals
    n_counterfactuals = 15

    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset_id", nargs='?', default=44089, type=int)
    parser.add_argument('--model', nargs='?', default='nf', type=str, choices=['nf', 'clg', 'gt'])
    parser.add_argument('--parallelize', action=argparse.BooleanOptionalAction)
    parser.add_argument('--cv_dir', nargs='?', default='./results/exp_cv/', type=str)
    args = parser.parse_args()

    model_str: str = args.model
    dataset_id = args.dataset_id
    parallelize = args.parallelize

    # Some hard-coded parameters
    generations = 1000
    pop_size = 100
    param_grid = {
        'eta_crossover': [10, 15, 20],  # Example range for crossover eta
        'eta_mutation': [10, 20, 30],  # Example range for mutation eta
        'selection_type': ["tourn", "ran"]  # Example range for selection type
        # Types of selection methods
    }


    DUMMY = False
    if DUMMY:
        chunks = 2
        n_counterfactuals = 2
        vertices_list = [0,1]
        generations = 2
        pop_size = 10
        param_grid = {
            'eta_crossover': [10],  # Example range for crossover eta
            'eta_mutation': [10,20],  # Example range for mutation eta
            'selection_type': ["tourn"]  # Example range for selection type
        }

    random.seed(0)

    results_cv_dir = args.cv_dir + str(dataset_id) + '/'
    results_dir = os.path.join(results_cv_dir, "opt_results")

    df_train, df_counterfactuals, gt_estimator, gt_estimator_path, clg_network, clg_network_path, normalizing_flow, nf_path = setup_experiment(
        results_cv_dir, dataset_id, n_counterfactuals)
    sampling_range, mu_gt, std_gt, mae_gt, std_mae_gt = get_constraints(df_train, df_counterfactuals, gt_estimator)
    log_likelihood_threshold = mu_gt + likelihood_threshold_sigma * std_gt
    post_prob_threshold = min(mae_gt + post_prob_threshold_sigma * std_mae_gt, 0.99)
    # Check if there are instances with this threshold in the training set
    check_enough_instances(df_train, gt_estimator, log_likelihood_threshold, post_prob_threshold)
    print("Enough instances found. Running experiment.")

    if model_str == 'nf':
        density_estimator_path = nf_path
        density_estimator = normalizing_flow
    elif model_str == 'clg':
        density_estimator_path = clg_network_path
        density_estimator = clg_network
    elif model_str == 'gt':
        density_estimator_path = gt_estimator_path
        density_estimator = gt_estimator
    else:
        raise ValueError("Model not found")

    param_combinations = ParameterGrid(param_grid)

    # Sample n_counterfactuals random penalties. We only sample values contained in penalty range as a multinomial
    penalties = np.repeat(penalty_range, n_counterfactuals//len(penalty_range)+1)[:n_counterfactuals]

    results_df = pd.DataFrame(columns=[str(params) for params in param_combinations],
                              index=range(n_counterfactuals * len(vertices_list)))

    # Run the experiment
    print("Running with model " + model_str + " and dataset " + str(dataset_id))
    for params in param_combinations:
        print("Running with parameters: " + str(params))
        # Create dictionary of ace parameters
        ace_params = {"posterior_probability_threshold": post_prob_threshold,
                      "log_likelihood_threshold": log_likelihood_threshold, "chunks": chunks,
                      "sampling_range": sampling_range, "opt_algorithm_params": {
                "pop_size": pop_size,
                "crossover": SBX(eta=params["eta_crossover"]),
                "mutation": PM(eta=params["eta_mutation"]),
                "selection": TournamentSelection(func_comp=binary_tournament) if params["selection_type"] == "tourn" else RandomSelection()},
                      "generations": generations}

        if parallelize:
            pool = mp.Pool(min(mp.cpu_count() - 1, n_counterfactuals*len(vertices_list)))
            results = pool.starmap(worker,
                                   [(df_counterfactuals.iloc[[i]], gt_estimator_path,
                                     penalties[i], n_vertices, ace_params) for i,n_vertices in
                                    product(range(n_counterfactuals), vertices_list)])
            pool.close()
            pool.join()
        else:
            results = []
            for i,n_vertices in product(range(n_counterfactuals), vertices_list):
                instance = df_counterfactuals.iloc[[i]]
                results.append(get_counterfactual(instance, density_estimator,
                                                  penalties[i], n_vertices, ace_params))

        results_df[str(params)] = list(zip(*results))[0]

    if not DUMMY:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        results_df.to_csv(os.path.join(results_dir, 'results_data' + str(dataset_id) + '_' + model_str + '.csv'))
    else:
        print("Results")
        print(results_df)
