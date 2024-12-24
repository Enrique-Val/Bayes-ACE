import random
import os
import multiprocessing as mp

import pickle

import argparse

import time
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
def worker(instance, density_estimator_path, gt_estimator_path, penalty, n_vertices, likelihood_threshold,
           accuracy_threshold, chunks, sampling_range, eta_c, eta_m, selection_type):
    torch.set_num_threads(1)
    density_estimator = pickle.load(open(density_estimator_path, 'rb'))
    gt_estimator = pickle.load(open(gt_estimator_path, 'rb'))
    return get_counterfactuals(instance, density_estimator, gt_estimator, penalty, n_vertices,
                               likelihood_threshold, accuracy_threshold, chunks, sampling_range,
                               eta_c, eta_m, selection_type)


def get_counterfactuals(instance, density_estimator, gt_estimator, penalty, n_vertices, likelihood_threshold,
                        accuracy_threshold, chunks,
                        sampling_range, eta_c, eta_m, selection_type):
    distances = np.zeros(n_vertices)
    times = np.zeros(n_vertices)
    for n_vertex in range(n_vertices):
        target_label = get_other_class(instance["class"].cat.categories, instance["class"].values[0])
        t0 = time.time()
        alg = BayesACE(density_estimator=density_estimator, features=instance.columns[:-1],
                       n_vertex=n_vertex+1,
                       accuracy_threshold=accuracy_threshold, log_likelihood_threshold=likelihood_threshold,
                       chunks=chunks, penalty=penalty, sampling_range=sampling_range,
                       initialization="guided", seed=0, verbose=False, opt_algorithm=NSGA2,
                       opt_algorithm_params={"pop_size": 100, "crossover": SBX(eta=eta_c, prob=0.9),
                                             "mutation": PM(eta=eta_m), "selection": selection_type},
                       generations=1000, parallelize=False)
        result = alg.run(instance, target_label=target_label)
        tf = time.time() - t0
        if result.counterfactual is None:
            distances[n_vertex] = np.nan
            times[n_vertex] = tf
        else:
            path_to_compute = path(result.path.values, chunks=chunks)
            pll = path_likelihood_length(
                pd.DataFrame(path_to_compute, columns=instance.columns[:-1]),
                density_estimator=gt_estimator, penalty=penalty)
            if pll == np.inf:
                warnings.warn("Path length over ground truth is infinite for instance " + str(instance.index[0]) + ", "
                              + str(n_vertex) + " vertices, penalty of " + str(penalty) + "and estimator " + str(
                    type(density_estimator)))
            distances[n_vertex] = pll
            times[n_vertex] = tf
    return distances, times


if __name__ == "__main__":
    # ALGORITHM PARAMETERS The likelihood parameter is relative. I.e. the likelihood threshold will be the mean logl
    # for that class plus "likelihood_threshold_sigma" sigmas of the logl std
    likelihood_threshold_sigma = -0.5
    post_prob_threshold_sigma = -0.5
    n_vertices = 3
    penalties = [1, 10, 20]
    # Number of points for approximating integrals:
    chunks = 20
    # Number of counterfactuals
    n_counterfactuals = 20

    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset_id", nargs='?', default=-1, type=int)
    parser.add_argument('--parallelize', action=argparse.BooleanOptionalAction)
    parser.add_argument('--cv_dir', nargs='?', default='./results/exp_cv_2/', type=str)
    parser.add_argument('--results_dir', nargs='?', default='./results/exp_opt_cv/', type=str)
    args = parser.parse_args()

    dataset_id = args.dataset_id
    parallelize = args.parallelize

    random.seed(0)

    results_cv_dir = args.cv_dir + str(dataset_id) + '/'
    results_dir = args.results_dir

    df_train, df_counterfactuals, gt_estimator, gt_estimator_path, clg_network, clg_network_path, normalizing_flow, nf_path = setup_experiment(
        results_cv_dir, dataset_id, n_counterfactuals)
    sampling_range, mu_gt, std_gt, mae_gt, std_mae_gt = get_constraints(df_train, df_counterfactuals, gt_estimator)
    likelihood_threshold = mu_gt + likelihood_threshold_sigma * std_gt
    post_prob_threshold = min(mae_gt + post_prob_threshold_sigma * std_mae_gt, 0.99)
    # Check if there are instances with this threshold in the training set
    check_enough_instances(df_train, gt_estimator, likelihood_threshold, post_prob_threshold)

    density_estimator_path = nf_path
    density_estimator = normalizing_flow

    param_grid = {
        'eta_crossover': [10, 15, 20],  # Example range for crossover eta
        'eta_mutation': [10, 20, 30],  # Example range for mutation eta
        'selection_type': ["tourn", "ran"]  # Example range for selection type
        # Types of selection methods
    }
    param_combinations = ParameterGrid(param_grid)

    results_df = pd.DataFrame(columns=[str(params) for params in param_combinations],
                              index=range(n_counterfactuals * len(penalties) * n_vertices))

    for params in param_combinations:
        eta_c = params['eta_crossover']
        eta_m = params['eta_mutation']
        selection_type = TournamentSelection(func_comp=binary_tournament) if params['selection_type'] == "tourn" else RandomSelection()
        distances = []
        for penalty in penalties:
            print("Running with parameters: " + str(params) + "      Penalty " + str(penalty))
            # Result storage
            distances_mat = np.zeros((n_counterfactuals, n_vertices))
            if parallelize:
                pool = mp.Pool(min(mp.cpu_count() - 1, n_counterfactuals))
                results = pool.starmap(worker,
                                       [(df_counterfactuals.iloc[[i]], gt_estimator_path, gt_estimator_path,
                                         penalty, n_vertices, likelihood_threshold, post_prob_threshold,
                                         chunks, sampling_range, eta_c, eta_m, selection_type) for i in
                                        range(n_counterfactuals)])
                pool.close()
                pool.join()

                for i in range(n_counterfactuals):
                    distances_mat[i], _ = results[i]
            else:
                for i in range(n_counterfactuals):
                    instance = df_counterfactuals.iloc[[i]]
                    distances_mat[i], _ = get_counterfactuals(instance, density_estimator, gt_estimator,
                                                              penalty,
                                                              n_vertices, likelihood_threshold,
                                                              post_prob_threshold,
                                                              chunks, sampling_range, eta_c, eta_m,
                                                              selection_type)
            distances_pen = distances_mat.flatten()
            distances.append(distances_pen)
        distances = np.concatenate(distances)
        results_df[str(params)] = distances

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_df.to_csv(os.path.join(results_dir, 'results_data' + str(dataset_id) + '.csv'))
