import random
import os
import multiprocessing as mp

import pickle

import argparse
import time

import torch
from pymoo.algorithms.moo.nsga2 import NSGA2

from bayesace.utils import *
from bayesace.algorithms.bayesace_algorithm import BayesACE
from experiments.utils import setup_experiment, get_constraints, check_enough_instances, get_best_opt_params


# Worker function for parallelization
def worker(instance, density_estimator_path, gt_estimator_path, n_vertices, ace_params):
    torch.set_num_threads(1)
    density_estimator = pickle.load(open(density_estimator_path, 'rb'))
    gt_estimator = pickle.load(open(gt_estimator_path, 'rb'))
    return get_counterfactuals(instance, density_estimator, gt_estimator, n_vertices,
                               ace_params)


def get_counterfactuals(instance, density_estimator : ConditionalDE, gt_estimator, n_vertices, ace_params):
    class_var_name = density_estimator.get_class_var_name()
    distances = np.zeros(n_vertices)
    real_distances = np.zeros(n_vertices)
    times = np.zeros(n_vertices)
    for n_vertex in range(n_vertices):
        target_label = get_other_class(instance[class_var_name].cat.categories, instance[class_var_name].values[0])
        t0 = time.time()
        alg = BayesACE(density_estimator=density_estimator, features=instance.columns[:-1],
                       n_vertex=n_vertex, **ace_params)
        result = alg.run(instance, target_label=target_label)
        tf = time.time() - t0
        if result.counterfactual is None:
            distances[n_vertex] = np.nan
            real_distances[n_vertex] = np.nan
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
            distances[n_vertex] = result.distance
            real_distances[n_vertex] = pll
            times[n_vertex] = tf
    return distances, real_distances, times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset_id", nargs='?', default=-1, type=int)
    parser.add_argument('--parallelize', action=argparse.BooleanOptionalAction)
    parser.add_argument('--cv_dir', nargs='?', default='./results/exp_cv_2/', type=str)
    parser.add_argument('--results_dir', nargs='?', default='./results/exp_1/', type=str)
    parser.add_argument('--cv_opt_dir', nargs='?', default='./results/exp_opt2/', type=str)
    args = parser.parse_args()

    dataset_id = args.dataset_id
    parallelize = args.parallelize

    # ALGORITHM PARAMETERS The likelihood parameter is relative. I.e. the likelihood threshold will be the mean logl
    # for that class plus "likelihood_threshold_sigma" sigmas of the logl std
    likelihood_threshold_sigma = -0.5
    post_prob_threshold_sigma = -0.5
    n_vertices = 4
    penalties = [1, 5, 10,15,20]
    # Number of points for approximating integrals:
    chunks = 20
    # Number of counterfactuals
    n_counterfactuals = 20

    # Other hard-coded params
    pop_size = 100
    generations = 1000


    random.seed(0)

    results_cv_dir = os.path.join(args.cv_dir, str(dataset_id))
    results_dir = os.path.join(args.results_dir, str(dataset_id))
    results_opt_cv_dir = os.path.join(results_cv_dir, "opt_results")

    df_train, df_counterfactuals, gt_estimator, gt_estimator_path, clg_network, clg_network_path, normalizing_flow, nf_path = setup_experiment(
        results_cv_dir, dataset_id, n_counterfactuals)
    sampling_range, mu_gt, std_gt, mae_gt, std_mae_gt = get_constraints(df_train, df_counterfactuals, gt_estimator)
    log_likelihood_threshold = mu_gt + likelihood_threshold_sigma * std_gt
    post_prob_threshold = min(mae_gt + post_prob_threshold_sigma * std_mae_gt, 0.99)
    # Check if there are instances with this threshold in the training set
    check_enough_instances(df_train, gt_estimator, log_likelihood_threshold, post_prob_threshold)

    for density_estimator_path, density_estimator, model_str in zip([clg_network_path, nf_path], [clg_network, normalizing_flow], ["clg", "nf"]):
        opt_algorithm_params = get_best_opt_params(model=model_str, dataset_id=dataset_id,
                                                   dir=results_opt_cv_dir)
        opt_algorithm_params["population_size"] = 100
        for penalty in penalties:
            # Result storage
            distances_mat = np.zeros((n_counterfactuals, n_vertices))
            real_distances_mat = np.zeros((n_counterfactuals, n_vertices))
            times_mat = np.zeros((n_counterfactuals, n_vertices))
            ace_params = {"posterior_probability_threshold": post_prob_threshold,
                          "log_likelihood_threshold": log_likelihood_threshold, "chunks": chunks,
                          "sampling_range": sampling_range, "opt_algorithm_params": opt_algorithm_params,
                          "generations": generations, "penalty": penalty}
            if parallelize:
                pool = mp.Pool(min(mp.cpu_count() - 1, n_counterfactuals))
                results = pool.starmap(worker,
                                       [(df_counterfactuals.iloc[[i]], density_estimator_path, gt_estimator_path,
                                         n_vertices, ace_params) for i in
                                        range(n_counterfactuals)])
                pool.close()
                pool.join()

                for i in range(n_counterfactuals):
                    distances_mat[i], real_distances_mat[i], times_mat[i] = results[i]
            else:
                results = []
                for i in range(n_counterfactuals):
                    instance = df_counterfactuals.iloc[[i]]
                    results_i = get_counterfactuals(instance, density_estimator, gt_estimator,
                                                    n_vertices, ace_params)
                    results[i] = results_i
            for i in range(n_counterfactuals):
                distances_mat[i], real_distances_mat[i], times_mat[i] = results[i]

            # Create the file names
            suffix = str(dataset_id) + '_model' + model_str + '_penalty' + str(penalty)
            model_dir = os.path.join(results_dir, model_str)

            # Check if the target directory exists, if not create it
            if not os.path.exists(results_dir + model_str + '/'):
                os.makedirs(results_dir + model_str + '/')

            to_ret = pd.DataFrame(data=distances_mat, columns=range(n_vertices))
            to_ret.to_csv(os.path.join(model_dir, 'distances_data' + suffix + '.csv'))

            to_ret = pd.DataFrame(data=real_distances_mat, columns=range(n_vertices))
            to_ret.to_csv(os.path.join(model_dir, 'real_distances_data' + suffix + '.csv'))

            to_ret = pd.DataFrame(data=times_mat, columns=range(n_vertices))
            to_ret.to_csv(os.path.join(model_dir, 'time_data' + suffix + '.csv'))
