import random
import os

import pickle
from itertools import product
import argparse

import pandas as pd
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from bayesace.algorithms.wachter import WachterCounterfactual
from bayesace.utils import *
from bayesace.algorithms.bayesace_algorithm import BayesACE
from bayesace.algorithms.face import FACE

import time

import multiprocessing as mp

from experiments.utils import setup_experiment, get_constraints, get_counterfactual_from_algorithm, get_best_opt_params

# Constant string
FACE_BASELINE = "face_baseline"
FACE_KDE = "face_kde"
FACE_EPS = "face_eps"
WACHTER = "wachter"
BAYESACE = "bayesace"


def worker(instance, algorithm_path, gt_estimator_path, penalty, chunks):
    torch.set_num_threads(1)
    algorithm = pickle.load(open(algorithm_path, 'rb'))
    gt_estimator = pickle.load(open(gt_estimator_path, 'rb'))
    return get_counterfactual_from_algorithm(instance, algorithm, gt_estimator, penalty, chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset_id", nargs='?', default=44120, type=int)
    parser.add_argument('--parallelize', action=argparse.BooleanOptionalAction)
    parser.add_argument('--cv_dir', nargs='?', default='./results/exp_cv_2/', type=str)
    parser.add_argument('--results_dir', nargs='?', default='./results/exp_2/', type=str)
    parser.add_argument('--multiobjective', action=argparse.BooleanOptionalAction)
    parser.add_argument('--penalty', nargs='?', default=1, type=float)
    args = parser.parse_args()

    dataset_id = args.dataset_id
    parallelize = args.parallelize
    verbose = False

    # ALGORITHM PARAMETERS The likelihood parameter is relative. I.e. the likelihood threshold will be the mean logl
    # for that class plus "likelihood_threshold_sigma" sigmas of the logl std
    n_vertices = [0,1,2,3]
    penalty = args.penalty
    likelihood_dev_list = [-1, -0.5, 0]
    post_prob_dev_list = [-0.5, 0]
    # Number of points for approximating integrals:
    chunks = 20
    # Number of counterfactuals
    n_counterfactuals = 10
    eps = np.inf
    n_train_size = 1000
    n_generations = 500

    # Activate for multiple objectives
    multi_objective = args.multiobjective

    dummy = False
    if dummy:
        chunks = 3
        n_counterfactuals = 2
        likelihood_dev_list = likelihood_dev_list[-1:]
        accuracy_threshold_list = post_prob_dev_list[-1:]
        n_train_size = 10
        n_vertices = n_vertices[:1]
        n_generations = 10
        verbose = True
        parallelize = False

    # Folder for storing the results
    results_dir = os.path.join(args.results_dir, str(dataset_id), str(args.penalty))

    random.seed(0)

    # Split the dataset into train and test. Test only contains the n_counterfactuals counterfactuals to be evaluated
    results_cv_dir = os.path.join(args.cv_dir, str(dataset_id))
    results_opt_cv_dir = os.path.join(results_cv_dir, 'opt_results')
    df_train, df_counterfactuals, gt_estimator, gt_estimator_path, clg_network, clg_network_path, normalizing_flow, nf_path = setup_experiment(
        results_cv_dir, dataset_id, n_counterfactuals, seed=42)
    df_total = pd.concat([df_train, df_counterfactuals])
    sampling_range, mu_gt, std_gt, mae_gt, std_mae_gt = get_constraints(df_total, df_total, gt_estimator)
    df_train = df_train.head(n_train_size)

    # Names of the models
    models = [normalizing_flow, clg_network]
    models_str = ["nf", "clg", "gt"]

    # Name of the class variable
    class_var_name = gt_estimator.get_class_var_name()

    # Directory for the algorithms
    algorithm_dir = os.path.join(results_dir, "algorithms")
    if not os.path.exists(algorithm_dir):
        os.makedirs(algorithm_dir)

    # List for storing the models
    algorithms = []
    algorithm_str_list = []
    algorithms_paths = []


    # Check if a certain file exists
    if os.path.exists(os.path.join(results_dir, 'construction_time_' + str(dataset_id) + '.csv')):
        construction_time_df = pd.read_csv(os.path.join(results_dir, 'construction_time_' + str(dataset_id) + '.csv'),
                                             index_col=0)
        # Convert to series
        construction_time_df = construction_time_df.squeeze()
        for alg_name in os.listdir(algorithm_dir):
            algorithms.append(pickle.load(open(os.path.join(algorithm_dir, alg_name), 'rb')))
            algorithm_str_list.append(alg_name.split(".")[0])
            algorithms_paths.append(os.path.join(algorithm_dir, alg_name))

    else :
        # I want to store the times of building the algorithms
        construction_time_list = []

        def add_algorithm(alg, alg_name, tf):
            algorithms.append(alg)
            algorithm_str_list.append(alg_name)
            algorithms_paths.append(os.path.join(algorithm_dir, alg_name + ".pkl"))
            if not dummy:
                pickle.dump(alg, open(os.path.join(algorithm_dir, alg_name + ".pkl"), 'wb'))
            construction_time_list.append(tf)

        for density_estimator, graph_type, alg_name in zip([gt_estimator, normalizing_flow, normalizing_flow],
                                                           ["integral", "kde", "epsilon"],
                                                           [FACE_BASELINE, FACE_KDE, FACE_EPS]):
            t0 = time.time()
            alg = FACE(density_estimator=density_estimator, features=df_train.columns[:-1], chunks=chunks,
                       dataset=df_train.drop(class_var_name, axis=1),
                       distance_threshold=eps, graph_type=graph_type, f_tilde=None, seed=0, verbose=verbose,
                       log_likelihood_threshold=0.00, posterior_probability_threshold=0.00, penalty=penalty,
                       parallelize=parallelize)
            tf = time.time() - t0
            add_algorithm(alg, alg_name, tf)

        t0 = time.time()
        alg = WachterCounterfactual(density_estimator=gt_estimator, features=df_train.columns[:-1],
                                    log_likelihood_threshold=0.00, posterior_probability_threshold=0.00, dataset=df_train)
        tf = time.time() - t0
        add_algorithm(alg, WACHTER, tf)

        # Create as many BayesACE (both with normalizing flow and CLG) as vertices
        for model_str, model, model_path in zip(models_str, [normalizing_flow, clg_network, gt_estimator],
                                                [nf_path, clg_network_path, gt_estimator_path]):
            opt_algorithm_params = get_best_opt_params(model=model_str, dataset_id=dataset_id, dir=args.cv_dir)
            opt_algorithm_params["pop_size"] = 100
            for n_vertex in n_vertices:
                t0 = time.time()
                alg = BayesACE(density_estimator=model, features=df_train.columns[:-1],
                               n_vertices=n_vertex, chunks=chunks,
                               posterior_probability_threshold=0.00, log_likelihood_threshold=0.00,
                               penalty=penalty, sampling_range=sampling_range,
                               initialization="guided",
                               seed=0, verbose=verbose, opt_algorithm=NSGA2,
                               opt_algorithm_params=opt_algorithm_params,
                               generations=n_generations,
                               parallelize=parallelize,
                               multi_objective=multi_objective)
                tf = time.time() - t0
                alg_name = BAYESACE + "_" + model_str + "_v" + str(n_vertex)
                add_algorithm(alg, alg_name, tf)
        # Set parallelism to False for each algorithm
        for alg in algorithms:
            alg.parallelize = False

        # Store the construction time. The dataset need to be identified.
        construction_time_df = pd.Series(construction_time_list, index=algorithm_str_list, name="construction_time")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        if not dummy:
            construction_time_df.to_csv(os.path.join(results_dir, 'construction_time_' + str(dataset_id) + '.csv'))

    metrics = ["distance", "path_l0", "distance_l2", "counterfactual", "time", "time_w_construct",
               "distance_to_face_baseline", "real_logl", "real_pp"]

    # Folder in case we want to store every result:
    if not os.path.exists(results_dir + 'paths/'):
        os.makedirs(results_dir + 'paths/')

    for likelihood_dev, post_prob_dev in zip(likelihood_dev_list, post_prob_dev_list):
        print("Likelihood dev:", likelihood_dev, "    Accuracy threshold:", post_prob_dev)
        # Result storage
        results_dfs = {i: pd.DataFrame(columns=algorithm_str_list, index=range(n_counterfactuals)) for i in metrics}
        # Name the index column with the dataset id
        for i in metrics:
            results_dfs[i].index.name = dataset_id
        # Set the proper likelihood  and accuracy thresholds
        for algorithm, algorithm_str in zip(algorithms, algorithm_str_list):
            algorithm.set_log_likelihood_threshold(mu_gt + likelihood_dev * std_gt)
            algorithm.set_posterior_probability_threshold(min(mae_gt + std_mae_gt * post_prob_dev, 0.99))

        if parallelize:
            pool = mp.Pool(min(mp.cpu_count() - 1, n_counterfactuals*len(algorithms_paths)))
            results = pool.starmap(worker, [
                    (df_counterfactuals.iloc[[i]], tmp_file_str, gt_estimator_path,
                     penalty, chunks) for i,tmp_file_str in product(range(n_counterfactuals), algorithms_paths)])
            pool.close()
            pool.join()
        else:
            results = []
            for i, algorithm in product(range(n_counterfactuals), algorithms):
                instance = df_counterfactuals.iloc[[i]]
                results.append(get_counterfactual_from_algorithm(instance, algorithm, gt_estimator, penalty,
                                                                 chunks))
        for i, (instance_i, algorithm_str) in enumerate(product(range(n_counterfactuals), algorithm_str_list)):
            path_length_gt, path_l0, path_l2, tf, counterfactual, real_logl, real_pp = results[i]
            # Check if we are dealing with multiobjective BayesACE by checking the number of outputs
            if multi_objective and algorithm_str.startswith(BAYESACE) and not counterfactual is None:
                # First, if the no baseline counterfactual was found, then we just return the one with lower distance
                if results_dfs["counterfactual"].loc[instance_i, FACE_BASELINE] is None:
                    index = np.argmin(path_length_gt)
                else:
                    # First we try to select the counterfactuals that surpasses in likelihood and posterior prob
                    # to FACE baseline
                    logl_baseline = results_dfs["real_logl"].loc[instance_i, FACE_BASELINE]
                    pp_baseline = results_dfs["real_pp"].loc[instance_i, FACE_BASELINE]
                    distance_baseline = results_dfs["distance"].loc[instance_i, FACE_BASELINE]
                    mask = np.logical_and(real_logl > logl_baseline, real_pp > pp_baseline)

                    if mask.any():
                        path_length_gt[np.logical_not(mask)] = np.inf
                        index = np.argmin(path_length_gt)
                    # If none surpasses it take the one that is closer in terms of likelihood and posterior prob
                    else:
                        # Return path with minimum distance
                        index = np.argmin(path_length_gt)
                path_length_gt = path_length_gt[index]
                path_l0 = path_l0[index]
                path_l2 = path_l2[index]
                counterfactual = counterfactual[index]
                real_logl = real_logl[index]
                real_pp = real_pp[index]
            results_dfs["distance"].loc[instance_i, algorithm_str] = path_length_gt
            results_dfs["path_l0"].loc[instance_i, algorithm_str] = path_l0
            results_dfs["distance_l2"].loc[instance_i, algorithm_str] = path_l2
            results_dfs["counterfactual"].loc[instance_i, algorithm_str] = counterfactual
            results_dfs["time"].loc[instance_i, algorithm_str] = tf
            results_dfs["time_w_construct"].loc[instance_i, algorithm_str] = tf + construction_time_df[
                algorithm_str]
            results_dfs["real_logl"].loc[instance_i, algorithm_str] = real_logl
            results_dfs["real_pp"].loc[instance_i, algorithm_str] = real_pp

        # Prior to save the result, compute the distance between the counterfactual found by the first
        # FACE and the ones found by the other algorithms
        for i in range(n_counterfactuals):
            for algorithm_str in algorithm_str_list:
                if not results_dfs["counterfactual"].loc[i, FACE_BASELINE] is None and not \
                results_dfs["counterfactual"].loc[i, algorithm_str] is None:
                    results_dfs["distance_to_face_baseline"].loc[i, algorithm_str] = np.linalg.norm(
                        results_dfs["counterfactual"].loc[i, FACE_BASELINE] - results_dfs["counterfactual"].loc[
                            i, algorithm_str])
                elif results_dfs["counterfactual"].loc[i, FACE_BASELINE] is None and not \
                results_dfs["counterfactual"].loc[i, algorithm_str] is None:
                    results_dfs["distance_to_face_baseline"].loc[i, algorithm_str] = 0
                else:
                    results_dfs["distance_to_face_baseline"].loc[i, algorithm_str] = np.inf

        if not dummy:
            file_name = 'likelihood' + str(likelihood_dev) + '_pp' + str(post_prob_dev) + '.csv'
            # Save the results
            for i in metrics:
                metric_path = os.path.join(results_dir, i)
                if not os.path.exists(metric_path):
                    os.makedirs(metric_path)
                results_dfs[i].to_csv(os.path.join(metric_path, file_name))
        else:
            for i in metrics:
                print(i)
                print(results_dfs[i].to_string())
                print()
