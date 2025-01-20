import random
import os
import sys

import pickle
from itertools import product

import pandas as pd

import argparse

import torch
from pymoo.algorithms.moo.nsga2 import NSGA2, binary_tournament
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.selection.tournament import TournamentSelection

from bayesace.utils import *
from bayesace.algorithms.bayesace_algorithm import BayesACE
from bayesace.algorithms.face import FACE

import time

from experiments.utils import get_constraints, setup_experiment, get_counterfactual_from_algorithm, get_best_opt_params
import multiprocessing as mp


def worker(instance, algorithm_path, gt_estimator_path, penalty, chunks):
    torch.set_num_threads(1)
    algorithm = pickle.load(open(algorithm_path, 'rb'))
    gt_estimator = pickle.load(open(gt_estimator_path, 'rb'))
    return get_counterfactual_from_algorithm(instance, algorithm, gt_estimator, penalty, chunks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset_id", nargs='?', default=-1, type=int)
    parser.add_argument('--parallelize', action=argparse.BooleanOptionalAction)
    parser.add_argument('--cv_dir', nargs='?', default='./results/exp_cv_2_patience/', type=str)
    parser.add_argument('--results_dir', nargs='?', default='./results/exp_3/', type=str)
    parser.add_argument('--results_opt_dir', nargs='?', default="./results/exp_opt2/", type=str)
    args = parser.parse_args()

    # ALGORITHM PARAMETERS The likelihood parameter is relative. I.e. the likelihood threshold will be the mean logl
    # for that class plus "likelihood_threshold_sigma" sigmas of the logl std
    n_vertices = [0]
    penalties = [0,1,5,10,15,20]
    epsilons = [0.5,1,2,3]
    likelihood_dev_list = [0,0.5,1]
    post_prob_list = [0.9, 0.8, 0.7]
    # Number of points for approximating integrals:
    chunks = 20
    # Number of counterfactuals
    n_counterfactuals = 20

    # Folder for storing the results
    results_dir = args.results_dir
    results_dir = results_dir + str(args.dataset_id) + '/'
    results_opt_cv_dir = args.results_opt_dir

    dataset_id = args.dataset_id
    parallelize = args.parallelize
    verbose = False

    random.seed(0)

    # Split the dataset into train and test. Test only contains the n_counterfactuals counterfactuals to be evaluated
    results_cv_dir = args.cv_dir + str(dataset_id) + '/'
    df_train, df_counterfactuals, gt_estimator, gt_estimator_path, clg_network, clg_network_path, normalizing_flow, nf_path = setup_experiment(
        results_cv_dir, dataset_id, n_counterfactuals)
    sampling_range, mu_gt, std_gt, mae_gt, std_mae_gt = get_constraints(df_train, df_counterfactuals, gt_estimator)
    df_train = df_train.head(1000)

    # Names of the models
    models = [normalizing_flow, clg_network]
    models_str = ["nf", "clg"]
    faces_str = [face_str+"_eps"+str(eps) for face_str,eps in product(["face_baseline", "face_kde", "face_eps"],epsilons)]
    algorithm_str_list = faces_str + ["bayesace_" + model_str + "_v" + str(n_vertex) + "_pen" + str(penalty) for model_str, n_vertex,penalty in product(models_str, n_vertices, penalties)]

    # Store the name of the class variable
    class_var_name = gt_estimator.get_class_var_name()

    # List for storing the models
    algorithms = []

    # I want to store the times of building the algorithms
    construction_time_df = pd.DataFrame(columns=["construction_time"],
                                        index=algorithm_str_list)
    for eps in epsilons :
        t0 = time.time()
        alg = FACE(density_estimator=gt_estimator, features=df_train.columns[:-1], chunks=chunks,
                   dataset=df_train.drop("class", axis = 1),
                   distance_threshold=eps, graph_type="integral", f_tilde=None, seed=0, verbose=verbose,
                   log_likelihood_threshold=0.00, posterior_probability_threshold=0.00, penalty=1, parallelize=parallelize)
        tf = time.time()-t0
        algorithms.append(alg)
        construction_time_df.loc["face_baseline"+"_eps"+str(eps), "construction_time"] = tf

        t0 = time.time()
        alg = FACE(density_estimator=normalizing_flow, features=df_train.columns[:-1], chunks=chunks,
                   dataset=df_train.drop(class_var_name, axis = 1),
                   distance_threshold=eps, graph_type="kde", f_tilde=None, seed=0, verbose=verbose,
                   log_likelihood_threshold=0.00, posterior_probability_threshold=0.00, penalty=1, parallelize=parallelize)
        tf = time.time()-t0
        algorithms.append(alg)
        construction_time_df.loc["face_kde"+"_eps"+str(eps), "construction_time"] = tf

        t0 = time.time()
        alg = FACE(density_estimator=normalizing_flow, features=df_train.columns[:-1], chunks=chunks,
                   dataset=df_train.drop(class_var_name, axis = 1),
                   distance_threshold=eps, graph_type="epsilon", f_tilde="identity", seed=0, verbose=verbose,
                   log_likelihood_threshold=0.00, posterior_probability_threshold=0.00, penalty=1, parallelize=parallelize)
        tf = time.time()-t0
        algorithms.append(alg)
        construction_time_df.loc["face_eps"+"_eps"+str(eps), "construction_time"] = tf

    # I need as many BayesACE (both with normalizing flow and CLG) as vertices
    for model_str,model in zip(["nf", "clg"], [normalizing_flow, clg_network]):
        opt_algorithm_params = get_best_opt_params(model = model_str, dataset_id=dataset_id, dir=results_opt_cv_dir)
        for n_vertex in n_vertices:
            for penalty in penalties :
                t0 = time.time()
                alg = BayesACE(density_estimator=model, features=df_train.columns[:-1],
                               n_vertices=n_vertex,
                               posterior_probability_threshold=0.00, log_likelihood_threshold=0.00,
                               chunks=chunks, penalty=penalty, sampling_range=sampling_range,
                               initialization="guided",
                               seed=0, verbose=verbose, opt_algorithm=NSGA2,
                               opt_algorithm_params=opt_algorithm_params,
                               generations=1000,
                               parallelize=parallelize)
                tf = time.time()-t0
                algorithms.append(alg)
                construction_time_df.loc["bayesace_" + model_str + "_v" + str(n_vertex)+"_pen"+ str(), "construction_time"] = tf
    # Set parallelism to False for each algorithm
    for alg in algorithms:
        alg.parallelize = False

    # Store the construction time. The dataset need to be identified.
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    construction_time_df.to_csv(results_dir+'construction_time_data' + str(dataset_id) + '.csv')

    metrics = ["distance", "counterfactual", "time"]

    # Folder in case we want to store every result:
    if not os.path.exists(results_dir+'paths/'):
        os.makedirs(results_dir+'paths/')

    for likelihood_dev in likelihood_dev_list:
        for posterior_probability_threshold in post_prob_list:
            # Result storage
            results_dfs = {i: pd.DataFrame(columns=algorithm_str_list, index=range(n_counterfactuals)) for i in metrics}
            for algorithm, algorithm_str in zip(algorithms, algorithm_str_list):
                # Set the proper likelihood  and accuracy thresholds
                algorithm.set_log_likelihood_threshold(mu_gt + likelihood_dev * std_gt)
                algorithm.set_posterior_probability_threshold(posterior_probability_threshold)
                for i in range(n_counterfactuals):
                    instance = df_counterfactuals.iloc[[i]]
                    if parallelize:
                        # Pickle the algorithm to avoid I/O in every worker. The file will later be deleted
                        tmp_file_str = results_dir + 'algorithm_' + algorithm_str + '.pkl'
                        pickle.dump(algorithm, open(tmp_file_str, 'wb'))
                        pool = mp.Pool(min(mp.cpu_count() - 1, n_counterfactuals))
                        results = pool.starmap(worker, [(df_counterfactuals.iloc[[i]], tmp_file_str, gt_estimator_path,
                                                         penalty, chunks) for i in range(n_counterfactuals)])
                        pool.close()
                        pool.join()
                        os.remove(tmp_file_str)
                        for i in range(n_counterfactuals):
                            results_dfs["distance"].loc[i, algorithm_str] = results[i][0]
                            results_dfs["counterfactual"].loc[i, algorithm_str] = results[i][2]
                            results_dfs["time"].loc[i, algorithm_str] = results[i][1]

                    else:
                        for i in range(n_counterfactuals):
                            instance = df_counterfactuals.iloc[[i]]
                            path_length_gt, tf, counterfactual = get_counterfactual_from_algorithm(instance, algorithm,
                                                                                                   gt_estimator,
                                                                                                   penalty,
                                                                                                   chunks)
                            results_dfs["distance"].loc[i, algorithm_str] = path_length_gt
                            results_dfs["counterfactual"].loc[i, algorithm_str] = counterfactual
                            results_dfs["time"].loc[i, algorithm_str] = tf
            # Save the results
            for i in metrics:
                if not os.path.exists(results_dir+i+'/'):
                    os.makedirs(results_dir+i+'/')
                results_dfs[i].to_csv(results_dir+i+'/data' + str(dataset_id) + '_likelihood' + str(likelihood_dev) + '_pp' + str(posterior_probability_threshold) + '.csv')

