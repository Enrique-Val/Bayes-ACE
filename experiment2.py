import random
import os
import sys

import pickle
from itertools import product

sys.path.append(os.getcwd())
import argparse

from bayesace.utils import *
from bayesace.algorithms.bayesace_algorithm import BayesACE
from bayesace.algorithms.face import FACE

import time


if __name__ == "__main__":
    # ALGORITHM PARAMETERS The likelihood parameter is relative. I.e. the likelihood threshold will be the mean logl
    # for that class plus "likelihood_threshold_sigma" sigmas of the logl std
    n_vertices = [0]
    penalty = 1
    likelihood_dev_list = [0,0.5,1]
    accuracy_threshold_list = [0.9,0.8,0.7]
    # Number of points for approximating integrals:
    chunks = 10
    # Number of counterfactuals
    n_counterfactuals = 30

    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset_id", nargs='?', default=-1, type=int)
    parser.add_argument('--parallelize', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    dataset_id = args.dataset_id
    parallelize = args.parallelize
    verbose = False

    random.seed(0)

    # Split the dataset into train and test. Test only contains the n_counterfactuals counterfactuals to be evaluated
    df_train = pd.read_csv('./results/exp_cv_2/' + str(dataset_id) + '/resampled_data' + str(dataset_id) + '.csv',
                           index_col=0)
    # Transform the class into a categorical variable
    class_processed = df_train[df_train.columns[-1]].astype('string').astype('category')
    df_train = df_train.drop(df_train.columns[-1], axis=1)
    df_train["class"] = class_processed
    df_train = df_train.head(10000)

    # Get the bounds for the optimization problem. The initial sampling will rely on this, so we call it sampling_range
    xu = df_train.drop(columns=['class']).max().values + 0.0001
    xl = df_train.drop(columns=['class']).min().values - 0.0001
    sampling_range = (xl, xu)

    # Load the pickled gt density estimator from the correct folder
    gt_estimator = pickle.load(
        open('./results/exp_cv_2/' + str(dataset_id) + '/gt_nf_' + str(dataset_id) + '.pkl', 'rb'))
    df_counterfactuals = gt_estimator.sample(n_counterfactuals, seed=0).to_pandas()
    clg_network = hill_climbing(data=df_train, bn_type="CLG")
    normalizing_flow = pickle.load(
        open('./results/exp_cv_2/' + str(dataset_id) + '/nf_' + str(dataset_id) + '.pkl', 'rb'))
    cv_results = pd.read_csv('./results/exp_cv_2/' + str(dataset_id) + '/data' + str(dataset_id) + '_bis.csv',
                             index_col=0)

    mu_gt = float(cv_results.loc["Logl_mean", "GT_SD"])
    std_gt = float(cv_results.loc["LoglStd_mean", "GT_SD"])

    # Names of the models
    model_str_list = ["face_baseline", "face_kde", "face_eps"] + ["bayesace_nf_v" + str(n_vertex) for n_vertex in n_vertices] + ["bayesace_clg_v" + str(n_vertex) for n_vertex in n_vertices]

    # List for storing the models
    algorithms = []

    # I want to store the times of building the algorithms
    construction_time_df = pd.DataFrame(columns=["construction_time"],
                                        index=model_str_list)

    t0 = time.time()
    alg = FACE(density_estimator=gt_estimator, features=df_train.columns[:-1], chunks=chunks,
                         dataset=df_train.drop("class", axis = 1),
                         distance_threshold=5, graph_type="integral", f_tilde=None, seed=0, verbose=verbose,
                         likelihood_threshold=0.00, accuracy_threshold=0.00, penalty=1, parallelize=parallelize)
    tf = time.time()-t0
    algorithms.append(alg)
    construction_time_df.loc["face_baseline", "construction_time"] = tf

    t0 = time.time()
    alg = FACE(density_estimator=normalizing_flow, features=df_train.columns[:-1], chunks=chunks,
                    dataset=df_train.drop("class", axis = 1),
                    distance_threshold=5, graph_type="kde", f_tilde=None, seed=0, verbose=verbose,
                    likelihood_threshold=0.00, accuracy_threshold=0.00, penalty=1,parallelize=parallelize)
    tf = time.time()-t0
    algorithms.append(alg)
    construction_time_df.loc["face_kde", "construction_time"] = tf

    t0 = time.time()
    alg = FACE(density_estimator=normalizing_flow, features=df_train.columns[:-1], chunks=chunks,
                    dataset=df_train.drop("class", axis = 1),
                    distance_threshold=5, graph_type="epsilon", f_tilde="identity", seed=0, verbose=verbose,
                    likelihood_threshold=0.00, accuracy_threshold=0.00, penalty=1,parallelize=parallelize)
    tf = time.time()-t0
    algorithms.append(alg)
    construction_time_df.loc["face_eps", "construction_time"] = tf

    # I need as many BayesACE (both with normalizing flow and CLG) as vertices
    for n_vertex in n_vertices:
        t0 = time.time()
        alg = BayesACE(density_estimator=normalizing_flow, features=df_train.columns[:-1],
                               n_vertex=n_vertex,
                               accuracy_threshold=0.00, likelihood_threshold=0.00,
                               chunks=chunks, penalty=penalty, sampling_range=sampling_range,
                               initialization="default",
                               seed=0, verbose=verbose, pop_size=100, parallelize=parallelize)
        tf = time.time()-t0
        algorithms.append(alg)
        construction_time_df.loc["bayesace_nf_v", "construction_time"] = tf

    for n_vertex in n_vertices:
        t0 = time.time()
        alg = BayesACE(density_estimator=clg_network, features=df_train.columns[:-1],
                                n_vertex=n_vertex,
                                accuracy_threshold=0.00, likelihood_threshold=0.00,
                                chunks=chunks, penalty=penalty, sampling_range=sampling_range,
                                initialization="default",
                                seed=0, verbose=verbose, pop_size=100,parallelize=parallelize)
        tf = time.time()-t0
        algorithms.append(alg)
        construction_time_df.loc["bayesace_clg_v", "construction_time"] = tf

    # Store the construction time. The dataset need to be identified.
    if not os.path.exists('./results/exp_2/'):
        os.makedirs('./results/exp_2/')
    construction_time_df.to_csv('./results/exp_2/construction_time_data' + str(dataset_id) + '.csv')

    metrics = ["distance", "time"]
    results_cols = [metric + "_" + model_str for metric, model_str in product(metrics, model_str_list)]

    for likelihood_dev in likelihood_dev_list:
        for accuracy_threshold in accuracy_threshold_list:
            # Result storage
            results_df = pd.DataFrame(columns=results_cols, index=range(n_counterfactuals))
            for algorithm, model_str in zip(algorithms, model_str_list):
                # Set the proper likelihood  and accuracy thresholds
                algorithm.likelihood_threshold = np.e ** (mu_gt + likelihood_dev * std_gt)
                algorithm.accuracy_threshold = accuracy_threshold
                for i in range(n_counterfactuals):
                    instance = df_counterfactuals.iloc[[i]]
                    t0 = time.time()
                    result = algorithm.run(instance, target_label="b" if (instance["class"] != "b").all() else "a")
                    tf = time.time()-t0
                    # TODO check if indeed a counterfactual was found
                    path_to_compute = path(result.path.values, chunks=chunks)
                    path_length_gt = path_likelihood_length(
                        pd.DataFrame(path_to_compute, columns=instance.columns[:-1]),
                        density_estimator=gt_estimator, penalty=penalty)
                    results_df.loc[i, "distance_" + model_str] = path_length_gt
                    results_df.loc[i, "time_" + model_str] = tf
                    plot_path(df_train, result)
                    plt.title("Counterfactual" + str(i) + " model" + model_str)
                    plt.show()

            # Save the results
            if not os.path.exists('./results/exp_2/'):
                os.makedirs('./results/exp_2/')
            results_df.to_csv('./results/exp_2/results_data' + str(dataset_id) + '_likelihood' + str(likelihood_dev) + '_acc' + str(accuracy_threshold) + '.csv')


