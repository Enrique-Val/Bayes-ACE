import random
import os
import multiprocessing as mp

import pickle

import argparse

import torch

from bayesace.utils import *
from bayesace.algorithms.bayesace_algorithm import BayesACE


# Worker function for parallelization
def worker(instance, density_estimator_path, gt_estimator_path, penalty, n_vertices, likelihood_threshold,
           accuracy_threshold, chunks, sampling_range):
    torch.set_num_threads(1)
    density_estimator = pickle.load(open(density_estimator_path, 'rb'))
    gt_estimator = pickle.load(open(gt_estimator_path, 'rb'))
    return get_counterfactuals(instance, density_estimator, gt_estimator, penalty, n_vertices,
                               likelihood_threshold, accuracy_threshold, chunks, sampling_range)


def get_counterfactuals(instance, density_estimator, gt_estimator, penalty, n_vertices, likelihood_threshold,
                        accuracy_threshold, chunks,
                        sampling_range):
    distances = np.zeros(n_vertices)
    times = np.zeros(n_vertices)
    for n_vertex in range(n_vertices):
        target_label = get_other_class(instance["class"].cat.categories, instance["class"].values[0])
        t0 = time.time()
        alg = BayesACE(density_estimator=density_estimator, features=instance.columns[:-1],
                       n_vertex=n_vertex,
                       accuracy_threshold=accuracy_threshold, log_likelihood_threshold=likelihood_threshold,
                       chunks=chunks, penalty=penalty, sampling_range=sampling_range,
                       initialization="guided",
                       seed=0, verbose=False, pop_size=100, generations=1000, parallelize=False)
        result = alg.run(instance, target_label=target_label)
        tf = time.time() - t0
        # print(result.distance)
        path_to_compute = path(result.path.values, chunks=chunks)
        distances[n_vertex] = path_likelihood_length(
            pd.DataFrame(path_to_compute, columns=instance.columns[:-1]),
            density_estimator=gt_estimator, penalty=penalty)
        times[n_vertex] = tf
    return distances, times


if __name__ == "__main__":
    # ALGORITHM PARAMETERS The likelihood parameter is relative. I.e. the likelihood threshold will be the mean logl
    # for that class plus "likelihood_threshold_sigma" sigmas of the logl std
    likelihood_threshold_sigma = 0.0
    accuracy_threshold = 0.9
    n_vertices = 3
    penalties = [1, 5, 10]
    # Number of points for approximating integrals:
    chunks = 10
    # Number of counterfactuals
    n_counterfactuals = 20

    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset_id", nargs='?', default=-1, type=int)
    parser.add_argument('--parallelize', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    dataset_id = args.dataset_id
    parallelize = args.parallelize

    random.seed(0)

    results_cv_dir = './results/exp_cv_2/' + str(dataset_id) + '/'
    results_dir = './results/exp_1/' + str(dataset_id) + '/'

    # Split the dataset into train and test. Test only contains the n_counterfactuals counterfactuals to be evaluated
    df_train = pd.read_csv(results_cv_dir + 'resampled_data' + str(dataset_id) + '.csv',
                           index_col=0)
    # Transform the class into a categorical variable
    class_processed = df_train[df_train.columns[-1]].astype('string').astype('category')
    df_train = df_train.drop(df_train.columns[-1], axis=1)
    df_train["class"] = class_processed

    # Get the bounds for the optimization problem. The initial sampling will rely on this, so we call it sampling_range
    xu = df_train.drop(columns=['class']).max().values + 0.0001
    xl = df_train.drop(columns=['class']).min().values - 0.0001
    sampling_range = (xl, xu)

    # Load the pickled gt density estimator from the correct folder
    gt_estimator_path = results_cv_dir + 'gt_nf_' + str(dataset_id) + '.pkl'
    gt_estimator: ConditionalNF = pickle.load(
        open(gt_estimator_path, 'rb'))

    # Generate a test sample
    df_counterfactuals = gt_estimator.sample(n_counterfactuals, seed=0).to_pandas()

    # Open the Bayesian network (conditional linear Gaussian)
    clg_network_path = results_cv_dir + 'clg_' + str(dataset_id) + '.pkl'
    try:
        clg_network = pickle.load(open(clg_network_path, 'rb'))
    except FileNotFoundError:
        clg_network = hill_climbing(data=df_train, bn_type="CLG")
        pickle.dump(clg_network, open(clg_network_path, 'wb'))

    # Open the NF
    nf_path = results_cv_dir + 'nf_' + str(dataset_id) + '.pkl'
    normalizing_flow = pickle.load(open(nf_path, 'rb'))

    # Get the cross-validation results
    cv_results = pd.read_csv(results_cv_dir + 'data_' + str(dataset_id) + '.csv',
                             index_col=0)

    mu_gt = float(cv_results.loc["Logl_mean", "GT_SD"])
    std_gt = float(cv_results.loc["LoglStd_mean", "GT_SD"])
    likelihood_threshold = mu_gt + likelihood_threshold_sigma * std_gt

    for density_estimator_path,density_estimator in zip([clg_network_path,],[clg_network, normalizing_flow]):
        for penalty in penalties:
            # Result storage
            times_mat = np.zeros((n_counterfactuals, n_vertices))
            evaluations_mat = np.zeros((n_counterfactuals, n_vertices))
            if parallelize:
                pool = mp.Pool(min(mp.cpu_count(), n_counterfactuals))
                results = pool.starmap(worker, [(df_counterfactuals.iloc[[i]], density_estimator_path, gt_estimator_path,
                                                penalty, n_vertices, likelihood_threshold, accuracy_threshold,
                                                chunks, sampling_range) for i in range(n_counterfactuals)])
                pool.close()
                pool.join()

                for i in range(n_counterfactuals):
                    times_mat[i], evaluations_mat[i] = results[i]
            else :
                for i in range(n_counterfactuals):
                    instance = df_counterfactuals.iloc[[i]]
                    times_mat[i], evaluations_mat[i] = get_counterfactuals(instance, density_estimator, gt_estimator,
                                                                           penalty,
                                                                           n_vertices, likelihood_threshold,
                                                                           accuracy_threshold,
                                                                           chunks, sampling_range)

            print("Distances mat")
            print(times_mat)
            print("Evaluations mat")
            print(evaluations_mat)
            print()

            model_str = "NF" if density_estimator == normalizing_flow else "CLG"

            # Check if the target directory exists, if not create it
            if not os.path.exists(results_dir + model_str + '/'):
                os.makedirs(results_dir + model_str + '/')

            to_ret = pd.DataFrame(data=times_mat, columns=range(n_vertices))
            to_ret.to_csv(results_dir + model_str + '/distances_data' + str(dataset_id) +'_model' + model_str + '_penalty' + str(
                penalty) + '.csv')

            to_ret = pd.DataFrame(data=evaluations_mat, columns=range(n_vertices))
            to_ret.to_csv(
                results_dir + model_str + '/time_data' + str(dataset_id) + '_model' + model_str + '_penalty' + str(penalty) + '.csv')
