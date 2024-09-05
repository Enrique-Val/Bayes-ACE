import random
import os
import sys

import pickle

sys.path.append(os.getcwd())
import argparse

from bayesace.utils import *
from bayesace.algorithms.bayesace_algorithm import BayesACE

if __name__ == "__main__":
    # ALGORITHM PARAMETERS The likelihood parameter is relative. I.e. the likelihood threshold will be the mean logl
    # for that class plus "likelihood_threshold_sigma" sigmas of the logl std
    likelihood_threshold_sigma = 0.5
    accuracy_threshold = 0.9
    n_vertices = 5
    penalties = [1, 5, 10, 15, 20]
    # Number of points for approximating integrals:
    chunks = 10
    # Number of counterfactuals
    n_counterfactuals = 30

    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset_id", nargs='?', default=-1, type=int)
    args = parser.parse_args()

    dataset_id = args.dataset_id

    random.seed(0)

    # Split the dataset into train and test. Test only contains the n_counterfactuals counterfactuals to be evaluated
    df_train = pd.read_csv('./results/exp_cv_2/' + str(dataset_id) + '/resampled_data' + str(dataset_id) + '.csv', index_col=0)
    # Transform the class into a categorical variable
    class_processed = df_train[df_train.columns[-1]].astype('string').astype('category')
    df_train = df_train.drop(df_train.columns[-1], axis=1)
    df_train["class"] = class_processed

    # Get the bounds for the optimization problem. The initial sampling will rely on this, so we call it sampling_range
    xu = df_train.drop(columns=['class']).max().values + 0.0001
    xl = df_train.drop(columns=['class']).min().values - 0.0001
    sampling_range = (xl, xu)

    # Load the pickled gt density estimator from the correct folder
    gt_estimator:ConditionalNF = pickle.load(open('./results/exp_cv_2/' + str(dataset_id) + '/gt_nf_' + str(dataset_id) + '.pkl', 'rb'))
    df_counterfactuals = gt_estimator.sample(n_counterfactuals, seed = 0).to_pandas()
    clg_network = hill_climbing(data=df_train, bn_type="CLG")  # TODO Maybe dill the CLG as well
    normalizing_flow = pickle.load(
        open('./results/exp_cv_2/' + str(dataset_id) + '/nf_' + str(dataset_id) + '.pkl', 'rb'))
    cv_results = pd.read_csv('./results/exp_cv_2/' + str(dataset_id) + '/data_' + str(dataset_id) + '.csv',
                             index_col=0)

    mu_gt = float(cv_results.loc["Logl_mean", "GT_SD"])
    std_gt = float(cv_results.loc["LoglStd_mean", "GT_SD"])

    for density_estimator in [clg_network, normalizing_flow]:
        #np.random.seed(0)
        # np.seterr(divide='ignore')
        for penalty in penalties:
            # Result storage
            times_mat = np.zeros((n_counterfactuals, n_vertices))
            evaluations_mat = np.zeros((n_counterfactuals, n_vertices))
            for i in range(n_counterfactuals):
                instance = df_counterfactuals.iloc[[i]]
                likelihood_threshold = mu_gt + likelihood_threshold_sigma * std_gt
                distances = np.zeros(n_vertices)
                times = np.zeros(n_vertices)
                for n_vertex in range(n_vertices):
                    target_label = get_other_class(df_train["class"].cat.categories, instance["class"].values[0])
                    t0 = time.time()
                    alg = BayesACE(density_estimator=density_estimator, features=df_train.columns[:-1],
                                   n_vertex=n_vertex,
                                   accuracy_threshold=accuracy_threshold, log_likelihood_threshold=likelihood_threshold,
                                   chunks=chunks, penalty=penalty, sampling_range=sampling_range,
                                   initialization="guided",
                                   seed=0, verbose=False, pop_size=100, generations=1000)
                    result = alg.run(instance, target_label=target_label)
                    tf = time.time()-t0
                    # print(result.distance)
                    path_to_compute = path(result.path.values, chunks=chunks)
                    distances[n_vertex] = path_likelihood_length(pd.DataFrame(path_to_compute, columns=instance.columns[:-1]),
                                                                 density_estimator=gt_estimator, penalty=penalty)
                    times[n_vertex] = tf
                    '''
                    plot_path(df_train, result)
                    plt.title("Counterfactual"+str(i)+" and vertex"+str(n_vertex))
                    plt.show()
                    '''
                times_mat[i] = distances
                evaluations_mat[i] = times
            print("Distances mat")
            print(times_mat)
            print()
            print("Evaluations mat")
            print(evaluations_mat)
            print()
            print()

            model_str = "nf" if density_estimator == normalizing_flow else "clg"

            # Check if the target directory exists, if not create it
            if not os.path.exists('./results/exp_1/'+model_str+'/'):
                os.makedirs('./results/exp_1/'+model_str+'/')

            to_ret = pd.DataFrame(data=times_mat, columns=range(n_vertices))
            to_ret.to_csv('./results/exp_1/'+model_str+'/distances_data' + str(dataset_id) + '_penalty' + str(penalty) + '.csv')
    
            to_ret = pd.DataFrame(data=evaluations_mat, columns=range(n_vertices))
            to_ret.to_csv('./results/exp_1/'+model_str+'/time_data' + str(dataset_id) + '_penalty' + str(penalty) + '.csv')

