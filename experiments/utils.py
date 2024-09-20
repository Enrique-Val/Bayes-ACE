import pickle
import time

import pandas as pd
import numpy as np

from bayesace import get_other_class, path, path_likelihood_length
from bayesace.models.conditional_normalizing_flow import ConditionalNF


def setup_experiment(results_cv_dir: str, dataset_id: int, n_counterfactuals: int) :
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
    clg_network = pickle.load(open(clg_network_path, 'rb'))

    # Open the NF
    nf_path = results_cv_dir + 'nf_' + str(dataset_id) + '.pkl'
    normalizing_flow = pickle.load(open(nf_path, 'rb'))

    return df_train, df_counterfactuals, gt_estimator, gt_estimator_path, clg_network, clg_network_path, normalizing_flow, nf_path

def get_constraints(results_cv_dir, df_train, dataset_id) :
    xu = df_train.drop(columns=['class']).max().values + 0.0001
    xl = df_train.drop(columns=['class']).min().values - 0.0001
    sampling_range = (xl, xu)

    # Get the cross-validation results
    cv_results = pd.read_csv(results_cv_dir + 'data_' + str(dataset_id) + '.csv',
                             index_col=0)
    mu_gt = float(cv_results.loc["Logl_mean", "GT_SD"])
    std_gt = float(cv_results.loc["LoglStd_mean", "GT_SD"])

    return sampling_range, mu_gt, std_gt

def get_counterfactual_from_algorithm(instance, algorithm, gt_estimator, penalty, chunks):
    target_label = get_other_class(instance["class"].cat.categories, instance["class"].values[0])
    t0 = time.time()
    result = algorithm.run(instance, target_label=target_label)
    tf = time.time() - t0
    '''
    # Uncomment if all paths want to be stored
    result.path.to_csv(results_dir+'paths/data' + str(dataset_id) + '_likelihood' + str(
    likelihood_dev) + '_acc' + str(accuracy_threshold) +  + algorithm_str + '_counterfactual' + 
    str(i) + '.csv') 
    '''
    # Check first if indeed a counterfactual was found
    if result.counterfactual is None:
        return np.nan, tf, np.nan
    else:
        path_to_compute = path(result.path.values, chunks=chunks)
        path_length_gt = path_likelihood_length(
            pd.DataFrame(path_to_compute, columns=instance.columns[:-1]),
            density_estimator=gt_estimator, penalty=penalty)
        return path_length_gt, tf, result.counterfactual.values

