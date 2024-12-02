import os.path
import pickle
import time

import pandas as pd
import numpy as np
import torch

from bayesace import get_other_class, path, path_likelihood_length, hill_climbing, log_likelihood, \
    posterior_probability, total_l0_path
from bayesace.models.conditional_normalizing_flow import ConditionalNF

import pandas as pd
import pybnesian as pb

import platform


def setup_experiment(results_cv_dir: str, dataset_id: int, n_counterfactuals: int) -> tuple[
    pd.DataFrame, pd.DataFrame, ConditionalNF, str, pb.CLGNetwork, str, ConditionalNF, str]:
    # Split the dataset into train and test. Test only contains the n_counterfactuals counterfactuals to be evaluated
    df_train = pd.read_csv(results_cv_dir + 'resampled_data' + str(dataset_id) + '.csv',
                           index_col=0)
    # Transform the class into a categorical variable
    class_processed = df_train[df_train.columns[-1]].astype('string').astype('category')
    df_train = df_train.drop(df_train.columns[-1], axis=1)
    df_train["class"] = class_processed

    # Load the pickled gt density estimator from the correct folder
    gt_estimator_path = results_cv_dir + 'gt_nf_' + str(dataset_id) + '.pkl'
    gt_estimator: ConditionalNF = pickle.load(
        open(gt_estimator_path, 'rb'))

    # Generate a test sample
    torch.manual_seed(0)
    df_counterfactuals = gt_estimator.sample(n_counterfactuals, seed=0).to_pandas()

    # Open the Bayesian network (conditional linear Gaussian)
    clg_network_path = results_cv_dir + 'clg_' + str(dataset_id) + '.pkl'
    try:
        clg_network = pickle.load(open(clg_network_path, 'rb'))
    except:
        clg_network = hill_climbing(df_train, seed=0, bn_type="CLG")
        pickle.dump(clg_network, open(clg_network_path, 'wb'))

    # Open the NF
    nf_path = results_cv_dir + 'nf_' + str(dataset_id) + '.pkl'
    normalizing_flow = pickle.load(open(nf_path, 'rb'))

    # Name the index column
    df_train.index.name = dataset_id
    df_counterfactuals.index.name = dataset_id

    return df_train, df_counterfactuals, gt_estimator, gt_estimator_path, clg_network, clg_network_path, normalizing_flow, nf_path


def get_constraints(df_train, df_counterfactuals, gt_estimator: ConditionalNF, eps=0.01):
    assert eps >= 0, "Epsilon must be greater or equal to 0"
    df_total = pd.concat([df_train, df_counterfactuals]).reset_index(drop=True)
    xl = df_total.drop(columns=['class']).min().values - eps
    xu = df_total.drop(columns=['class']).max().values + eps
    sampling_range = (xl, xu)

    logl_train_with_class = gt_estimator.logl(df_train)
    logl_train_without_class = gt_estimator.log_likelihood(df_train)
    post_prob_train = np.exp(logl_train_with_class - logl_train_without_class)

    return sampling_range, logl_train_without_class.mean(), logl_train_without_class.std(), post_prob_train.mean(), post_prob_train.std()


def check_enough_instances(df_train, gt_estimator, likelihood_threshold, post_prob_threshold, min_instances=50):
    logl_train_with_class = gt_estimator.logl(df_train)
    logl_train_without_class = gt_estimator.log_likelihood(df_train)
    post_prob_train = np.exp(logl_train_with_class - logl_train_without_class)
    is_plausible = logl_train_without_class > likelihood_threshold
    is_accurate = post_prob_train > post_prob_threshold
    if (is_accurate & is_plausible).sum() < min_instances:
        print("There are not enough instances in the training set that are both accurate and plausible")
        raise Exception("Not enough instances")


def get_counterfactual_from_algorithm(instance: pd.DataFrame, algorithm, gt_estimator, penalty, chunks, l0_epsilon=0.1):
    print("Instance",instance.index[0])
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
    # Check first if the algorithm is multiobjective (i.e., a list of counterfactuals is returned)
    if isinstance(result, list):
        cfx_array = np.empty(shape=(len(result), len(instance.columns) - 1))
        path_lengths_gt = np.zeros(shape=len(result))
        path_l0 = np.zeros(shape=len(result))
        for i,_ in enumerate(result):
            path_to_compute = path(result[i].path.values, chunks=chunks)
            path_length_gt = path_likelihood_length(
                pd.DataFrame(path_to_compute, columns=instance.columns[:-1]),
                density_estimator=gt_estimator, penalty=penalty)
            path_lengths_gt[i] = path_length_gt
            path_l0[i] = total_l0_path(result[i].path.values, l0_epsilon)
            cfx_array[i] = result[i].counterfactual.values
        cfx_df = pd.DataFrame(cfx_array, columns=instance.columns[:-1])
        real_logl = log_likelihood(cfx_df, gt_estimator)
        real_pp = posterior_probability(cfx_df, target_label, gt_estimator)
        path_l2 = np.linalg.norm(cfx_array - instance.drop(columns="class").values.flatten(), axis=1)
        return path_lengths_gt, path_l0, path_l2, tf, cfx_array, real_logl, real_pp
    # Check if indeed a counterfactual was found
    elif result.counterfactual is None:
        print("Counterfactual for:", instance.index[0], "not found")
        return np.nan, np.inf, np.inf, tf, np.nan, -np.inf, 0
    else:
        path_to_compute = path(result.path.values, chunks=chunks)
        path_length_gt = path_likelihood_length(
            pd.DataFrame(path_to_compute, columns=instance.columns[:-1]),
            density_estimator=gt_estimator, penalty=penalty)
        cfx_df = pd.DataFrame([result.counterfactual.values], columns=instance.columns[:-1])
        real_logl = log_likelihood(cfx_df, gt_estimator)
        real_pp = posterior_probability(cfx_df, target_label , gt_estimator)
        path_l2 = np.linalg.norm(result.counterfactual.values - instance.drop(columns="class").values.flatten())
        path_l0 = total_l0_path(result.path.values, l0_epsilon)
        print("Counterfactual:", instance.index[0], "    Distance", path_length_gt)
        return path_length_gt, path_l0, path_l2, tf, result.counterfactual.values, real_logl, real_pp


def friedman_posthoc(data, correct = "bergmann") -> dict[str, pd.DataFrame | pd.Series]:
    '''
    Perform the Friedman test and the Bermann-Hommel post-hoc test using the scmamp package in R

    Parameters
    ----------
    data : pandas.DataFrame
        A pandas DataFrame where each column is a different outcome to test and each row is a different instance.
    correct : str
        String indicating the correction method to use for the p-values. The possible values are: "shaffer", "bergmann",
         "holland", "finner", "rom" and "li"

    Returns
    -------
    dict
        A dictionary containing the summary statistics of the post-hoc test. The dictionary contains the following keys:
        - "summary": A pandas Series containing the summary statistics of the post-hoc test.
        - "p_values": A pandas DataFrame containing the p-values of the Friedman test.
        - "p_adjusted": A pandas DataFrame containing the adjusted p-values of the Bergmann-Hommel post-hoc test.
    '''

    from rpy2.robjects import pandas2ri
    import rpy2.robjects.packages as rpackages

    # Activate the automatic conversion of pandas objects to R data frames
    pandas2ri.activate()

    # Import the scmamp package from R
    if platform.system() == 'Windows':
        r_lib_path = os.path.expanduser('~/AppData/Local/R/win-library/4.3').replace("\\", "/")
    else:
        r_lib_path = os.path.expanduser('~/R/x86_64-pc-linux-gnu-library/4.4')
    scmamp = rpackages.importr('scmamp', lib_loc=r_lib_path)

    # Convert the Pandas DataFrame to an R data frame automatically
    r_data = pandas2ri.py2rpy(data)

    # Perform the post-hoc test in R using scmamp::postHocTest
    bh_posthoc_scmamp = scmamp.postHocTest(r_data, test="friedman", correct=correct)

    # Convert the rpy2 ListVector to a Python dictionary
    bh_posthoc = {}
    summary = pd.Series(bh_posthoc_scmamp[0][0], index=data.columns)
    bh_posthoc["summary"] = summary
    bh_posthoc["summary_ranks"] = data.rank("columns").mean(axis=0)
    bh_posthoc["p_values"] = pd.DataFrame(bh_posthoc_scmamp[1], index=data.columns, columns=data.columns).fillna(1.0)
    bh_posthoc["p_adjusted"] = (pd.DataFrame(bh_posthoc_scmamp[2], index=data.columns, columns=data.columns)).fillna(1.0)+0.0001

    return bh_posthoc
