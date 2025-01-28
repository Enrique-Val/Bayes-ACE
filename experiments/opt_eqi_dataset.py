import random
import os
import multiprocessing as mp

import pickle

import argparse

import pandas as pd
from sklearn.model_selection import ParameterGrid

from bayesace.utils import *
from experiments.experiment_opt_cv import optimize_ga
from experiments.utils import get_constraints, check_enough_instances


if __name__ == "__main__":
    # ALGORITHM PARAMETERS The likelihood parameter is relative. I.e. the likelihood threshold will be the mean logl
    # for that class plus "likelihood_threshold_sigma" sigmas of the logl std
    likelihood_threshold_sigma = -0.5
    post_prob_threshold_sigma = -0.5
    vertices_list = [0,1]
    penalty_range = (1,5)
    # Number of points for approximating integrals:
    chunks = 20
    # Number of counterfactuals
    n_counterfactuals = 15

    #Other hardcoded params
    pop_size = 100
    generations = 1000


    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('--parallelize', action=argparse.BooleanOptionalAction)
    parser.add_argument('--dir_name', nargs='?', default="./results/exp_eqi/", type=str)
    parser.add_argument('--model', default="bn_restricted_lim_arcs", choices=["bn_restricted", "bn_restricted_lim_arcs", "nf"])
    args = parser.parse_args()

    model_str: str = args.model
    parallelize = args.parallelize

    results_dir = os.path.join(args.dir_name, "data_processed/")
    model_dir = os.path.join(args.dir_name, "models")
    results_opt_dir = os.path.join(results_dir, "opt_results")

    random.seed(0)



    # Load the models
    model_path = os.path.join(model_dir, model_str + ".pkl")
    model = pickle.load(open(model_path, 'rb'))

    # Load the normalizing flow model to get the constraints
    model_nf_path = os.path.join(model_dir, "nf.pkl")
    model_nf: ConditionalDE = pickle.load(open(model_nf_path, 'rb'))

    # Name of the class variable
    class_var_name = model_nf.get_class_var_name()

    # Load some train data and the different estimators to fine tune a genetic algorithm
    df_train = pd.read_csv(os.path.join(results_dir, "data_train.csv"), index_col=0)
    # Cobvert the class to a string and categorical variable
    df_train[class_var_name] = df_train[class_var_name].astype('string').astype('category')
    # Load the scaler and apply to train_data
    scaler = pickle.load(open(os.path.join(model_dir, "scaler.pkl"), 'rb'))
    df_train[df_train.columns[:-1]] = scaler.transform(df_train[df_train.columns[:-1]])

    # Load and scale also the test data
    df_test = pd.read_csv(os.path.join(results_dir, "data_test.csv"), index_col=0)
    df_test[class_var_name] = df_test[class_var_name].astype('string').astype('category')
    df_test[df_test.columns[:-1]] = scaler.transform(df_test[df_test.columns[:-1]])
    # Select only the instances whose target class is below 5 (improvable EQI)
    class_int = df_train[class_var_name].astype(int)
    df_counterfactuals = df_train[class_int < 5].head(n_counterfactuals)

    # The constraints will be defined by the performance of the normalizing flow model on unseen data
    sampling_range, mu_gt, std_gt, mae_gt, std_mae_gt = get_constraints(df_train, df_test, model_nf)
    print("Constraints: ", mu_gt, std_gt, mae_gt, std_mae_gt)
    log_likelihood_threshold = mu_gt + likelihood_threshold_sigma * std_gt
    post_prob_threshold = min(mae_gt + post_prob_threshold_sigma * std_mae_gt, 0.99)
    # Check if there are instances with this threshold in the training set
    check_enough_instances(df_train, model, log_likelihood_threshold, post_prob_threshold)

    param_grid = {
        'eta_crossover': [10, 15, 20],  # Example range for crossover eta
        'eta_mutation': [10, 20, 30],  # Example range for mutation eta
        'selection_type': ["tourn", "ran"]  # Example range for selection type
        # Types of selection methods
    }
    param_combinations = ParameterGrid(param_grid)
    ace_params = {"posterior_probability_threshold": post_prob_threshold,
                  "log_likelihood_threshold": log_likelihood_threshold, "chunks": chunks,
                  "sampling_range": sampling_range, "opt_algorithm_params": {
            "pop_size": pop_size},
                  "generations": generations}

    results_df = optimize_ga(ace_params, param_combinations, df_counterfactuals, model_path, penalty_range,
                             n_counterfactuals, vertices_list, model_str, "EQI", parallelize)

    if not os.path.exists(results_opt_dir):
        os.makedirs(results_opt_dir)
    results_df.to_csv(os.path.join(results_opt_dir, 'results_dataEQI_' + model_str + '.csv'))
