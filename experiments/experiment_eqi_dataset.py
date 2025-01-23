import argparse
import os
import pickle
from itertools import product

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2

from bayesace.algorithms.bayesace_algorithm import BayesACE

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('--parallelize', action=argparse.BooleanOptionalAction)
    parser.add_argument('--dir_name', nargs='?', default="./results/exp_cv_eqi/", type=str)
    parser.add_argument('--dummy', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    # Hard code some parameters
    penalty_list = [1, 5][:1]
    vertices_list = [0, 1, 2][:1]
    max_n_counterfactuals = 20

    # Load all the models and store their paths
    models = {}
    models_path = {}
    model_dir = os.path.join(args.dir_name, "models")
    for model in os.listdir(model_dir):
        if model != "bn_restricted.pkl":
            continue
        model_path = os.path.join(model_dir, model)
        with open(model_path, "rb") as f:
            models[model] = pickle.load(f)
            models_path[model] = model_path

    # Load the test data
    data_dir = os.path.join(args.dir_name, "data_processed")
    test_data = pd.read_csv(os.path.join(data_dir, "data_test.csv"), index_col=0)
    # Convert the class to a string and categorical variable
    test_data["class"] = test_data["class"].astype('string').astype('category')

    # Load the scaler and apply to test_data
    scaler_path = os.path.join(data_dir, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        test_data[test_data.columns[:-1]] = scaler.transform(test_data[test_data.columns[:-1]])

    # Select only the instances whose target class is 0 (bad EQI)
    test_data = test_data[test_data["class"] == "0"]
    test_data = test_data.head(1)

    print(test_data)

    # Create the necessary algorithms for the experiment. All BayesACE models
    algorithms = {}
    for model, vertices, penalty in product(models.keys(), vertices_list, penalty_list):
        alg = BayesACE(density_estimator=models[model], features=test_data.columns[:-1], n_vertices=vertices,
                       generations=100, opt_algorithm=NSGA2, opt_algorithm_params={"pop_size": 100}, seed=0,
                       penalty=penalty, verbose=True, posterior_probability_threshold=0.90, log_likelihood_threshold=-150,
                       parallelize=False)
        algorithms[(model, vertices, penalty)] = alg

    results_dir = os.path.join(args.dir_name, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Run the experiments on the test data
    for model, vertices, penalty in algorithms.keys():
        print(f"Running experiment for model {model}, vertices {vertices}, penalty {penalty}")
        df_counterfactuals = pd.DataFrame(columns=test_data.columns[:-1], index=test_data.index)
        distances = pd.Series(index=test_data.index)
        alg = algorithms[(model, vertices, penalty)]
        for i in range(len(test_data.index)):
            instance = test_data.iloc[[i]]
            print(instance)
            result = alg.run(instance, target_label="4")
            if result.counterfactual is not None:
                print("Not none")
                df_counterfactuals.iloc[i] = result.counterfactual.values
                distances.iloc[i] = result.distance
            else:
                df_counterfactuals.iloc[i] = np.nan
                distances.iloc[i] = np.nan

        # Save the results
        print(test_data)
        print(df_counterfactuals)
        diff_df = df_counterfactuals - test_data[test_data.columns[:-1]]
        diff_df.to_csv(os.path.join(results_dir, f"diff_{model}_{vertices}_{penalty}.csv"))
