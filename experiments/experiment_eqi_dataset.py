import argparse
import os
import pickle
from itertools import product

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2

from bayesace.algorithms.bayesace_algorithm import BayesACE
from bayesace.algorithms.face import FACE
from experiments.utils import get_constraints

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('--parallelize', action=argparse.BooleanOptionalAction)
    parser.add_argument('--dir_name', nargs='?', default="./results/exp_eqi/", type=str)
    parser.add_argument('--dummy', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    data_dir = os.path.join(args.dir_name, "data_processed")
    model_dir = os.path.join(args.dir_name, "models")

    # Hard code some parameters
    penalty_list = [1, 1, 5, 5][:1]
    vertices_list = [0, 1, 1, 2][:1]
    n_counterfactuals = 20
    sigma = -0.25
    chunks = 20

    # Load all the models and store their paths
    models = {}
    models_path = {}
    model_dir = os.path.join(args.dir_name, "models")
    for model in os.listdir(model_dir):
        model_path = os.path.join(model_dir, model)
        model_name = model.split(".")[0]
        with open(model_path, "rb") as f:
            models[model_name] = pickle.load(f)
            models_path[model_name] = model_path

    class_var_name = models["nf"].get_class_var_name()

    # Load and scale the data
    df_train = pd.read_csv(os.path.join(data_dir, "data_train.csv"), index_col=0)
    df_train[class_var_name] = df_train[class_var_name].astype('string').astype('category')
    scaler = pickle.load(open(os.path.join(model_dir, "scaler.pkl"), 'rb'))
    df_train[df_train.columns[:-1]] = scaler.transform(df_train[df_train.columns[:-1]])

    df_test = pd.read_csv(os.path.join(data_dir, "data_test.csv"), index_col=0)
    df_test[class_var_name] = df_test[class_var_name].astype('string').astype('category')
    df_test[df_test.columns[:-1]] = scaler.transform(df_test[df_test.columns[:-1]])

    # Select only the instances whose target class is below 5 (improvable EQI)
    class_int = df_test[class_var_name].astype(int)
    df_counterfactuals = df_test[class_int < 5].head(n_counterfactuals)

    # Load the metadata
    metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv"), index_col=0)
    print("df_counterfactuals")
    print(metadata.loc[df_counterfactuals.index].to_string())
    print("df_test")
    print(metadata.loc[df_test.index].to_string())
    raise Exception("Stop here")

    # The constraints will be defined by the performance of the normalizing flow model on unseen data
    sampling_range, mu_gt, std_gt, mae_gt, std_mae_gt = get_constraints(df_train, df_test, models["nf"])
    print("Constraints: ", mu_gt, std_gt, mae_gt, std_mae_gt)
    logl_threshold = mu_gt + sigma * std_gt
    pp_threshold = min(mae_gt + sigma * std_mae_gt, 0.99)

    # Create the necessary algorithms for the experiment.
    algorithms = {}
    # First, a FACE instance using the normalizing flow model
    for penalty in penalty_list :
        alg = FACE(density_estimator=models["nf"], features=df_train.columns[:-1], chunks=chunks,
                   dataset=df_train.drop(class_var_name, axis=1),
                   distance_threshold=np.inf, graph_type="integral", f_tilde=None, seed=0,
                   log_likelihood_threshold=logl_threshold, posterior_probability_threshold=pp_threshold,
                   penalty=penalty, parallelize=args.parallelize)
        alg.parallelize = False
        algorithms[("nf", 0, penalty)] = alg

    for vertices, penalty in zip(vertices_list, penalty_list):
        alg = BayesACE(density_estimator=models[models["bn_unrestricted_lim_arcs"]], features=df_train.columns[:-1], n_vertices=vertices,
                       generations=100, opt_algorithm=NSGA2, opt_algorithm_params={"pop_size": 100}, seed=0,
                       penalty=penalty, verbose=True, posterior_probability_threshold=0.90, log_likelihood_threshold=-150,
                       parallelize=False)
        algorithms[("bn", vertices, penalty)] = alg

    results_dir = os.path.join(args.dir_name, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Run the experiments on the test data
    for model, vertices, penalty in algorithms.keys():
        print(f"Running experiment for model {model}, vertices {vertices}, penalty {penalty}")
        df_counterfactuals_res = df_counterfactuals.drop(class_var_name, axis=1)
        distances = pd.Series(index=df_counterfactuals_res.index)
        alg = algorithms[(model, vertices, penalty)]
        for i in range(len(df_counterfactuals.index)):
            instance = df_counterfactuals.iloc[[i]]
            target_label = str(int(instance[class_var_name].to_numpy()[0])+2)
            result = alg.run(instance, target_label=target_label)
            if result.counterfactual is not None:
                print("Not none")
                df_counterfactuals_res.iloc[i] = result.counterfactual.values
                distances.iloc[i] = result.distance
            else:
                df_counterfactuals_res.iloc[i] = np.nan
                distances.iloc[i] = np.nan

        # Save the results
        print(df_counterfactuals_res)
        diff_df = df_counterfactuals_res - df_counterfactuals[df_counterfactuals.columns[:-1]]
        diff_df.to_csv(os.path.join(results_dir, f"diff_{model}_{vertices}_{penalty}.csv"))
        diff_df_unscaled = scaler.inverse_transform(diff_df)
        diff_df_unscaled.to_csv(os.path.join(results_dir, f"diffunscaled_{model}_{vertices}_{penalty}.csv"))
        distances.to_csv(os.path.join(results_dir, f"distances_{model}_{vertices}_{penalty}.csv"))
