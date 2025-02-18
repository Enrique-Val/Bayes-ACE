import argparse
import os
import pickle

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2

from bayesace.algorithms.bayesace_algorithm import BayesACE
from bayesace.algorithms.face import FACE
from bayesace.algorithms.wachter import WachterCounterfactual
from experiments.utils import get_constraints, get_best_opt_params
import multiprocessing as mp

def worker(alg, instance) :
    class_var_name = alg.density_estimator.get_class_var_name()
    target_label = str(int(instance[class_var_name].to_numpy()[0]) + 2)
    result = alg.run(instance, target_label=target_label)
    return result


if __name__ == "__main__":
    mp.set_start_method('fork', force=True)
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('--parallelize', action=argparse.BooleanOptionalAction)
    parser.add_argument('--dir_name', nargs='?', default="./results/exp_eqi/", type=str)
    parser.add_argument('--dummy', action=argparse.BooleanOptionalAction)
    parser.add_argument('--penalty', nargs='?', default=1, type=int)
    args = parser.parse_args()

    data_dir = os.path.join(args.dir_name, "data_processed")
    model_dir = os.path.join(args.dir_name, "models")
    penalty = args.penalty

    # Hard code some parameters
    vertices_list = [0, 1, 2, 3]
    n_counterfactuals = 150
    sigma = -0.25
    chunks = 20
    graph_size = 1000
    verbose = False
    n_gen = 500

    # dummy mode
    if args.dummy:
        n_counterfactuals = 2
        vertices_list = [0]
        chunks = 3
        graph_size = 10
        verbose = True
        n_gen = 5

    # Load all the models and store their paths
    models = {}
    models_path = {}
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

    # Select only the instances whose target class is above 1 (improvable EQI)
    class_int = df_test[class_var_name].astype(int)
    df_counterfactuals = df_test[class_int > 1].head(n_counterfactuals)

    # The constraints will be defined by the performance of the normalizing flow model on unseen data
    sampling_range, mu_gt, std_gt, mae_gt, std_mae_gt = get_constraints(pd.concat([df_train, df_test]), df_test, models["nf"])
    print("Constraints: ", mu_gt, std_gt, mae_gt, std_mae_gt)
    logl_threshold = mu_gt + sigma * std_gt
    pp_threshold = 0.7

    manual_change = True
    if manual_change :
        # Append at the beginning of the counterfactuals California, Los Conejos (Colorado) and New York
        move_list = [6037, 8021, 36061]
        data_move = df_train[df_train.index.isin(move_list)]
        df_train = df_train[~df_train.index.isin(move_list)]
        df_counterfactuals = pd.concat([data_move, df_counterfactuals])
        df_test = pd.concat([data_move, df_test])

    # Create the necessary algorithms for the experiment.
    algorithms = {}

    # Check if they exist already
    algorithm_dir = os.path.join(args.dir_name, "algorithms")
    algorithms_paths = {}
    if not os.path.exists(algorithm_dir):
        os.makedirs(algorithm_dir)

    algorithms_paths["wachter"] = os.path.join(algorithm_dir, f"wachter.pkl")
    algorithms_paths["face"] = os.path.join(algorithm_dir, f"face_{penalty}.pkl")
    for vertices in vertices_list:
        algorithms_paths["bayesace_"+str(vertices)] = os.path.join(algorithm_dir, f"bayesace_{vertices}_{penalty}.pkl")


    # First, a FACE instance using the normalizing flow model
    if os.path.exists(algorithms_paths["face"]) and not args.dummy:
        with open(algorithms_paths["face"], "rb") as f:
            algorithms["face"] = pickle.load(f)
    else:
        alg = FACE(density_estimator=models["nf"], features=df_train.columns[:-1], chunks=chunks,
                   dataset=df_train.drop(class_var_name, axis=1).head(graph_size),
                   distance_threshold=np.inf, graph_type="kde", f_tilde=None, seed=0,
                   log_likelihood_threshold=logl_threshold, posterior_probability_threshold=pp_threshold,
                   penalty=penalty, parallelize=False, verbose=verbose)
        algorithms["face"] = alg
        print("Trained FACE")
        if not args.dummy:
            pickle.dump(alg, open(algorithms_paths["face"], "wb"))

    # Second, a Wachter instance using the normalizing flow model
    if os.path.exists(algorithms_paths["wachter"]) and not args.dummy:
        with open(algorithms_paths["wachter"], "rb") as f:
            algorithms["wachter"] = pickle.load(f)
    else:
        alg = WachterCounterfactual(density_estimator=models["nf"], features=df_train.columns[:-1],
                                    dataset=df_train,
                                    log_likelihood_threshold=logl_threshold, posterior_probability_threshold=pp_threshold,
                                    )
        algorithms["wachter"] = alg
        if not args.dummy and False:
            pickle.dump(alg, open(algorithms_paths["wachter"], "wb"))


    for vertices in vertices_list:
        if os.path.exists(algorithms_paths["bayesace_"+str(vertices)]) and not args.dummy:
            with open(algorithms_paths["bayesace_"+str(vertices)], "rb") as f:
                algorithms["bayesace_"+str(vertices)] = pickle.load(f)
        else:
            # Load the opt_algortihm params
            opt_algorithm_params = get_best_opt_params(model="bn_restricted_lim_arcs", dataset_id="EQI", dir=data_dir)
            opt_algorithm_params["pop_size"] = 100
            alg = BayesACE(density_estimator=models["bn_restricted_lim_arcs"], features=df_train.columns[:-1],
                           n_vertices=vertices, generations=n_gen, opt_algorithm=NSGA2, sampling_range=sampling_range,
                           opt_algorithm_params=opt_algorithm_params, seed=0, chunks=chunks, penalty=penalty,
                           posterior_probability_threshold=pp_threshold, log_likelihood_threshold=logl_threshold,
                           parallelize=False, verbose=verbose)
            algorithms["bayesace_"+str(vertices)] = alg
            if not args.dummy:
                pickle.dump(alg, open(algorithms_paths["bayesace_"+str(vertices)], "wb"))

    results_dir = os.path.join(args.dir_name, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save the set of counterfactuals
    if not args.dummy:
        df_counterfactuals.to_csv(os.path.join(results_dir, f"cf_{penalty}.csv"))

    # Run the experiments on the test data
    for algorithm in algorithms.keys():
        print(f"Running experiment for algorithm {algorithm}, penalty {penalty}")
        df_counterfactuals_res = df_counterfactuals.drop(class_var_name, axis=1)
        distances = pd.Series(index=df_counterfactuals_res.index)
        alg = algorithms[algorithm]
        if not args.parallelize or algorithm == "face" or algorithm == "wachter":
            results = []
            for i in range(len(df_counterfactuals.index)):
                instance = df_counterfactuals.iloc[[i]]
                target_label = str(int(instance[class_var_name].to_numpy()[0])-2)
                result = alg.run(instance, target_label=target_label)
                results.append(result)
        else:
            pool = mp.Pool(min(mp.cpu_count()-1,len(df_counterfactuals.index)))
            results = pool.starmap(worker, [(alg, df_counterfactuals.loc[[i]]) for i in df_counterfactuals.index])
            pool.close()
            pool.join()

        for i,result in enumerate(results):
            if result.counterfactual is not None:
                print("Not none")
                df_counterfactuals_res.iloc[i] = result.counterfactual.values
                distances.iloc[i] = result.distance
            else:
                df_counterfactuals_res.iloc[i] = np.nan
                distances.iloc[i] = np.nan

        # Save the results
        if args.dummy:
            print("Analysis for dummy mode with", algorithm)
        else :
            diff_df = df_counterfactuals_res - df_counterfactuals[df_counterfactuals.columns[:-1]]
            diff_df.to_csv(os.path.join(results_dir, f"diff_{algorithm}_{penalty}.csv"))
            distances.to_csv(os.path.join(results_dir, f"distances_{algorithm}_{penalty}.csv"))
            results_i_dir = os.path.join(results_dir, f"{algorithm}_{penalty}")
            if not os.path.exists(results_i_dir):
                os.makedirs(results_i_dir)
            for i,result in enumerate(results):
                pickle.dump(result, open(os.path.join(results_i_dir, f"{i}.pkl"), "wb"))

