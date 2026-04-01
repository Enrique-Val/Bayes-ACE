import argparse
import os
import pickle
import time
from itertools import product

import numpy as np
import pandas as pd

from bayesace.algorithms.bayesace_autodiff import SGDACE
from bayesace.algorithms.face import FACE
from bayesace.algorithms.wachter import WachterCounterfactual
from bayesace.models.lingam_cat import LingamClassifier
from experiments.utils import get_constraints, sgd_rs
import multiprocessing as mp

def worker(alg : SGDACE, instance, model_path : str, vertices, results_dir) :
    with open(model_path, "rb") as f:
        model : LingamClassifier = pickle.load(f)
    alg.density_estimator = model
    alg.n_vertices = vertices
    class_var_name = alg.density_estimator.get_class_var_name()
    target_label = str(int(instance[class_var_name].to_numpy()[0]) - 2)
    # Get the right limit of the target interval
    result, best_lr, best_time = sgd_rs(alg, instance, target_label, lr_range=(1e-5,1e-1), iters=20, seed=0, verbose=False)
    # Pickle the results
    save_path = os.path.join(results_dir, f"bayesace_{vert}_{penalty}", "lingam_"+str(i), f"{args.cf_id}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(result, f)
    return result


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('--parallelize', action=argparse.BooleanOptionalAction)
    parser.add_argument('--dir_name', nargs='?', default="./results/exp_eqi_sens/", type=str)
    parser.add_argument('--dummy', action=argparse.BooleanOptionalAction)
    parser.add_argument('--penalty', nargs='?', default=1, type=int)
    parser.add_argument('--cf_id', nargs='?', default=0, type=int)
    args = parser.parse_args()

    data_dir = os.path.join(args.dir_name, "data_processed")
    model_dir = os.path.join(args.dir_name, "models")
    penalty = args.penalty
    args.parallelize = True

    # Hard code some parameters
    n_counterfactuals = 150 # Max number of counterfactuals to generate (will be filtered by constraints later)
    vertices_list = [0, 1, 2]
    sigma = 0
    chunks = 10
    graph_size = 1000
    verbose = False
    n_perturbations = 30
    perturb_noise_std = 0.1

    # dummy mode
    if args.dummy:
        n_counterfactuals = 2
        vertices_list = [0]
        chunks = 3
        graph_size = 10
        verbose = True
        n_gen = 5

    # Load all the models and store their paths
    model_name = "lingam"
    model_path = os.path.join(model_dir, model_name+".pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    model.class_var_name = "EQI"
    class_var_name = model.get_class_var_name()

    # Load and scale the data
    df_train = pd.read_csv(os.path.join(data_dir, "data_train.csv"), index_col=0)
    df_train[class_var_name] = df_train[class_var_name].astype('string').astype('category')
    #Drop continuous variables
    df_train = df_train.drop(columns=[class_var_name+"_cont"])
    scaler = pickle.load(open(os.path.join(model_dir, "scaler.pkl"), 'rb'))
    df_train[df_train.columns[:-1]] = scaler.transform(df_train[df_train.columns[:-1]])

    df_test = pd.read_csv(os.path.join(data_dir, "data_test.csv"), index_col=0)
    df_test[class_var_name] = df_test[class_var_name].astype('string').astype('category')
    #Drop continuous variables
    df_test = df_test.drop(columns=[class_var_name+"_cont"])
    df_test[df_test.columns[:-1]] = scaler.transform(df_test[df_test.columns[:-1]])

    # Select only the instances whose target class is above 1 (improvable EQI)
    class_int = df_test[class_var_name].astype(int)
    df_counterfactuals = df_test[class_int > 1].head(n_counterfactuals)

    # The constraints will be defined by the performance of the normalizing flow model on unseen data
    sampling_range, mu_gt, std_gt, mae_gt, std_mae_gt = get_constraints(pd.concat([df_train, df_test]), df_test, model)
    print("Constraints: ", mu_gt, std_gt, mae_gt, std_mae_gt)
    logl_threshold = mu_gt + sigma * std_gt
    pp_threshold = 0.8

    manual_change = True
    if manual_change :
        # Append at the beginning of the counterfactuals California, Los Conejos (Colorado) and New York
        move_list = [6037, 8021, 36061]
        data_move = df_train[df_train.index.isin(move_list)]
        df_train = df_train[~df_train.index.isin(move_list)]
        df_counterfactuals = pd.concat([data_move, df_counterfactuals])
        df_test = pd.concat([data_move, df_test])

    # Create the necessary models for the experiment.
    if not os.path.exists(os.path.join(model_dir, "perturbed_"+str(perturb_noise_std))):
        os.makedirs(os.path.join(model_dir, "perturbed_"+str(perturb_noise_std)))
    models_path = {}
    models_path[0] = model_path

    '''for i in range(n_perturbations):
        perturbed_model = model.perturb(noise_std=perturb_noise_std)
        perturbed_model_path = os.path.join(model_dir, "perturbed_"+str(perturb_noise_std), f"lingam_{i+1}.pkl")
        with open(perturbed_model_path, "wb") as f:
            pickle.dump(perturbed_model, f)
        models_path[i+1] = perturbed_model'''
    for i in range(n_perturbations):
        perturbed_model_path = os.path.join(model_dir, "perturbed_"+str(perturb_noise_std), f"lingam_{i+1}.pkl")
        models_path[i+1] = perturbed_model_path

    # First, create a DAACE instance. The vertices, lr and model will be updated at each iteration,
    # this is just for a more straightforward parallelization.
    alg = SGDACE(density_estimator=model, chunks = chunks, features= df_train.columns[:-1],
                         n_vertices= 0, penalty = penalty, lr=1e-7, log_likelihood_threshold=logl_threshold,
                         posterior_probability_threshold=pp_threshold, max_epochs=1000, trim_features=5)

    results_dir = os.path.join(args.dir_name, "results", "perturb_"+str(perturb_noise_std))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create additional dirs
    for vert in vertices_list:
        save_path = os.path.join(results_dir, f"bayesace_{vert}_{penalty}", "lingam_" + str(i))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    # Run the experiments on the test data
    df_counterfactuals_res = df_counterfactuals.drop(class_var_name, axis=1)
    distances = pd.Series(index=df_counterfactuals_res.index)
    results = []
    if not args.parallelize :
        for vert,i in product(vertices_list, range(n_perturbations+1)) :
            instance = df_counterfactuals.iloc[[args.cf_id]]
            with open(models_path[i], "rb") as f:
                model = pickle.load(f)
            result = worker(alg, instance, model, vert, results_dir)
            results.append(result)
    else:
        with mp.Pool(processes=20, maxtasksperchild=1) as pool:
            results = pool.starmap(worker, [(alg, df_counterfactuals.iloc[[args.cf_id]], models_path[i], vert, results_dir) for vert,i in product(vertices_list, range(n_perturbations+1))])
    # Extract results
    # Already done in real time
    '''for vert_i,result in enumerate(results):
        vert = vert_i // n_perturbations
        i = vert_i % n_perturbations
        # Save in a subfolder for each algorithm
        save_path = os.path.join(results_dir, f"bayesace_{vert}_{penalty}", "lingam_"+str(i))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pickle.dump(result, open(os.path.join(save_path, f"{args.cf_id}.pkl"), "wb"))'''
