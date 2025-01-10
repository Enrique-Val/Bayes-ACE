import argparse
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from skopt.space import Real, Integer

from sklearn.preprocessing import StandardScaler

from bayesace import hill_climbing, predict_class, brier_score, auc
from experiments.experiment_cv import get_best_normalizing_flow


def read_eqi_dataset(to_del=None):
    if to_del is None:
        # Hard coded columns to remove. They are ordinal and have a very low number of unique values:
        to_del = ["Radon", "W_ETHYLBENZ_ln", "W_HG_ln"]

    # Import EQI dataset
    data_eqi = pd.read_csv("datasets/EQI_2010/2006_2010_EQI_clean.csv", index_col=0)
    data_eqi = data_eqi.drop(columns=["State", "County_Name", "cat_RUCC"])
    data_eqi = data_eqi.dropna()

    # Discretize EQI_2Jan2018_VC and code it as str and pd.Categorical
    # Discretize in quantiles: 0-0.05, 0.05-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-0.0.95, 0.95-1
    data_eqi["EQI_2Jan2018_VC"] = pd.qcut(data_eqi["EQI_2Jan2018_VC"], q=[0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1], labels=False)
    #data_eqi["EQI_2Jan2018_VC"] = pd.qcut(data_eqi["EQI_2Jan2018_VC"], q=5, labels=False)
    data_eqi["EQI_2Jan2018_VC"] = data_eqi["EQI_2Jan2018_VC"].astype(str).astype('category')

    # Rename as class
    data_eqi = data_eqi.rename(columns={"EQI_2Jan2018_VC": "class"})

    # Store the EQI columns
    eqis = list(data_eqi.columns[:-1])

    # Load the rest of the features
    data_features = pd.read_csv("datasets/EQI_2010/PCA_Input_Variables.csv", index_col=0)

    # Delete the columns that are not needed
    data_features = data_features.drop(columns=to_del)
    features = list(data_features.columns[3:])

    # Join features and eqi
    data = data_features.join(data_eqi)
    # Drop rows with nans
    data = data.dropna()

    # Convert the necessary features
    data["Med_HH_Value"] = data["Med_HH_Value"].astype(float)

    # Save metadata and delete the selected features
    data_metadata = data[["State", "County_Name", "cat_RUCC"]]
    data = data.drop(columns=["State", "County_Name", "cat_RUCC"])

    # Create the variable dictionary. Load the corresponding file
    variable_description = pd.read_csv("datasets/EQI_2010/Data Dictionary Variables EQI 2006_2010.csv", sep=";")

    # Drop from here as well
    variable_description = variable_description[~variable_description["Variable Name"].isin(to_del)]

    # Create a dictionary with the variables, associated to a domain
    var_types = {}
    for i in np.unique(variable_description["Domain"]):
        var_types[i] = variable_description[variable_description["Domain"] == i]["Variable Name"].values

    return data, data_metadata, var_types, features, eqis


def get_bn_restrictions(features, eqis, var_types):
    # Create whitelist. Force arcs from class to all EQIs
    whitelist = []
    for i in eqis:
        whitelist.append(("class", i))

    # Create blacklist. Force no arcs between variables in different domains
    domains = np.array(list(var_types.keys()))
    # Remove IDs
    domains = np.delete(domains, np.where(domains == "ID"))
    print(domains)
    blacklist = []
    for i in domains:
        for j in domains:
            if i != j:
                for k in var_types[i]:
                    for l in var_types[j]:
                        # the first char of k and l in lower case
                        blacklist.append((k, l))

    # Discard arcs between class and the rest of the variables
    for feature in features:
        blacklist.append(("class", feature))

    # Remove arcs between eqi and the rest of the variables
    for eqi in eqis:
        for feature in features:
            blacklist.append((feature, eqi))

    # Remove arcs between domain EQIs and variables of other domains
    for eqi in eqis:
        eqi_domain = eqi.split("_")[0]
        for domain in domains:
            if not eqi_domain in domain.lower():
                for feature in var_types[domain]:
                    blacklist.append((eqi, feature))
    return whitelist, blacklist


def cross_validate_restricted_bn(dataset, max_indegree=0, blacklist=None, whitelist=None, hc_seed=None,
                                 kfold_object=None):
    if blacklist is None:
        blacklist = []
    if whitelist is None:
        whitelist = []
    if kfold_object is None:
        kfold_object = KFold(n_splits=10)
    fold_indices = list(kfold_object.split(dataset))
    # Validate Gaussian network
    bn_results = []
    # Metrics to use and metric storage
    metric = ["Logl", "LoglStd", "Brier", "AUC", "Time"]
    bn_results = pd.DataFrame(columns=metric, index=range(len(fold_indices)))

    for i, (train_index, test_index) in enumerate(fold_indices):
        df_train = dataset.iloc[train_index].reset_index(drop=True)
        scaler = StandardScaler()
        df_train[df_train.columns[:-1]] = scaler.fit_transform(df_train[df_train.columns[:-1]])
        df_val = dataset.iloc[test_index].reset_index(drop=True)
        df_val[df_val.columns[:-1]] = scaler.transform(df_val[df_val.columns[:-1]])
        t0 = time.time()
        network = hill_climbing(df_train, seed=hc_seed, bn_type="CLG", max_indegree=max_indegree,
                                arc_whitelist=whitelist,
                                arc_blacklist=blacklist, initial_structure="empty")
        time_i = time.time() - t0
        tmp = network.logl(df_val)
        bn_results.loc[i, "Logl"] = tmp.mean()
        bn_results.loc[i, "LoglStd"] = tmp.std()
        predictions = predict_class(df_val.drop("class", axis=1), network)
        brier_i = brier_score(df_val["class"].values, predictions)
        bn_results.loc[i, "Brier"] = brier_i
        auc_i = auc(df_val["class"].values, predictions)
        bn_results.loc[i, "AUC"] = auc_i
        bn_results.loc[i, "Time"] = time_i

    bn_results_mean = bn_results.mean(axis=0)
    bn_results_std = bn_results.std(axis=0)
    # Intercale in a dictionary
    results = {}
    for i in metric:
        results[i + "_mean"] = bn_results_mean[i]
        results[i + "_std"] = bn_results_std[i]
    results["params"] = {"max_indegree": max_indegree, "score": "BIC"}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('--n_iter', nargs='?', default=100, type=int)
    parser.add_argument('--parallelize', action=argparse.BooleanOptionalAction)
    parser.add_argument('--dir_name', nargs='?', default="./results/exp_cv_eqi/", type=str)
    parser.add_argument('--dummy', action=argparse.BooleanOptionalAction)
    parser.add_argument('--graphical', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    # Create dir if it does not exist
    if not os.path.exists(args.dir_name):
        os.makedirs(args.dir_name)

    # Hard coded params of the second round
    param_space = [
        Real(5e-5, 5e-4, name='lr'),
        Real(1e-4, 1, name='weight_decay'),
        Integer(2, 5, name='hidden_units'),
        Integer(1, 5, name='layers'),
        Integer(1, 8, name='n_flows'),
        Real(0.1, 0.5, name='sam_noise', prior='log-uniform')
    ]
    '''# Hard coded params of the first round
    param_space = [
        Real(1e-4, 1e-3, name='lr'),
        Real(1e-4, 1e-2, name='weight_decay'),
        Integer(2, 5, name='hidden_units'),
        Integer(1, 3, name='layers'),
        Integer(1, 5, name='n_flows'),
        Real(0.05, 0.4, name='sam_noise', prior='log-uniform')
    ]'''
    nn_params_fixed = {"steps": 500, "batch_size": 256}
    n_folds = 10
    max_indegree = 3

    DUMMY = args.dummy
    if DUMMY:
        args.n_iter = 10
        args.parallelize = False
        nn_params_fixed = {"steps": 2, "batch_size": 2000}
        n_folds = 2
        max_indegree = 1

    data, data_metadata, var_types, features, eqis = read_eqi_dataset()
    whitelist, blacklist = get_bn_restrictions(features, eqis, var_types)

    data_train, data_test = train_test_split(data, test_size=0.2, random_state=0)
    scaler = StandardScaler()
    data_train_scaled = data_train.copy()
    data_train_scaled[features + eqis] = scaler.fit_transform(data_train[features + eqis])

    if args.graphical:
        color_palette = {"0": "red", "1": "blue", "2": "green", "3": "orange", "4": "purple",
                         "5": "olive", "6": "cyan"}

        # Print a histogram of every variable. Do it in many 4x4 axis to not saturate the calls
        # Print each class with a different color (variable class)
        for features_16 in range(0, len(data_train.columns), 16):
            fig, axs = plt.subplots(4, 4, figsize=(20, 20))
            for i, feature in enumerate(data_train.columns[features_16:features_16 + 16]):
                for class_i in data_train["class"].unique():
                    axs[i // 4, i % 4].hist(data_train[data_train["class"] == class_i][feature], bins=30,
                                            color=color_palette[class_i], alpha=0.5)
                axs[i // 4, i % 4].set_title(feature)
            plt.show()

    # Create a fold object
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    # Learn the restricted BN with only max_indegree parents per node
    bn_restricted_lim_arcs = hill_climbing(data_train_scaled, seed=0, bn_type="CLG", max_indegree=max_indegree, arc_whitelist=whitelist,
                                  arc_blacklist=blacklist, initial_structure="empty")
    metrics_restricted_lim_arcs = cross_validate_restricted_bn(data_train, max_indegree=max_indegree, blacklist=blacklist,
                                                      whitelist=whitelist,
                                                      hc_seed=0,
                                                      kfold_object=kf)
    print("Restricted BN learned with limited arcs learned")

    if not DUMMY:
        # Learn the restricted BN with only max_indegree parents per node
        bn_restricted = hill_climbing(data_train_scaled, seed=0, bn_type="CLG", max_indegree=0,
                                               arc_whitelist=whitelist,
                                               arc_blacklist=blacklist, initial_structure="empty")
        metrics_restricted = cross_validate_restricted_bn(data_train, max_indegree=0, blacklist=blacklist,
                                                          whitelist=whitelist,
                                                          hc_seed=0,
                                                          kfold_object=kf)
        print("Restricted BN learned")

        bn = hill_climbing(data_train_scaled, seed=0, bn_type="CLG", initial_structure="empty")
        metrics_unrestricted = cross_validate_restricted_bn(data_train, max_indegree=0, blacklist=[], whitelist=[],
                                                            hc_seed=0, kfold_object=kf)
        print("Unrestricted BN learned")

    best_nf, metrics, result_gp = get_best_normalizing_flow(data_train_scaled, kf=kf, n_iter=args.n_iter,
                                                            nn_params_fixed=nn_params_fixed,
                                                            model_type="NVP",
                                                            parallelize=args.parallelize, working_dir=args.dir_name,
                                                            param_space=param_space)

    # Pop some params to not print/store them
    metrics[-1].pop("perms_instantiation")
    metrics[-1].pop("split_dim")

    # Create df for results. Same index as metrics keys
    results_df = pd.DataFrame(index=list(metrics_restricted_lim_arcs.keys()))
    results_df["Restricted (limited arcs)"] = metrics_restricted_lim_arcs.values()
    if not DUMMY:
        results_df["Restricted"] = metrics_restricted.values()
        results_df["Unrestricted"] = metrics_unrestricted.values()
    results_df["NF"] = metrics

    if not DUMMY:
        # Store the results_df, training data, test_data and fold object in a folder
        results_dir = args.dir_name + "data_processed/"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        results_df.to_csv(results_dir + "results.csv")
        data_train.to_csv(results_dir + "data_train.csv")
        data_test.to_csv(results_dir + "data_test.csv")
        pickle.dump(kf, open(results_dir + "kf.pkl", 'wb'))

        # Store the models
        model_dir = args.dir_name + "models/"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        pickle.dump(bn, open(model_dir + "bn_unrestricted.pkl", 'wb'))
        pickle.dump(bn_restricted_lim_arcs, open(model_dir + "bn_restricted_lim_arcs.pkl", 'wb'))
        pickle.dump(bn_restricted, open(model_dir + "bn_restricted.pkl", 'wb'))
        pickle.dump(best_nf, open(model_dir + "nf.pkl", 'wb'))

        # Store the scaler
        pickle.dump(scaler, open(model_dir + "scaler.pkl", 'wb'))

    else :
        # Just print the results_df
        print(results_df.to_string())

