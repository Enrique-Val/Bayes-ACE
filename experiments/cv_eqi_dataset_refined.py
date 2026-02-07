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

from bayesace import brier_score, auc
from bayesace.models.bayesian_network_classifier import BayesianNetworkClassifier
from bayesace.models.lingam_cat import LingamClassifier
from experiments.experiment_cv import get_best_normalizing_flow

class_var_name = "EQI"

def read_eqi_dataset(delete_features = True):
    # Import EQI dataset
    data_eqi = pd.read_csv("datasets/EQI_2010/2006_2010_EQI_clean.csv", index_col=0)
    data_eqi = data_eqi.drop(columns=["State", "County_Name", "cat_RUCC"])
    data_eqi = data_eqi.dropna()

    # Retrieve class
    cont_class = data_eqi["EQI_2Jan2018_VC"]
    data_eqi.drop(columns=["EQI_2Jan2018_VC"], inplace=True)

    # Bin the class into 7 categories, using quantiles
    binned_class, bins = pd.qcut(cont_class, q=[0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0], retbins=True, labels=False)
    # Remove first and last bin
    bins = bins[1:-1]

    data_eqi[class_var_name+"_cont"] = cont_class
    # Convert to category and string
    data_eqi[class_var_name] = binned_class.astype(str).astype('category')

    categories = [str(i) for i in range(len(bins) + 1)]

    # Store the EQI columns
    eqis = list(data_eqi.columns[:-1])

    # Load the rest of the features
    data_features = pd.read_csv("datasets/EQI_2010/PCA_Input_Variables.csv", index_col=0)

    # Save the name of the metadata features.
    metadata_features = list(data_features.columns[:3])

    # Store the data separately, for the moment
    data_tmp = data_features.drop(columns=metadata_features)

    # Set data to float
    data_tmp = data_tmp.astype(float)

    to_del = []
    # If we want to delete the features, then preprocess as usual
    if delete_features:
        data_features_norm = data_tmp
        data_features_norm[:] = StandardScaler().fit_transform(data_features_norm)
        for i, col in enumerate(data_features_norm.columns):
            if data_features_norm[col].nunique() < 20 or np.sort(np.histogram(data_features_norm[col], bins=100)[0])[-3:].sum() > len(data_features) * 0.9:
                to_del.append(col)

    data_features = data_features.drop(columns=to_del)

    # After the deletion, store the data features
    features = list(data_features.columns[3:])

    # Join features and eqi
    data = data_features.join(data_eqi)
    # Drop rows with nans
    data = data.dropna()

    # Save metadata and delete it from the actual data. Cast it to float (except the class, last column)
    data_metadata = data[metadata_features]
    data = data.drop(columns=metadata_features) # or data = data[features+eqis]
    data[data.columns[:-1]] = data[data.columns[:-1]].astype(float)

    # Create the variable dictionary. Load the corresponding file
    variable_description = pd.read_csv("datasets/EQI_2010/Data Dictionary Variables EQI 2006_2010.csv", sep=";")

    # Drop from here as well
    variable_description = variable_description[~variable_description["Variable Name"].isin(to_del)]

    # Create a dictionary with the variables, associated to a domain
    var_types = {}
    for i in np.unique(variable_description["Domain"]):
        var_types[i] = variable_description[variable_description["Domain"] == i]["Variable Name"].to_numpy()

    return data, data_metadata, var_types, features, eqis, bins, categories


def get_bn_restrictions(features, eqis, var_types):
    total_n = len(features) + len(eqis)
    prior_knowledge = -np.ones((total_n, total_n))

    # 1 means forced arc, -1 means forbidden arc, 0 means no restriction
    # Force arcs from each EQI to the class variable
    for i in eqis[:-1]:
        prior_knowledge[-1, eqis.index(i) + len(features)] = 1  # from EQI to class

    # Forbid arcs between variables in different domains
    domains = np.array(list(var_types.keys()))
    # Remove IDs
    domains = np.delete(domains, np.where(domains == "ID"))

    for i in domains:
        for j in domains:
            if i != j:
                for k in var_types[i]:
                    for l in var_types[j]:
                        prior_knowledge[features.index(k), features.index(l)] = 0

    # Discard arcs between class and the rest of the variables
    for feature in features:
        prior_knowledge[features.index(feature), -1] = 0
        prior_knowledge[-1, features.index(feature)] = 0

    # Remove arcs between eqi and the rest of the variables
    for eqi in eqis:
        for feature in features:
            prior_knowledge[features.index(feature), eqis.index(eqi) + len(features)] = 0

    # Remove arcs between domain EQIs and variables of other domains
    for eqi in eqis:
        eqi_domain = eqi.split("_")[0]
        for domain in domains:
            if not eqi_domain in domain.lower():
                for feature in var_types[domain]:
                    prior_knowledge[eqis.index(eqi) + len(features), features.index(feature)] = 0
    print(prior_knowledge)
    # Dump the prior knowledge
    pd_prior_knowledge = pd.DataFrame(prior_knowledge, index=features + eqis, columns=features + eqis)
    pd_prior_knowledge.to_csv("results/exp_cv_eqi/prior_knowledge.csv")
    return prior_knowledge


def cross_validate_lingam(dataset, kfold_object=None, bin_edges=None, bin_names=None, prior_knowledge=None, seed=0):
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
        X_train = df_train.drop(class_var_name, axis=1)
        X_train = X_train.drop(class_var_name+"_cont", axis=1)
        y_train = df_train[class_var_name]
        y_train_cont = df_train[class_var_name+"_cont"]
        bin_edges_scaled = scaler_class.fit_transform(np.array(bin_edges).reshape(-1, 1)).flatten()
        df_val = dataset.iloc[test_index].reset_index(drop=True)
        df_val[df_val.columns[:-1]] = scaler.transform(df_val[df_val.columns[:-1]])
        t0 = time.time()
        network = LingamClassifier(bin_edges=bin_edges_scaled, bin_names=bin_names, prior_knowledge=prior_knowledge,
                                   random_state=seed)
        network.fit(X_train, y_train_cont)
        time_i = time.time() - t0
        X_val = df_val.drop(class_var_name, axis=1)
        X_val = X_val.drop(class_var_name+"_cont", axis=1)
        y_val = df_val[class_var_name]
        tmp = network.logl(X_val, y_val)
        bn_results.loc[i, "Logl"] = tmp.mean()
        bn_results.loc[i, "LoglStd"] = tmp.std()
        predictions = network.predict_proba(X_val.to_numpy(), output="pandas")
        brier_i = brier_score(y_val.to_numpy(), predictions)
        bn_results.loc[i, "Brier"] = brier_i
        auc_i = auc(y_val.to_numpy(), predictions)
        bn_results.loc[i, "AUC"] = auc_i
        bn_results.loc[i, "Time"] = time_i

    bn_results_mean = bn_results.mean(axis=0)
    bn_results_std = bn_results.std(axis=0)
    # Intercale in a dictionary
    results = {}
    for i in metric:
        results[i + "_mean"] = bn_results_mean[i]
        results[i + "_std"] = bn_results_std[i]
    results["params"] = {}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('--n_iter', nargs='?', default=100, type=int)
    parser.add_argument('--parallelize', action=argparse.BooleanOptionalAction)
    parser.add_argument('--dir_name', nargs='?', default="./results/exp_eqi/", type=str)
    parser.add_argument('--dummy', action=argparse.BooleanOptionalAction)
    parser.add_argument('--graphical', action=argparse.BooleanOptionalAction)
    parser.add_argument('--gpu', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    # Create dir if it does not exist
    if not os.path.exists(args.dir_name):
        os.makedirs(args.dir_name)

    nn_params_fixed = {"steps": 500, "batch_size": 256}
    n_folds = 2

    DUMMY = args.dummy
    if DUMMY:
        args.n_iter = 10
        args.parallelize = False
        nn_params_fixed = {"steps": 2, "batch_size": 2000}
        n_folds = 2

    data, data_metadata, var_types, features, eqis, bins, categories = read_eqi_dataset(delete_features=True)
    #pickle.dump(var_types, open(os.path.join(args.dir_name, "data_processed", "var_types.pkl"), "wb"))
    #data_metadata.to_csv(os.path.join(args.dir_name, "data_processed", "metadata.csv"))
    prior_knowledge = get_bn_restrictions(features, eqis, var_types)

    # Remove EQI_cont from eqis
    eqis = eqis[:-1]

    # Save the metadata
    #data_metadata.to_csv(os.path.join(args.dir_name,"data_processed", "metadata.csv"))

    data_train, data_test = train_test_split(data, test_size=0.2, random_state=0)
    # If Los Angeles (6037) or Northwest Arctic (2188) are in train, put them in test, at the beginning
    manual_change = False
    if manual_change:
        move_list = [6037, 2188, 36061]
        data_move = data_train[data_train.index.isin(move_list)]
        data_train = data_train[~data_train.index.isin(move_list)]
        data_test = pd.concat([data_move, data_test])

    scaler = StandardScaler()
    data_train_scaled = data_train.copy()
    data_train_scaled[features + eqis] = scaler.fit_transform(data_train[features + eqis])
    #pickle.dump(scaler, open(os.path.join(args.dir_name, "models/scaler.pkl"), "wb"))
    X_train = data_train_scaled[data_train_scaled.columns[:-2]]
    y_train_cont = data_train_scaled[class_var_name+"_cont"]
    scaler_class = StandardScaler()
    y_train_cont = scaler_class.fit_transform(y_train_cont.to_numpy().reshape(-1, 1)).flatten()
    y_train = data_train_scaled[class_var_name]

    nn_params_fixed["split_dim"] = len(X_train.columns) // 2

    if args.graphical:
        n_features = len(data_train.columns) - 1
        n_final_features = 0
        color_palette = {"0": "red", "1": "blue", "2": "green", "3": "orange", "4": "purple",
                         "5": "olive", "6": "cyan"}

        # Print a histogram of every variable. Do it in many 4x4 axis to not saturate the calls
        # Print each class with a different color (variable class)
        for features_16 in range(0, len(data_train.columns), 16):
            fig, axs = plt.subplots(4, 4, figsize=(20, 20))
            for i, feature in enumerate(data_train.columns[features_16:features_16 + 16]):
                for class_i in data_train[class_var_name].unique():
                    axs[i // 4, i % 4].hist(data_train[data_train[class_var_name] == class_i][feature], bins=30,
                                            color=color_palette[class_i], alpha=0.5)
                axs[i // 4, i % 4].set_title(feature)
            plt.show()
        # Now plot for all the EQIs
        for features_16 in range(0, len(data_train.columns[:-1]), 16):
            fig, axs = plt.subplots(4, 4, figsize=(20, 20))
            for i, feature in enumerate(data_train.columns[features_16:features_16 + 16]):
                if data_train[feature].nunique() < 20 or np.sort(np.histogram(data_train[feature], bins=100)[0])[-3:].sum() > len(data_train) * 0.95:
                    axs[i // 4, i % 4].hist(data_train[feature], bins=100, color="red")
                elif data_train[feature].nunique() < 20 or np.sort(np.histogram(data_train[feature], bins=100)[0])[-3:].sum() > len(data_train) * 0.90:
                    axs[i // 4, i % 4].hist(data_train[feature], bins=100, color="orange")

                else :
                    axs[i // 4, i % 4].hist(data_train[feature], bins=100, color = "green")
                    n_final_features += 1
                axs[i // 4, i % 4].set_title(feature)
            plt.show()
        print(f"Initial features: {n_features}, final features: {n_final_features}")
        raise Exception("Graphical mode finished")

    # Create a fold object
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    bins_scaled = scaler_class.transform(np.array(bins).reshape(-1, 1)).flatten()
    bn_restricted_lim_arcs = LingamClassifier(bin_edges=bins_scaled, bin_names=categories, prior_knowledge=prior_knowledge,
                                              random_state=0)
    bn_restricted_lim_arcs.fit(X_train, y_train_cont)
    '''metrics_restricted_lim_arcs = cross_validate_lingam(dataset=data_train, kfold_object=kf,
                                                        bin_edges=bins_scaled, bin_names=categories,
                                                        prior_knowledge=prior_knowledge, seed=0)'''
    print("Lingam learned")
    pickle.dump(bn_restricted_lim_arcs, open(os.path.join(args.dir_name, "models/lingam_simple.pkl"), "wb"))
    raise Exception("Graphical mode finished")


    # Create df for results. Same index as metrics keys
    results_dir = args.dir_name + "data_processed/"
    results_df = pd.read_csv(results_dir + "results.csv", index_col=0)
    results_df["Lingam"] = metrics_restricted_lim_arcs.values()

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
        pickle.dump(bn_restricted_lim_arcs, open(model_dir + "lingam.pkl", 'wb'))

        # Store the scaler
        pickle.dump(scaler, open(model_dir + "scaler.pkl", 'wb'))
        pickle.dump(scaler_class, open(model_dir + "scaler_class.pkl", 'wb'))

    else:
        # Just print the results_df
        print(results_df.to_string())
