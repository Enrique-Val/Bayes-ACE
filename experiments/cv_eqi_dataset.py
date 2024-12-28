import argparse
import time

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
    # data["EQI_2Jan2018_VC"] = pd.qcut(data["EQI_2Jan2018_VC"], q=[0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1], labels=False)
    data_eqi["EQI_2Jan2018_VC"] = pd.qcut(data_eqi["EQI_2Jan2018_VC"], q=5, labels=False)
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
    parser.add_argument('--n_iter', nargs='?', default=50, type=int)
    parser.add_argument('--parallelize', action=argparse.BooleanOptionalAction)
    parser.add_argument('--dir_name', nargs='?', default="./results/exp_cv_eqi/", type=str)
    args = parser.parse_args()

    data, data_metadata, var_types, features, eqis = read_eqi_dataset()
    whitelist, blacklist = get_bn_restrictions(features, eqis, var_types)

    data_train, data_test = train_test_split(data, test_size=0.2, random_state=0)
    scaler = StandardScaler()
    data_train_scaled = data_train.copy()
    data_train_scaled[features + eqis] = scaler.fit_transform(data_train[features + eqis])

    bn_restricted = hill_climbing(data_train_scaled, seed=0, bn_type="CLG", max_indegree=3, arc_whitelist=whitelist,
                                  arc_blacklist=blacklist, initial_structure="empty")
    # Create a fold object
    kf = KFold(n_splits=10, shuffle=True, random_state=0)

    # Print the validation metrics
    metrics_restricted = cross_validate_restricted_bn(data_train, max_indegree=3, blacklist=blacklist,
                                                      whitelist=whitelist,
                                                      hc_seed=0,
                                                      kfold_object=kf)

    bn = hill_climbing(data_train_scaled, seed=0, bn_type="CLG", initial_structure="empty")

    metrics_unrestricted = cross_validate_restricted_bn(data_train, max_indegree=0, blacklist=[], whitelist=[],
                                                        hc_seed=0, kfold_object=kf)

    # Define the param space for searching for the best normalizing flow
    param_space = [
        Real(1e-4, 5e-3, name='lr'),
        Real(1e-4, 1e-2, name='weight_decay'),
        Integer(2, 5, name='hidden_units'),
        Integer(1, 3, name='layers'),
        Integer(1, 8, name='n_flows')
    ]
    best_nf, metrics, result_gp = get_best_normalizing_flow(data_train_scaled, kf=kf, n_iter=args.n_iter,
                                                            nn_params_fixed={"steps": 500, "batch_size": 256},
                                                            model_type="NVP",
                                                            parallelize=args.parallelize, working_dir=args.dir_name)

