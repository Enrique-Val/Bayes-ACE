import numpy as np
import pandas as pd
from pybnesian import LinearGaussianCPD
from scipy.stats import kstest
from sklearn.model_selection import train_test_split
from bayesace.models.conditional_kde import ConditionalKDE
from bayesace.models.conditional_nvp import ConditionalNVP
from sklearn.preprocessing import StandardScaler

from bayesace import hill_climbing, predict_class, brier_score, auc

def read_eqi_dataset(to_del = None) :
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

def get_bn_restrictions(features, eqis, var_types) :
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

if __name__ == "__main__":

    data, data_metadata, var_types, features, eqis = read_eqi_dataset()
    whitelist, blacklist = get_bn_restrictions(features, eqis, var_types)

    data_train, data_test = train_test_split(data, test_size=0.2, random_state=0)
    scaler = StandardScaler()
    data_train[features+eqis] = scaler.fit_transform(data_train[features+eqis])

    bn = hill_climbing(data_train, seed=0, bn_type="CLG", max_indegree=3, arc_whitelist=whitelist,
                          arc_blacklist=blacklist, initial_structure="empty")

    # Print the learned network
    print(bn.arcs())

    for i in bn.nodes():
        print(bn.cpd(i))

    # Print the Brier score
    data_test_scaled = scaler.transform(data_test[features+eqis])
    data_test_scaled = pd.DataFrame(data_test_scaled, columns=features+eqis, index=data_test.index)
    data_test_scaled["class"] = data_test["class"]
    predictions = predict_class(data_test_scaled.drop("class", axis=1), bn)
    brier = brier_score(data_test["class"].values, predictions)
    print("Brier score:", brier)

    # Print the AUC
    auc_i = auc(data_test["class"].values, predictions)
    print("AUC:", auc_i)

    # Print logl mean (only for samples over 1% cuantile)
    logl = bn.logl(data_test_scaled)
    print("Logl mean:", logl.mean())

    '''# Train a CKDE
    ckde = ConditionalKDE(bandwidth=0.5)
    ckde.train(data)
    print(ckde.logl(df_test).mean())

    predictions_kde = predict_class(df_test.drop("class", axis=1), ckde)
    brier_kde = brier_score(df_test["class"].values, predictions_kde)
    print("Brier score KDE:", brier_kde)'''


    '''# Train a nf model, specifically, nvp
    cnvp = ConditionalNVP()
    cnvp.train(data, layers=3, hidden_units=140*5, batch_size=256, lr=0.01, n_flows=8, steps=500)
    print(cnvp.logl(data_test).mean())

    predictions_nvp = predict_class(data_test.drop("class", axis=1), cnvp)
    brier_nvp = brier_score(data_test["class"].values, predictions_nvp)
    print("Brier score NVP:", brier_nvp)

    auc_nvp = auc(data_test["class"].values, predictions_nvp)
    print("AUC NVP:", auc_nvp)'''



