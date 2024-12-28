import argparse
import os
import pickle
from itertools import product

import numpy as np
import pandas as pd
import ucimlrepo
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from skopt.space import Real, Integer

from bayesace import hill_climbing
from experiments.experiment_cv import get_best_normalizing_flow, cross_validate_bn

TAIWANESE_BANKRUPCY_PREDICTION = 572
STUDENT_PERFORMANCE = 320
COMMUNITIES_CRIME = 183
COMMUNITIES_CRIME_UNNORMALIZED = 211

DATASET_SELECTED = COMMUNITIES_CRIME_UNNORMALIZED

# Define the parameter value range IF using Bayesian optimization
param_space = [
    Real(1e-4, 5e-3, name='lr'),
    Real(1e-4, 1e-3, name='weight_decay'),
    Integer(2, 5, name='hidden_units'),
    Integer(1, 3, name='layers'),
    Integer(1, 5, name='n_flows')
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('--results_dir', nargs='?', default='./results/exp_applied/', type=str)
    args = parser.parse_args()

    # Create results dir if it does not exist
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    uci_dataset = ucimlrepo.fetch_ucirepo(id=DATASET_SELECTED)
    data = uci_dataset.data.features
    data_ids = uci_dataset.data.ids
    data = pd.concat([data_ids, data], axis=1)
    data_og = data.copy()

    # Only keep numerical columns
    data = data.select_dtypes(include=[np.number])
    data = data.astype(float)

    # Convert to categorical and eliminate some rows
    if DATASET_SELECTED == STUDENT_PERFORMANCE:
        data["class"] = uci_dataset.data.targets[uci_dataset.data.targets.columns[0]]
        data["class"] = data["class"].apply(lambda x: "Pass" if x >= 10 else "Fail")
        data = data.drop(columns=["age", "Medu", "Fedu"])
    elif DATASET_SELECTED == TAIWANESE_BANKRUPCY_PREDICTION:
        data["class"] = data["class"].apply(lambda x: "True" if x == 0 else "False")
    elif DATASET_SELECTED == COMMUNITIES_CRIME:
        data["class"] = uci_dataset.data.targets[uci_dataset.data.targets.columns[0]]
        # Check 0.25 and 0.75 quantiles
        q1 = data["class"].quantile(0.25)
        q2 = data["class"].quantile(0.75)
        data["class"] = data["class"].apply(lambda x: "Safe" if x <= q1 else "Regular" if x <= q2 else "Unsafe")
        data = data[["medFamInc", "PctPopUnderPov", "PctLess9thGrade",
                     "PctUnemployed", "PersPerFam", "PctNotSpeakEnglWell", "PctPersDenseHous", "RentMedian",
                     "MedRentPctHousInc", "NumInShelters", "NumStreet", "PopDens",
                     "class"]]
        # population, PopDens, pctUrban
        identifiers = ["state", "county", "community", "communityname"]
    elif DATASET_SELECTED == COMMUNITIES_CRIME_UNNORMALIZED:
        data["class"] = uci_dataset.data.targets[uci_dataset.data.targets.columns[-2]]
        q1 = 150
        q2 = 750
        data["class"] = data["class"].apply(lambda x: "Safe" if x <= q1 else "Regular" if x <= q2 else "Unsafe")
        data = data[["medFamIncome", "pctPoverty", "pctLowEdu",
                     "pctUnemploy", "persPerFam", "pctNotSpeakEng", "pctPopDenseHous", "rentMed",
                     "medRentpctHousInc", "persEmergShelt", "persHomeless", "popDensity",
                     "class"]]
        # pop, popDensity, pctUrban
        identifiers = ["State", "communityname"]

    # Convert the class to categorical
    data["class"] = data["class"].astype('category')
    data["class"] = data["class"].cat.remove_unused_categories()

    # Drop nan rows
    data = data.dropna()

    # Plot data histogram per feature
    for i in range(data.shape[1] - 1):
        data.hist(column=data.columns[i], bins=50)
        plt.show()
        plt.close()

    print(len(data))

    # Remove outliers
    for column in data.columns[:-1]:
        data = data[np.abs(data[column] - data[column].mean()) <= (3 * data[column].std())]

    print(len(data))

    # Storage of results
    result_metrics = ["Logl", "LoglStd", "Brier", "AUC", "Time"]
    cartesian_product = list(product(result_metrics, ["_mean", "_std"]))
    # Flattening the list of tuples into a single list
    cartesian_product = [word1 + word2 for word1, word2 in cartesian_product]
    results_df = pd.DataFrame(
        index=cartesian_product + ["params"])
    results_df.index.name = "Crimes"

    # Use a StandardScaler
    scaler = StandardScaler()
    data[data.columns[:-1]] = scaler.fit_transform(data[data.columns[:-1]])
    # Pickle scaler in the corresponding dir
    pickle.dump(scaler, open(args.results_dir + 'scaler.pkl', 'wb'))

    # Add small noise to the data N(0, 0.01)
    data[data.columns[:-1]] += np.random.normal(0, 0.1, data[data.columns[:-1]].shape)

    kf = KFold(n_splits=3, shuffle=True, random_state=0)


    # Validate Gaussian network
    bn_results = cross_validate_bn(data, fold_indices)
    results_df["CLG"] = bn_results

    # Print results
    print("Bayesian network learned")
    dict_print = {result_metrics[i]: bn_results[i * 2] for i in range(len(result_metrics))}
    print(str(dict_print))
    print()

    # Train a and pickle the Bayesian network
    bn = hill_climbing(data=data, bn_type="CLG")

    # Train a normalizing flow
    model, metrics, result = get_best_normalizing_flow(data, kf, param_space=param_space, batch_size=128, n_iter=10, parallelize=False, steps=100, working_dir=args.results_dir)
    results_df["NF"] = metrics

    print(results_df)
    print(result)