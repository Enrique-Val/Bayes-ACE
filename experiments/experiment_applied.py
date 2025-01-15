# Import Taiwan bankrupcy dataset from UCI
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2

from bayesace import brier_score, auc
from bayesace.algorithms.bayesace_algorithm import BayesACE
from bayesace.algorithms.face import FACE
from bayesace.algorithms.wachter import WachterCounterfactual

from sklearn.preprocessing import StandardScaler

import ucimlrepo

TAIWANESE_BANKRUPCY_PREDICTION = 572
STUDENT_PERFORMANCE = 320
COMMUNITIES_CRIME = 183
COMMUNITIES_CRIME_UNNORMALIZED = 211
# GERMAN_CREDIT_DATA = 144

DATASET_SELECTED = COMMUNITIES_CRIME_UNNORMALIZED

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
        data = data[["pctUrban", "medFamInc", "PctPopUnderPov", "PctLess9thGrade",
            "PctUnemployed", "PersPerFam", "PctNotSpeakEnglWell", "PctPersDenseHous", "RentMedian",
            "MedRentPctHousInc", "NumInShelters", "NumStreet", "PopDens",
            "class"]]
        # population, PopDens
        identifiers = ["state", "county", "community", "communityname"]
    elif DATASET_SELECTED == COMMUNITIES_CRIME_UNNORMALIZED:
        data["class"] = uci_dataset.data.targets[uci_dataset.data.targets.columns[-2]]
        q1 = 150
        q2 = 750
        data["class"] = data["class"].apply(lambda x: "Safe" if x <= q1 else "Regular" if x <= q2 else "Unsafe")
        data = data[["pctUrban", "medFamIncome", "pctPoverty", "pctLowEdu",
                     "pctUnemploy", "persPerFam", "pctNotSpeakEng", "pctPopDenseHous", "rentMed",
                     "medRentpctHousInc", "persEmergShelt", "persHomeless", "popDensity",
                     "class"]]
        # pop, popDensity
        identifiers = ["State", "communityname"]

    # Convert the class to categorical
    data["class"] = data["class"].astype('category')
    data["class"] = data["class"].cat.remove_unused_categories()

    # Drop nan rows
    data = data.dropna()

    # Use a StandardScaler
    scaler = StandardScaler()
    data[data.columns[:-1]] = scaler.fit_transform(data[data.columns[:-1]])
    # Pickle scaler in the corresponding dir
    pickle.dump(scaler, open(args.results_dir + 'scaler.pkl', 'wb'))

    xl = data.drop(columns=['class']).min().values
    xu = data.drop(columns=['class']).max().values
    sampling_range = (xl, xu)

    # Split the data into train and test
    training_size = 0.9
    data_train, data_test = data.iloc[:int(training_size * len(data))], data.iloc[int(training_size * len(data)):]
    #print(data_test)
    bn = hill_climbing(data_train, seed=0, bn_type="CLG")
    print(bn.cpd("class"))
    print(bn.arcs())
    print(np.mean(bn.logl(data_test)))
    predictions = bn.predict_proba(data_test.drop(columns="class").values, output="pandas")
    predictions_c = predictions.copy()
    predictions_c["real_class"] = data_test["class"].reset_index(drop=True)
    predictions_c["logl"] = bn.logl(data_test)
    # print(predictions_c)
    brier_i = brier_score(data_test["class"].values, predictions)
    print("Brier", brier_i)
    auc_i = auc(data_test["class"].values, predictions)
    print("AUC", auc_i)

    data_test_unsafe = data_test[data_test["class"] == "Unsafe"]
    print("Testing", data_test_unsafe.shape[0], "unsafe instances")
    target_label = "Safe"

    chunks = 20
    logl_thresh = bn.logl(data_test).mean() + 1 * bn.logl(data_test).std()
    pp_thresh = 0.95
    print("Logl threshold", logl_thresh)

    algorithms = []
    alg = BayesACE(bn, data_train.columns[:-1], chunks=chunks, log_likelihood_threshold=logl_thresh, n_vertex=0,
                   posterior_probability_threshold=pp_thresh, penalty=1, seed=0, verbose=True, parallelize=False,
                   opt_algorithm=NSGA2, opt_algorithm_params={"pop_size": 100}, sampling_range=sampling_range,
                   generations=100, multi_objective=False)
    algorithms.append(alg)

    alg = WachterCounterfactual(density_estimator=bn, features=data_train.columns[:-1],
                                log_likelihood_threshold=-np.inf, accuracy_threshold=0, dataset=data_train[data_train["class"] == "Safe"])
    algorithms.append(alg)

    '''alg = FACE(density_estimator=bn, features=data_train.columns[:-1], chunks=10,
                       dataset=data_train.drop("class", axis=1),
                       distance_threshold=np.inf, graph_type="kde", f_tilde=None, seed=0,
                       log_likelihood_threshold=logl_thresh, accuracy_threshold=pp_thresh, penalty=1, parallelize=False)
    algorithms_str = ["BayesACE", "Wachter", "FACE"]'''
    algorithms_str = ["BayesACE", "Wachter"]

    # Store of results

    for i in data_test_unsafe.index:
        # Look for a counterfactual
        og_instance = data_test_unsafe.loc[[i]]
        print(data_og[identifiers].loc[[i]])
        og_instance_transformed = pd.DataFrame(scaler.inverse_transform(og_instance.drop(columns="class")),
                                               columns=og_instance.columns[:-1])
        og_instance_transformed["class"] = og_instance["class"].values
        print(og_instance_transformed.iloc[0])
        for algorithm, algorithm_str in zip(algorithms, algorithms_str):
            result = algorithm.run(og_instance, target_label)
            result_transformed = scaler.inverse_transform(pd.DataFrame(result.counterfactual.values).T)
            result_transformed = pd.Series(result_transformed[0], index=og_instance.columns[:-1])
            print("Difference with", algorithm_str)
            print(-og_instance_transformed.iloc[0].drop("class") + result_transformed)
        print()
