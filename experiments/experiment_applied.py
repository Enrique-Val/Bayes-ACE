# Import Taiwan bankrupcy dataset from UCI
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2

from bayesace import hill_climbing, get_other_class, brier_score, predict_class
from bayesace.algorithms.bayesace_algorithm import BayesACE
from bayesace.algorithms.face import FACE
from bayesace.algorithms.wachter import WachterCounterfactual

from sklearn.preprocessing import StandardScaler

import ucimlrepo

TAIWANESE_BANKRUPCY_PREDICTION = 572
STUDENT_PERFORMANCE = 320
COMMUNITIES_CRIME = 183
#GERMAN_CREDIT_DATA = 144

DATASET_SELECTED = COMMUNITIES_CRIME


if __name__ == "__main__":
    uci_dataset = ucimlrepo.fetch_ucirepo(id=DATASET_SELECTED)
    data = uci_dataset.data.features
    data_og = data.copy()
    data_og["class"] = uci_dataset.data.targets[uci_dataset.data.targets.columns[0]]
    # Only keep numerical columns
    data = data.select_dtypes(include=[np.number])
    data = data.astype(float)

    data["class"] = uci_dataset.data.targets[uci_dataset.data.targets.columns[0]]
    # Convert to categorical and eliminate some rows
    if DATASET_SELECTED == STUDENT_PERFORMANCE:
        data["class"] = data["class"].apply(lambda x: "Pass" if x >= 10 else "Fail").astype('category')
        data = data.drop(columns= ["age", "Medu", "Fedu"])
    elif DATASET_SELECTED == TAIWANESE_BANKRUPCY_PREDICTION:
        data["class"] = data["class"].apply(lambda x: "True" if x == 0 else "False").astype('category')
    elif DATASET_SELECTED == COMMUNITIES_CRIME:
        # Check 0.33 and 0.66 quantiles
        q1 = data["class"].quantile(0.33)
        q2 = data["class"].quantile(0.66)
        data["class"] = data["class"].apply(lambda x: "Safe" if x <= q1 else "Regular" if x <= q2 else "Unsafe").astype('category')
        # Drop rows labelled as "Regular"
        #data = data[data["class"] != "Regular"]
        data = data[["population","householdsize","pctUrban","medFamInc","PctPopUnderPov","PctLess9thGrade","PctUnemployed","PersPerFam","PctNotSpeakEnglWell","PctPersDenseHous","RentMedian","MedRentPctHousInc","NumInShelters","NumStreet","PopDens","class"]]
    # Drop nan rows
    data = data.dropna()

    # Use a StandardScaler
    scaler = StandardScaler()
    data[data.columns[:-1]] = scaler.fit_transform(data[data.columns[:-1]])

    xl = data.drop(columns=['class']).min().values - 0.01
    xu = data.drop(columns=['class']).max().values + 0.01
    sampling_range = (xl, xu)

    data_train, data_test = data.iloc[:int(0.8*len(data))], data.iloc[int(0.8*len(data)):]
    print(data_test)
    bn = hill_climbing(data_train, seed=0, bn_type="CLG")
    print(bn.cpd("class"))
    print(bn.arcs())
    print(np.mean(bn.logl(data_test)))
    predictions = predict_class(data_test.drop("class", axis=1), bn)
    predictions_c = predictions.copy()
    predictions_c["real_class"] = data_test["class"].reset_index(drop=True)
    predictions_c["logl"] = bn.logl(data_test)
    print(predictions_c)
    brier_i = brier_score(data_test["class"].values, predictions)
    print(brier_i)

    data_set_unsafe = data_test[data_test["class"] == "Unsafe"]
    target_label = "Safe"

    chunks = 20
    logl_thresh = -10
    pp_thresh = 0.95

    algorithms = []
    alg = BayesACE(bn, data_train.columns[:-1], chunks=chunks, log_likelihood_threshold=logl_thresh, n_vertex=0, accuracy_threshold=pp_thresh, penalty=1, seed=0, verbose=True, parallelize=False,
                             opt_algorithm=NSGA2, opt_algorithm_params={"pop_size": 100}, sampling_range=sampling_range, generations=100, multi_objective=False)
    algorithms.append(alg)

    alg = WachterCounterfactual(density_estimator=bn, features=data_train.columns[:-1],
                   log_likelihood_threshold=logl_thresh, accuracy_threshold=pp_thresh, dataset=data_train)
    algorithms.append(alg)

    '''alg = FACE(density_estimator=bn, features=data_train.columns[:-1], chunks=10,
                       dataset=data_train.drop("class", axis=1),
                       distance_threshold=np.inf, graph_type="kde", f_tilde=None, seed=0,
                       log_likelihood_threshold=logl_thresh, accuracy_threshold=pp_thresh, penalty=1, parallelize=False)
    algorithms_str = ["BayesACE", "Wachter", "FACE"]'''
    algorithms_str = ["BayesACE", "Wachter"]
    for i in data_set_unsafe.index:
        # Look for a counterfactual
        og_instance = data_set_unsafe.loc[[i]]
        print(data_og[["state", "county", "community", "communityname"]].loc[[i]])
        og_instance_transformed = pd.DataFrame(scaler.inverse_transform(og_instance.drop(columns="class")),
                                               columns=og_instance.columns[:-1])
        og_instance_transformed["class"] = og_instance["class"]
        print(og_instance_transformed.iloc[0])
        for algorithm,algorithm_str in zip(algorithms,algorithms_str):
            result = algorithm.run(og_instance, target_label)
            result_transformed = scaler.inverse_transform(pd.DataFrame(result.counterfactual.values).T)
            result_transformed = pd.Series(result_transformed[0], index=og_instance.columns[:-1])
            print("Difference with", algorithm_str)
            print(-og_instance_transformed.iloc[0].drop("class")+result_transformed)
        print()
