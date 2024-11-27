# Import Taiwan bankrupcy dataset from UCI
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2

from bayesace import hill_climbing, get_other_class
from bayesace.algorithms.bayesace_algorithm import BayesACE
from bayesace.algorithms.face import FACE
from bayesace.algorithms.wachter import WachterCounterfactual

TAIWAN = 0
STUDENTS = 1

DATASET_SELECTED = STUDENTS


def get_opposite_class(param):
    pass


if __name__ == "__main__":
    # Set mp method to fork


    data = None
    if DATASET_SELECTED == TAIWAN :
        data = pd.read_csv("datasets/taiwan_bank.csv")

    elif DATASET_SELECTED == STUDENTS:
        data = pd.read_csv("datasets/student-mat.csv", sep=";")
        # Retain only numerical columns
        data = data.select_dtypes(include=['number'])
        data = data.drop(columns=['age','Medu','Fedu'])
        data = data.astype(float)
        # Normalize the data by standardizing it
        data = (data - data.mean()) / data.std()
        data["class"] = data["G3"].apply(lambda x: "pass" if x >= 0 else "fail").astype('category')
        data = data.drop(columns=['G1','G2','G3'])
        # Shuffle the data

    data_train, data_test = data.iloc[:int(0.8*len(data))], data.iloc[int(0.8*len(data)):]
    print(data_train[data_train["class"]=="pass"])
    bn = hill_climbing(data_train, seed=0, bn_type="CLG")
    print(bn.cpd("class"))
    print(np.mean(bn.logl(data_test)))

    # Look for a counterfactual
    og_instance = data_test.iloc[[1]]
    print(og_instance)

    xl = data.drop(columns=['class']).min().values-1
    xu = data.drop(columns=['class']).max().values+1
    sampling_range = (xl, xu)
    algorithm = BayesACE(bn, data_train.columns[:-1], chunks=10, log_likelihood_threshold=-14, n_vertex=1, accuracy_threshold=0.90, penalty=20, seed=0, verbose=True, parallelize=False,
                         opt_algorithm=NSGA2, opt_algorithm_params={"pop_size": 100}, sampling_range=sampling_range, generations=100, multi_objective=False)
    '''algorithm = FACE(density_estimator=bn, features=data_train.columns[:-1], chunks=10,
               dataset=data_train.drop("class", axis=1),
               distance_threshold=np.inf, graph_type="integral", f_tilde=None, seed=0,
               log_likelihood_threshold=-14, accuracy_threshold=0.95, penalty=1, parallelize=False)'''
    target_label = get_other_class(og_instance["class"].cat.categories, og_instance["class"].values[0])
    result = algorithm.run(og_instance, target_label)
    print(og_instance.iloc[0])
    print()
    print(result.counterfactual)
    print("Difference with BayesACE:")
    print(-og_instance.iloc[0].drop("class")+result.counterfactual)
    alg_wachter = WachterCounterfactual(density_estimator=bn, features=data_train.columns[:-1],
               log_likelihood_threshold=-14, accuracy_threshold=0.90, dataset=data_train)
    result_wachter = alg_wachter.run(og_instance, target_label)
    print("Difference with Wachter:")
    print(-og_instance.iloc[0].drop("class")+result_wachter.counterfactual)

    algorithm = FACE(density_estimator=bn, features=data_train.columns[:-1], chunks=10,
                   dataset=data_train.drop("class", axis=1),
                   distance_threshold=np.inf, graph_type="integral", f_tilde=None, seed=0,
                   log_likelihood_threshold=-14, accuracy_threshold=0.90, penalty=20, parallelize=False)
    result = algorithm.run(og_instance, target_label)
    print("Difference with FACE:")
    print(-og_instance.iloc[0].drop("class") + result.counterfactual)
