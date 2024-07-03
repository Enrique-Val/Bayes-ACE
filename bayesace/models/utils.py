import numpy as np
import pybnesian as pb
import pandas as pd
import multiprocessing as mp

from sklearn.preprocessing import StandardScaler
import openml as oml


class PybnesianParallelizationError(Exception):
    pass

def get_naive_structure(data: pd.DataFrame, type):
    naive = type(data.columns)
    for i in [i for i in data.columns if i != "class"]:
        naive.add_arc("class", i)
    return naive


def copy_structure(bn: pb.BayesianNetwork):
    copy = type(bn)(bn.nodes())
    for i in bn.arcs():
        copy.add_arc(i[0], i[1])
    return copy


def check_copy(bn):
    return bn.fitted()

def get_data(dataset_id: int):
    # Load the dataset
    data = oml.datasets.get_dataset(dataset_id, download_data=True, download_qualities=False,
                                    download_features_meta_data=False).get_data()[0]

    # Shuffle the dataset
    data = data.sample(frac=1, random_state=0)

    # Reset the index
    data = data.reset_index(drop=True)

    # Transform the class into a categorical variable
    data["class"] = data[data.columns[-1]].astype('string').astype('category')
    data = data.drop(data.columns[-2], axis=1)

    # Scale the rest of the dataset
    feature_columns = [i for i in data.columns if i != "class"]
    data[feature_columns] = StandardScaler().fit_transform(data[feature_columns].values)

    '''for i in data.columns[:-1]:
        data = data[data[i] < data[i].std()*3]
        data = data[data[i] > -data[i].std()*3]'''
    return data

def preprocess_train_data(data: pd.DataFrame | np.ndarray, jit_coef = 0):
    array_flag = False
    if isinstance(data, np.ndarray):
        # The following code but for an array instead of a dataframe:
        data = pd.DataFrame(data)
        array_flag = True
    for i in data.columns[:-1]:
        data = data[data[i] < data[i].std() * 3]
        data = data[data[i] > -data[i].std() * 3]
        data[i] = data[i] + np.random.normal(0, jit_coef*0.9 / (len(data) ** (1 / 5)), data[i].shape)
    if array_flag:
        return data.values
    else:
        return data

def hill_climbing(data: pd.DataFrame, bn_type: str, score=None, seed=0):
    bn = None
    data = preprocess_train_data(data)
    if bn_type == "CLG":
        if score is None:
            score = "bic"
        bn = pb.hc(data, start=get_naive_structure(data, pb.CLGNetwork), operators=["arcs"], score=score,
                   seed=seed)
        bn = copy_structure(bn)
    elif bn_type == "SP":
        if score is None:
            score = "validated-lik"
        # est = MMHC()
        # test = pb.MutualInformation(data, True)
        # bn = pb.MMHC().estimate(hypot_test = test, operators = pb.OperatorPool([pb.ChangeNodeTypeSet(),pb.ArcOperatorSet()]), score = pb.CVLikelihood(data), bn_type = pb.SemiparametricBNType(), patience = 20) #, score = "cv-lik"
        bn = pb.hc(data, start=get_naive_structure(data, pb.SemiparametricBN), operators=["arcs", "node_type"],
                   score=score,
                   seed=seed)
        bn = copy_structure(bn)
    elif bn_type == "Gaussian":
        if score is None:
            score = "bic"
        bn = pb.hc(data, start=get_naive_structure(data, pb.GaussianNetwork), operators=["arcs"], score=score,
                   seed=seed)
        bn = copy_structure(bn)
    else:
        raise PybnesianParallelizationError(
            "Only valid types are CLG, SP and Gaussian. For more customization use the hc method of pybnesian")
    bn.fit(data)
    bn.include_cpd = True
    pool = mp.Pool(1)
    res = pool.starmap(check_copy, [(bn,)])
    pool.close()
    if not res[0]:
        raise PybnesianParallelizationError(
            "As of version 0.4.3, PyBnesian Bayesian networks have internal and stochastic problems with the method "
            "\"copy()\"."
            "As such, the network is not parallelized correctly and experiments cannot be launched.")
    return bn