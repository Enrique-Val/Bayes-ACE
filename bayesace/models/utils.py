import numpy as np
import pybnesian as pb
import pandas as pd
import multiprocessing as mp

from sklearn.preprocessing import StandardScaler

class PybnesianParallelizationError(Exception):
    pass


def get_initial_structure(data: pd.DataFrame, bn_type, structure_type="naive"):
    initial = bn_type(data.columns)
    if structure_type == "naive":
        for i in [i for i in data.columns if i != "class"]:
            initial.add_arc("class", i)
    elif structure_type == "empty":
        pass
    else:
        raise ValueError("Invalid structure type. Only valid types are naive and empty.")
    return initial


def copy_structure(bn: pb.BayesianNetwork):
    copy = type(bn)(bn.nodes())
    for i in bn.arcs():
        copy.add_arc(i[0], i[1])
    return copy


def check_copy(bn):
    return bn.fitted()


def preprocess_data(data: pd.DataFrame | np.ndarray,  eliminate_outliers=np.inf, standardize=True,
                    min_unique_vals=20, max_cum_values=3, max_instances=100000):
    array_flag = False
    if isinstance(data, np.ndarray):
        # The following code but for an array instead of a dataframe:
        data = pd.DataFrame(data)
        array_flag = True
    # Separate the target column (last column) from the features
    data = data.head(max_instances)
    target_column = data.columns[-1]
    features = data.columns[:-1]

    feature_data = data[features]
    feature_data = feature_data.loc[:, feature_data.nunique() >= min_unique_vals]

    feature_data = feature_data.loc[:, feature_data.apply(lambda x: np.sort(np.histogram(x, bins=100)[0])[-max_cum_values:].sum() < len(data)*0.95, axis=0)]
    data = pd.concat([feature_data, data[target_column]], axis=1)

    means = data[data.columns[:-1]].mean()
    stds = data[data.columns[:-1]].std()
    data = data[(np.abs((data[data.columns[:-1]] - means) / stds) < eliminate_outliers).all(axis=1)]

    if standardize:
        data[data.columns[:-1]] = StandardScaler().fit_transform(data[data.columns[:-1]].values)
    # Assert that there are no missing values
    if data.isnull().values.any():
        raise ValueError("There are missing values in the post-processed dataset.")
    if array_flag:
        return data.values
    else:
        return data


def hill_climbing(data: pd.DataFrame, bn_type: str, score=None, max_indegree=0,
                  arc_blacklist=[], arc_whitelist=[], initial_structure="naive", seed=0):
    bn = None
    if bn_type == "CLG":
        if score is None:
            score = "bic"
        bn = pb.hc(data, start=get_initial_structure(data, pb.CLGNetwork,initial_structure), operators=["arcs"], score=score,
                   arc_blacklist=arc_blacklist, arc_whitelist=arc_whitelist, max_indegree=max_indegree, seed=seed)
        bn = copy_structure(bn)
    elif bn_type == "SP":
        if score is None:
            score = "validated-lik"
        # est = MMHC()
        # test = pb.MutualInformation(data, True)
        # bn = pb.MMHC().estimate(hypot_test = test, operators = pb.OperatorPool([pb.ChangeNodeTypeSet(),pb.ArcOperatorSet()]), score = pb.CVLikelihood(data), bn_type = pb.SemiparametricBNType(), patience = 20) #, score = "cv-lik"
        bn = pb.hc(data, start=get_initial_structure(data, pb.SemiparametricBN,initial_structure), operators=["arcs", "node_type"],
                   arc_blacklist=arc_blacklist, arc_whitelist=arc_whitelist, max_indegree = max_indegree, score=score,seed=seed)
        bn = copy_structure(bn)
    elif bn_type == "Gaussian":
        if score is None:
            score = "bic"
        bn = pb.hc(data, start=get_initial_structure(data, pb.GaussianNetwork,initial_structure), operators=["arcs"], score=score,
                   arc_blacklist=arc_blacklist, arc_whitelist=arc_whitelist, max_indegree=max_indegree, seed=seed)
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
