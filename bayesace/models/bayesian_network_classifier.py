import pandas as pd
import pybnesian as pb
import multiprocessing as mp
import numpy as np

from bayesace import ConditionalDE

class PybnesianParallelizationError(Exception):
    pass


def get_initial_structure(data: pd.DataFrame, bn_type, structure_type="naive"):
    class_var_name = data.columns[-1]
    initial = bn_type(data.columns)
    if structure_type == "naive":
        for i in [i for i in data.columns if i != class_var_name]:
            initial.add_arc(class_var_name, i)
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


class BayesianNetworkClassifier(ConditionalDE):
    def __init__(self, network_type="CLG"):
        super().__init__()
        self.bayesian_network: pb.BayesianNetwork = None
        self.class_var_name = None
        self.network_type: str = network_type

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray, initial_structure="naive",
            training_params=None):
        super().fit(X, y)
        if training_params is None:
            training_params = {}
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        dataset = X.copy()
        dataset[self.class_var_name] = y
        bn: pb.BayesianNetwork = None
        if self.network_type == "CLG":
            bn = pb.hc(dataset, start=get_initial_structure(dataset, pb.CLGNetwork, initial_structure),
                       operators=["arcs"], **training_params)
            bn = copy_structure(bn)
        elif self.network_type == "SP":
            # est = MMHC()
            # test = pb.MutualInformation(data, True)
            # bn = pb.MMHC().estimate(hypot_test = test, operators = pb.OperatorPool([pb.ChangeNodeTypeSet(),pb.ArcOperatorSet()]), score = pb.CVLikelihood(data), bn_type = pb.SemiparametricBNType(), patience = 20) #, score = "cv-lik"
            bn = pb.hc(dataset, start=get_initial_structure(dataset, pb.SemiparametricBN, initial_structure),
                       operators=["arcs", "node_type"], **training_params)
            bn = copy_structure(bn)
        elif self.network_type == "Gaussian":
            bn = pb.hc(dataset, start=get_initial_structure(dataset, pb.GaussianNetwork, initial_structure),
                       operators=["arcs"], **training_params)
            bn = copy_structure(bn)
        else:
            raise PybnesianParallelizationError(
                "Only valid types are CLG, SP and Gaussian.")
        bn.fit(dataset)
        bn.include_cpd = True
        pool = mp.Pool(1)
        res = pool.starmap(check_copy, [(bn,)])
        pool.close()
        if not res[0]:
            raise PybnesianParallelizationError(
                "As of version 0.4.3, PyBnesian Bayesian networks have internal and stochastic problems with the method "
                "\"copy()\"."
                "As such, the network is not parallelized correctly and experiments cannot be launched.")
        self.bayesian_network = bn
        self.trained = True

    def logl(self, X: pd.DataFrame, y: pd.Series | np.ndarray = None):
        """
        Compute the log-likelihood for the given data.
        Parameters:
        - data: Features and class labels.
        - class_var_name: name of the class column. Only to respect signature
        Returns:
        - Log-likelihood
        """
        if y is not None:
            if y is pd.Series:
                y = y.to_numpy()
            data = X.copy()
            data[self.class_var_name] = y
            data[self.class_var_name] = data[self.class_var_name].astype('string').astype('category')
            data[self.class_var_name] = data[self.class_var_name].cat.set_categories(self.get_class_labels())
            return self.bayesian_network.logl(data)
        else:
            '''class_cpd = self.bayesian_network.cpd(self.class_var_name)
            class_values = class_cpd.variable_values()
            n_samples = X.shape[0]
            likelihood_val = 0.0
            for v in class_values:
                likelihood_val = likelihood_val + np.e ** self.logl(X, np.repeat(v, n_samples))
            if (likelihood_val > 1).any():
                Warning(
                    "Likelihood of some points in the space is higher than 1.")
            return logl_from_likelihood(likelihood_val)'''
            log_likelihoods = []  # Store log-likelihoods for each class
            for i in self.get_class_distribution().keys():
                log_likelihoods.append(self.logl(X, np.repeat(i, X.shape[0])))

            # Stack log-likelihoods and apply the log-sum-exp trick
            log_likelihoods = np.stack(log_likelihoods, axis=0)  # Shape: (num_classes, num_samples)
            max_log_likelihoods = np.max(log_likelihoods, axis=0)  # Shape: (num_samples,)

            # Log-sum-exp computation
            lls = max_log_likelihoods + np.log(np.sum(np.exp(log_likelihoods - max_log_likelihoods), axis=0))

            return lls

    def predict_proba(self, X: np.ndarray, output="numpy") -> np.ndarray | pd.DataFrame:
        """
        Compute posterior probabilities P(Y|X).
        Parameters:
        - X: Features of shape (n_samples, n_features).
        Returns:
        - Posterior probabilities as a 2D array of shape (n_samples, n_classes).
        """
        # Get the possible class labels from the CPD
        class_cpd = self.bayesian_network.cpd(self.class_var_name)
        class_values = class_cpd.variable_values()
        X_df = pd.DataFrame(X, columns=self.columns)
        P_xY = np.zeros((X.shape[0], len(class_values)))
        for i, class_value in enumerate(class_values):
            X_df[self.class_var_name] = pd.Categorical([class_value] * X.shape[0], categories=class_values)
            P_xY[:, i] = np.e ** self.bayesian_network.logl(X_df)
        P_x = np.sum(P_xY, axis=1)
        P_x_given_Y = P_xY
        P_x_given_Y[P_x == 0] = 1 / len(class_values)
        P_x[P_x == 0] = 1
        P_Y_given_x = P_x_given_Y / P_x[:, None]
        if output == "pandas":
            return pd.DataFrame(P_Y_given_x, columns=class_values)
        return P_Y_given_x

    def get_class_labels(self):
        return self.bayesian_network.cpd(self.class_var_name).variable_values()

    def sample(self, n_samples, seed=None) -> pd.DataFrame:
        return self.bayesian_network.sample(n_samples, ordered=True, seed=seed).to_pandas()

    def fitted(self):
        return self.bayesian_network.fitted()

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.bayesian_network.include_cpd = True
