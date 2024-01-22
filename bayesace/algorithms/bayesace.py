from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from bayesace.utils import *


def bayes_ace(bayesian_network, instance, n_vertex=1, penalty=1):
    class BestPathFinder(ElementwiseProblem):
        def __init__(self, bayesian_network, instance, n_vertex=1, penalty=1):
            n_features = (len(instance.columns) - 1)
            super().__init__(n_var=n_features * (n_vertex + 1),
                             n_obj=1,
                             n_ieq_constr=2,
                             xl=np.array([-2] * (n_features * (n_vertex + 1))),
                             xu=np.array([2] * (n_features * (n_vertex + 1))))
            self.x_og = instance.drop("class", axis=1)
            self.y_og = "a"  # instance["class"]
            self.n_vertex = n_vertex
            self.penalty = penalty
            self.n_features = n_features
            self.bayesian_network = bayesian_network

        def _evaluate(self, x, out, *args, **kwargs):
            df_vertex = pd.DataFrame(columns=self.x_og.columns,
                                     data=np.resize(x, new_shape=(self.n_vertex + 1, self.n_features)))
            df_vertex = pd.concat([self.x_og, df_vertex])
            df_vertex = df_vertex.reset_index()
            path_x = path(df_vertex)
            likelihood_path = (-log_likelihood(path_x, self.bayesian_network) + 1) ** self.penalty
            f1 = np.sum(likelihood_path)
            out["F"] = np.column_stack([f1])

            x_cfx = self.x_og.copy()
            x_cfx[:] = x[-self.n_features:]
            # print(accuracy(self.x_cfx, self.y_og, self.bayesian_network))
            g1 = accuracy(x_cfx, self.y_og, self.bayesian_network) - 0.05  # -likelihood(x_cfx, learned)+0.0000001
            g2 = -0.1  # likelihood(x_cfx, self.bayesian_network)+0.005
            out["G"] = np.column_stack([g1, g2])

    problem = BestPathFinder(bayesian_network=bayesian_network, instance=instance, n_vertex=n_vertex, penalty=penalty)
    algorithm = NSGA2(pop_size=200)

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 10),
                   seed=1,
                   verbose=True)

    print(res.X)
    print(res.F)

    return res.X
