import math

import networkx as nx
import pandas as pd
import numpy as np
from itertools import combinations
from bayesace.utils import *
from bayesace.algorithms.algorithm import ACE, ACEResult
import multiprocessing as mp

NON_ZERO_CONST = 0.000001
n_processes = np.max((1, int(mp.cpu_count()/1)))


def compute_weight(point_i, point_j, epsilon, weight_function, *args):
    distance = euclidean_distance(point_i, point_j)
    # Directly use the provided weight function to compute the weight
    if distance < epsilon:
        weight = weight_function(point_i, point_j, distance, *args)
        return weight
    return None


def epsilon_weight(point1, point2, distance, epsilon, f_tilde):
    d = len(point1)
    if np.isin(epsilon):
        return distance
    return f_tilde(epsilon ** d / distance) * distance


def kde_weight(point1, point2, distance, density_estimator, f_tilde, variables):
    if f_tilde == neg_log:
        return -log_likelihood(pd.DataFrame([point1 + point2], columns=variables) / 2, density_estimator) * distance
    return f_tilde(density_estimator.likelihood(pd.DataFrame([point1 + point2], columns=variables) / 2)) * distance



def knn_weight(point1, point2, distance, k, n_instances, d, f_tilde):
    volume = np.pi ** (d / 2) / math.gamma(d / 2 + 1)
    r = k / (n_instances * volume)
    return f_tilde(r / distance) * distance


def integral_weight(point1, point2, distance, density_estimator, variables, chunks):
    path_ij = pd.DataFrame(data=np.linspace(point1, point2, chunks), columns=variables)
    return path_likelihood_length(path_ij, density_estimator, penalty=1)


def build_weighted_graph(dataframe: pd.DataFrame, epsilon, weight_function, weight_args=None, parallelize=False):
    graph = nx.Graph()
    mat = dataframe.to_numpy()
    graph.add_nodes_from(dataframe.index)

    combs = combinations(list(dataframe.index), 2)

    if parallelize:
        pool = mp.Pool(n_processes)

        result = pool.starmap(compute_weight, [
            (mat[i[0]], mat[i[1]], epsilon, weight_function, *weight_args)
            for i in combs
        ])

        pool.close()
        pool.join()

        for i, (i_idx, j_idx) in enumerate(combs):
            weight = result[i]
            if weight is not None:
                graph.add_edge(i_idx, j_idx, weight=weight)
    else:
        for i, (i_idx, j_idx) in enumerate(combs):
            weight = compute_weight(mat[i_idx], mat[j_idx], epsilon, weight_function, *weight_args)
            if weight is not None:
                graph.add_edge(i_idx, j_idx, weight=weight)

    return graph


def compute_path(graph, source_node, target_node):
    try:
        shortest_path = nx.shortest_path(graph, source=source_node, target=target_node, weight='weight')
        shortest_distance = nx.shortest_path_length(graph, source=source_node, target=target_node,
                                                    weight='weight')
        return (target_node, {'path': shortest_path, 'distance': shortest_distance})
    # If no path is found, not a problem, just keep iterating
    except nx.NetworkXNoPath:
        return None


def find_closest_paths(graph, source_node, target_nodes, parallelize=False):
    # Use Dijkstra's algorithm to find the shortest paths
    all_shortest_paths = {}

    # First check if all the weights are non negative
    weights = []
    for edge in graph.edges(data=True):
        weights.append(edge[2]["weight"])
    minimum = np.min(weights)

    if minimum < 0:
        for edge in graph.edges(data=True):
            edge[2]["weight"] = edge[2]["weight"] - minimum + NON_ZERO_CONST

    result = []
    if parallelize :
        pool = mp.Pool(n_processes)
        result = pool.starmap_async(compute_path, [(graph, source_node, target_node) for target_node in target_nodes])
        pool.close()
        for i in result.get():
            if i is not None:
                all_shortest_paths[i[0]] = i[1]
    else :
        for target_node in target_nodes :
            path = compute_path(graph, source_node, target_node)
            if path is not None:
                all_shortest_paths[target_node] = path[1]

    # Return weights to its original state
    if minimum < 0:
        for i, edge in enumerate(graph.edges(data=True)):
            edge[2]["weight"] = weights[i]

    return all_shortest_paths


class FACE(ACE):
    def __init__(self, density_estimator, features, chunks, dataset: pd.DataFrame, distance_threshold,
                 graph_type,
                 f_tilde=None, seed=0, verbose=False, log_likelihood_threshold=-np.inf, posterior_probability_threshold=0.50,
                 penalty=1, k=1, parallelize=True):
        assert (list(dataset.columns) == features).all()
        super().__init__(density_estimator, features, chunks, log_likelihood_threshold=log_likelihood_threshold,
                         posterior_probability_threshold=posterior_probability_threshold, penalty=penalty, seed=seed, verbose=verbose,
                         parallelize=parallelize)
        self.dataset = dataset
        self.epsilon = distance_threshold
        self.graph_type = graph_type

        self.f_tilde = f_tilde
        if f_tilde is None:
            self.f_tilde = neg_log
        elif f_tilde == "identity":
            self.f_tilde = identity

        self.weight_function = None
        self.weight_args = None
        # Determine the appropriate weight function and arguments based on the graph type
        if graph_type == "epsilon":
            self.weight_function = epsilon_weight
            self.weight_args = [self.epsilon, self.f_tilde]
        elif graph_type == "kde":
            self.weight_function = kde_weight
            self.weight_args = [density_estimator, self.f_tilde, dataset.columns]
        elif graph_type == "knn":
            self.weight_function = knn_weight
            self.weight_args = [k, dataset.shape[0], dataset.shape[1], self.f_tilde]
        elif graph_type == "integral":
            self.weight_function = integral_weight
            self.weight_args = [density_estimator, features, chunks]
        else:
            raise AttributeError("Invalid graph_type. Expected 'epsilon', 'kde', 'knn', or 'integral'.")

        self.graph = build_weighted_graph(dataset, epsilon=self.epsilon, weight_function=self.weight_function,
                                          weight_args = self.weight_args, parallelize=self.parallelize)
        self.y_pred = self.density_estimator.predict_proba(self.dataset.values, output="pandas")
        self.k = k

    def add_point_to_graph(self, instance: pd.DataFrame):
        assert (instance.columns == self.dataset.columns).all()
        self.graph.add_node(self.dataset.shape[0])
        mat = self.dataset.to_numpy()
        new_point = instance.values[0]

        if self.parallelize :
            pool = mp.Pool(n_processes)

            result = pool.starmap(compute_weight, [
                (new_point, mat[i], self.epsilon, self.weight_function, *self.weight_args) for i in self.dataset.index])

            pool.close()
            for i, weight_i in enumerate(result):
                if weight_i is not None:
                    self.graph.add_edge(self.dataset.shape[0], i, weight=weight_i)
        else :
            for i in self.dataset.index:
                point_i = mat[i]
                weight = compute_weight(point_i, new_point, self.epsilon, self.weight_function, *self.weight_args)
                if weight is not None :
                    self.graph.add_edge(i, len(self.dataset.index), weight=weight)

    def run(self, instance: pd.DataFrame, target_label) -> ACEResult:
        super().run(instance, target_label)
        x_og = instance.drop("class", axis=1)

        # Add node to the graph
        self.add_point_to_graph(x_og)

        # Run algorithm
        # Replace with the user-input node
        source_node = len(self.dataset.index)

        # Mark target nodes based on the posterior probability and likelihood threshold

        # Get likelihood and probability of the class
        logl = log_likelihood(pd.DataFrame(self.dataset, columns=self.features), self.density_estimator)
        #print(self.log_likelihood_threshold)
        #print(logl)
        post_prob = self.density_estimator.posterior_probability(pd.DataFrame(self.dataset, columns=self.features),
                                                                 target_label)
        #print(self.posterior_probability_threshold)
        #print(post_prob)

        # Filter for instances whose likelihood and posterior probability is above the threshold
        mask = (logl > self.log_likelihood_threshold) & (post_prob > self.posterior_probability_threshold)# & (self.dataset["class"] == target_label)
        target_nodes = self.dataset.index[mask]

        #target_nodes = self.dataset.index[self.y_pred[target_label] > self.posterior_probability_threshold]

        if len(target_nodes) > 0:
            true_array = log_likelihood(self.dataset.iloc[target_nodes], self.density_estimator) > self.log_likelihood_threshold
            target_nodes = target_nodes[true_array]

        if len(target_nodes) == 0:
            return ACEResult(None, x_og, np.inf)

        closest_paths = find_closest_paths(self.graph, source_node, target_nodes, parallelize=self.parallelize)
        if len(closest_paths) == 0:
            Warning(
                "No paths found. Perhaps your point was too far from the data or you specified a low distance "
                "threshold epsilon")
            return ACEResult(None, x_og, np.inf)

        min_len = np.inf
        closest_node = None
        for target_node, info in closest_paths.items():
            if info["distance"] < min_len:
                closest_node = target_node
                min_len = info["distance"]

        if self.verbose:
            # Print the resulting graph
            print("Graph edges:")
            print(self.graph.edges())

            # Print the closest paths and distances
            print(f"\nClosest paths from node {source_node} to target nodes:")
            for target_node, info in closest_paths.items():
                print(f"To node {target_node}:")
                print(f"   Path: {info['path']}")
                print(f"   Distance: {info['distance']}")
        path_x = self.dataset.iloc[closest_paths[closest_node]["path"][1:]]
        path_x = pd.concat([x_og, path_x])
        # Change index of original instance to -1
        copy_index = list(path_x.index)
        copy_index[0] = -1
        path_x.index = copy_index

        counterfactual = path_x.iloc[-1]

        # Delete node
        self.graph.remove_node(source_node)

        return ACEResult(counterfactual, path_x, min_len)
