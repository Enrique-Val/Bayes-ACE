import networkx as nx
import pandas as pd
import numpy as np
from itertools import combinations
from bayesace.utils import *
from bayesace.algorithms.algorithm import ACE, ACEResult
import multiprocessing as mp


NON_ZERO_CONST = 0.000001

def compute_weight_and_add_edge(variables, i, j, point_i, point_j, graph_type, epsilon, f_tilde, bn, chunks):
    distance = euclidean_distance(point_i, point_j)
    if distance < epsilon:
        # Calculate the weight based on the provided function
        weight = None
        if graph_type == "epsilon":
            weight = epsilon_weight_function(point_i, point_j, epsilon, f_tilde)
        elif graph_type == "kde":
            weight = kde_weight_function(point_i, point_j, bn, f_tilde, variables=variables)[0]
        elif graph_type == "integral":
            path_ij = pd.DataFrame(data=straight_path(point_i, point_j, chunks), columns=variables)
            weight = path_likelihood_length(path_ij, bn, penalty=1)
        else:
            raise AttributeError(
                "Parameter \"graph_type\" should take value \"epsilon\", \"kde\" or \"integral\"")
        return (i, j, weight)


def build_weighted_graph(dataframe: pd.DataFrame, graph_type, epsilon=0.25, f_tilde=None, bn=None, chunks=2):

    # Create an undirected graph
    graph = nx.Graph()

    mat = dataframe.to_numpy()

    # Add nodes to the graph
    graph.add_nodes_from(dataframe.index)

    combs = combinations(list(dataframe.index), 2)
    # Connect nodes if their Euclidean distance is below the threshold
    pool = mp.Pool(mp.cpu_count())

    result = pool.starmap_async(compute_weight_and_add_edge, [
        (dataframe.columns, i[0], i[1], mat[i[0]], mat[i[1]], graph_type, epsilon, f_tilde, bn,
         chunks) for i in combs])

    pool.close()
    for i in result.get():
        if i is not None :
            graph.add_edge(i[0], i[1], weight=i[2])
    return graph

def compute_path(graph, source_node, target_node) :
    try:
        shortest_path = nx.shortest_path(graph, source=source_node, target=target_node, weight='weight')
        shortest_distance = nx.shortest_path_length(graph, source=source_node, target=target_node,
                                                    weight='weight')
        return (target_node, {'path': shortest_path, 'distance': shortest_distance})
    # If no path is found, not a problem, just keep iterating
    except nx.NetworkXNoPath:
        return None


def find_closest_paths(graph, source_node, target_nodes):
    # Use Dijkstra's algorithm to find the shortest paths
    all_shortest_paths = {}

    # First check if all the weights are non negative
    weights = []
    for edge in graph.edges(data=True) :
        weights.append(edge[2]["weight"])
    minimum = np.min(weights)

    if minimum < 0 :
        for edge in graph.edges(data=True):
            edge[2]["weight"] = edge[2]["weight"]-minimum + NON_ZERO_CONST

    pool = mp.Pool(mp.cpu_count())
    result = pool.starmap_async(compute_path, [(graph, source_node, target_node) for target_node in target_nodes])
    pool.close()
    for i in result.get():
        if i is not None:
            all_shortest_paths[i[0]] = i[1]

    # Return weights to its original state
    if minimum < 0 :
        for i,edge in enumerate(graph.edges(data=True)):
            edge[2]["weight"] = weights[i]

    return all_shortest_paths


class FACE(ACE):
    def __init__(self, bayesian_network, features, chunks, dataset: pd.DataFrame, distance_threshold,
                 graph_type,
                 f_tilde=None, seed=0, verbose=False, likelihood_threshold=0.00, accuracy_threshold=0.50,
                 penalty=1):
        super().__init__(bayesian_network, features, chunks, likelihood_threshold=likelihood_threshold,
                         accuracy_threshold=accuracy_threshold, penalty=penalty, seed=seed, verbose=verbose)
        self.dataset = dataset
        self.epsilon = distance_threshold
        self.graph_type = graph_type
        self.f_tilde = f_tilde
        if f_tilde is None:
            self.f_tilde = neg_log
        self.graph= build_weighted_graph(dataset, graph_type, epsilon=self.epsilon, f_tilde=self.f_tilde,
                                          bn=self.bayesian_network, chunks=self.chunks)
        self.y_pred = predict_class(self.dataset, self.bayesian_network)

    def add_point_to_graph(self, instance: pd.DataFrame):
        assert (instance.columns == self.dataset.columns).all()
        self.graph.add_node(len(self.dataset.index))
        mat = self.dataset.to_numpy()
        new_point = instance.values[0]

        pool = mp.Pool(mp.cpu_count())

        result = pool.starmap_async(compute_weight_and_add_edge, [
            (self.dataset.columns, i, len(self.dataset.index), mat[i], new_point, self.graph_type, self.epsilon, self.f_tilde, self.bayesian_network,
             self.chunks) for i in self.dataset.index])

        pool.close()
        for i in result.get():
            if i is not None:
                self.graph.add_edge(i[0], i[1], weight=i[2])

        '''for i in range(mat.shape[0]):
            point_i = mat[i]
            tuple = compute_weight_and_add_edge(self.graph, self.dataset.columns, i, len(self.dataset.index), point_i,
                                        new_point, self.graph_type, self.epsilon, self.f_tilde, self.bayesian_network,
                                        self.chunks)
            if tuple is not None :
                self.graph.add_edge(i, len(self.dataset.index), weight = tuple[2])'''

    def run(self, instance: pd.DataFrame):
        x_og = instance.drop("class", axis=1)
        y_og = instance["class"].values[0]

        # Add node to the graph
        self.add_point_to_graph(x_og)

        # Run algorithm
        # Replace with the user-input node
        source_node = len(self.dataset.index)

        # Mark target nodes based on the accuracy and likelihood threshold
        target_nodes = self.dataset.index[self.y_pred[y_og] < self.accuracy_threshold]

        if len(target_nodes) > 0:
            true_array = likelihood(self.dataset.iloc[target_nodes], self.bayesian_network) > self.likelihood_threshold
            target_nodes = target_nodes[true_array]

        if len(target_nodes) == 0:
            return ACEResult(None, x_og, np.inf)

        closest_paths = find_closest_paths(self.graph, source_node, target_nodes)
        if len(closest_paths) == 0:
            Warning("No paths found. Perhaps your point was too far from the data or you specified a low distance threshold")
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

        counterfactual = path_x.iloc[-1]

        # Delete node
        self.graph.remove_node(source_node)

        return ACEResult(counterfactual, path_x, min_len)


def epsilon_weight_function(point1, point2, epsilon, f_tilde):
    d = len(point1)
    dist = euclidean_distance(point1, point2)
    if not dist > 0:
        return NON_ZERO_CONST
    return f_tilde(epsilon ** d / dist) * dist


def kde_weight_function(point1, point2, bn, f_tilde, variables):
    dist = euclidean_distance(point1, point2)
    return f_tilde(likelihood(pd.DataFrame([point1 + point2], columns=variables) / 2, bn)) * dist
