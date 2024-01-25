import networkx as nx
import pandas as pd
import numpy as np
from itertools import combinations
from bayesace.utils import *
from bayesace.algorithms.algorithm import ACE, ACEResult


def build_weighted_graph(dataframe: pd.DataFrame, graph_type, epsilon=0.25, f_tilde=None, bn=None, chunks=2):
    # Initial checks
    if f_tilde is None:
        f_tilde = identity

    # Create an undirected graph
    graph = nx.Graph()

    mat = dataframe.to_numpy()

    # Add nodes to the graph
    graph.add_nodes_from(dataframe.index)

    # Connect nodes if their Euclidean distance is below the threshold
    for i in range(mat.shape[0]):
        for j in range(i + 1, mat.shape[0]):
            point1 = mat[i]
            point2 = mat[j]
            distance = euclidean_distance(point1, point2)
            if distance < epsilon:
                # Calculate the weight based on the provided function
                weight = None
                if graph_type == "epsilon":
                    weight = epsilon_weight_function(point1, point2, epsilon, f_tilde)
                elif graph_type == "kde":
                    weight = kde_weight_function(point1, point2, bn, f_tilde, variables=dataframe.columns)
                elif graph_type == "integral":
                    path_ij = pd.DataFrame(data=straight_path(point1, point2, chunks), columns=dataframe.columns)
                    weight = path_likelihood_length(path_ij, bn, penalty=1)
                else:
                    raise AttributeError(
                        "Parameter \"graph_type\" should take value \"epsilon\", \"kde\" or \"integral\"")
                graph.add_edge(i, j, weight=weight)
    return graph


def find_closest_paths(graph, source_node, target_nodes):
    # Use Dijkstra's algorithm to find the shortest paths
    all_shortest_paths = {}
    for target_node in target_nodes:
        try:
            shortest_path = nx.shortest_path(graph, source=source_node, target=target_node, weight='weight')
            shortest_distance = nx.shortest_path_length(graph, source=source_node, target=target_node,
                                                        weight='weight')
            all_shortest_paths[target_node] = {'path': shortest_path, 'distance': shortest_distance}
        # If no path is found, not a problem, just keep iterating
        except nx.NetworkXNoPath:
            pass

    return all_shortest_paths


class FACE(ACE):
    def __init__(self, bayesian_network, features, penalty, chunks, dataset: pd.DataFrame, distance_threshold,
                 graph_type,
                 f_tilde=identity, seed=0, verbose=True, likelihood_threshold=0.00, accuracy_threshold=0.50):
        super().__init__(bayesian_network, features, penalty, chunks, likelihood_threshold=likelihood_threshold,
                         accuracy_threshold=accuracy_threshold, seed=seed, verbose=verbose)
        self.dataset = dataset
        self.epsilon = distance_threshold
        self.graph_type = graph_type
        self.f_tilde = f_tilde
        self.graph = build_weighted_graph(dataset, graph_type, epsilon=self.epsilon, f_tilde=f_tilde,
                                          bn=self.bayesian_network, chunks=self.chunks)
        self.y_pred = predict_class(self.dataset, self.bayesian_network)

    def add_point_to_graph(self, instance: pd.DataFrame):
        assert (instance.columns == self.dataset.columns).all()
        self.graph.add_node(len(self.dataset.index))
        mat = self.dataset.to_numpy()
        new_point = instance.values[0]

        for i in range(mat.shape[0]):
            point = mat[i]
            distance = euclidean_distance(new_point, point)
            if distance < self.epsilon:
                # Calculate the weight based on the provided function
                weight = None
                if self.graph_type == "epsilon":
                    weight = epsilon_weight_function(new_point, point, self.epsilon, self.f_tilde)
                elif self.graph_type == "kde":
                    weight = kde_weight_function(new_point, point, self.bayesian_network, self.f_tilde,
                                                 variables=self.features)
                elif self.graph_type == "integral":
                    path_ij = pd.DataFrame(data=straight_path(new_point, point, self.chunks), columns=self.features)
                    weight = path_likelihood_length(path_ij, self.bayesian_network, penalty=1)
                else:
                    raise AttributeError(
                        "Parameter \"graph_type\" should take value \"epsilon\", \"kde\" or \"integral\"")
                self.graph.add_edge(i, len(self.dataset.index), weight=weight)

    def run(self, instance: pd.DataFrame):
        x_og = instance.drop("class", axis=1)
        y_og = instance["class"].values[0]

        # Add node to the graph
        self.add_point_to_graph(x_og)

        # Run algorithm
        # Replace with the user-input node
        source_node = len(self.dataset.index)

        # Mark target nodes based on the provided function
        target_nodes = self.dataset.index[self.y_pred[y_og] < self.accuracy_threshold]

        closest_paths = find_closest_paths(self.graph, source_node, target_nodes)

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

        print("Llega")

        # Delete node
        self.graph.remove_node(source_node)

        return ACEResult(counterfactual, path_x, min_len)


def epsilon_weight_function(point1, point2, epsilon, f_tilde):
    d = len(point1)
    dist = euclidean_distance(point1, point2)
    if not dist > 0:
        return 0.0000001
    return f_tilde(epsilon ** d / dist) * dist


def kde_weight_function(point1, point2, bn, f_tilde, variables):
    dist = euclidean_distance(point1, point2)
    return f_tilde(likelihood(pd.DataFrame([point1 + point2], columns=variables) / 2, bn)) * dist




