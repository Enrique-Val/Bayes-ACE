import networkx as nx
import pandas as pd
import numpy as np
from itertools import combinations
from bayesace.utils import *
from numba import jit


def epsilon_weight_function(point1, point2, epsilon, f_tilde):
    d = len(point1)
    dist = euclidean_distance(point1, point2)
    return f_tilde(epsilon ** d / dist) * dist


def kde_weight_function(point1, point2, bn, f_tilde, variables):
    dist = euclidean_distance(point1, point2)
    return f_tilde(likelihood(pd.DataFrame([point1 + point2], columns=variables) / 2, bn)) * dist


def build_weighted_graph(dataframe : pd.DataFrame, graph_type, epsilon=0.25, f_tilde=None, bn=None, chunks=2):
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
        for j in range(i+1, mat.shape[0]) :
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
                    raise AttributeError("Parameter \"graph_type\" should take value \"epsilon\", \"kde\" or \"integral\"")
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


# Example weight function (you can customize this based on your requirements)
def custom_weight_function(point1, point2):
    # Example: Weight is the sum of X and Y coordinates of the two points
    return sum(point1) + sum(point2)


# Example target node function (you can customize this based on your requirements)
def is_target_node(row):
    # Example: Nodes where the X coordinate is greater than 2 are target nodes
    return row['X'] > 0.2


def get_target_nodes(dataset, y_pred, y_instance, confidence_threshold):
    print()
    return dataset.index[y_pred.transpose()[y_instance] < confidence_threshold]


def face_algorithm(dataset, y_pred, instance, y_instance, graph_type, distance_threshold=0.25,
                   confidence_threshold=0.25, bn=None, f_tilde=None, chunks=2, verbose=True):

    resulting_graph = build_weighted_graph(dataset,
                                           graph_type, epsilon=distance_threshold, bn=bn, f_tilde=f_tilde,
                                           chunks=chunks)

    # Replace with the user-input node
    source_node = instance

    # Mark target nodes based on the provided function
    target_nodes = get_target_nodes(dataset, y_pred, y_instance, confidence_threshold)

    print(target_nodes)

    closest_paths = find_closest_paths(resulting_graph, source_node, target_nodes)

    min_len = np.inf
    closest_node = None
    for target_node, info in closest_paths.items():
        if info["distance"] < min_len:
            closest_node = target_node
            min_len = info["distance"]

    if verbose:
        # Print the resulting graph
        print("Graph edges:")
        print(resulting_graph.edges())

        # Print nodes and their target_node attribute
        print("\nNodes and their target_node attribute:")
        for node, attributes in resulting_graph.nodes(data=True):
            print(f"Node {node}, target_node: {attributes.get('target_node', False)}")

        # Print the closest paths and distances
        print(f"\nClosest paths from node {source_node} to target nodes:")
        for target_node, info in closest_paths.items():
            print(f"To node {target_node}:")
            print(f"   Path: {info['path']}")
            print(f"   Distance: {info['distance']}")

    return closest_node

# Example usage:
# Assuming your DataFrame has columns 'X' and 'Y' for coordinates
# df = pd.DataFrame({'X': [0, 1, 2, 3, 4], 'Y': [0, 1, 2, 3, 4]})
# df = df*0.1
# res = face_algorithm(df, 0, None)
# print(res)
