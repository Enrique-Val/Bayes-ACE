import networkx as nx
import pandas as pd
import numpy as np
import math
from itertools import combinations

def build_weighted_graph(dataframe, coordinate_columns, weight_function, distance_threshold):
    # Create an undirected graph
    graph = nx.Graph()

    # Add nodes to the graph
    graph.add_nodes_from(dataframe.index)

    # Function to calculate Euclidean distance between two points
    def euclidean_distance(point1, point2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

    print(coordinate_columns)
    # Connect nodes if their Euclidean distance is below the threshold
    for (node1, point1), (node2, point2) in combinations(dataframe[coordinate_columns].iterrows(), 2):
        distance = euclidean_distance(point1, point2)
        if distance < distance_threshold:
            # Calculate the weight based on the provided function
            if weight_function is None:
                graph.add_edge(node1, node2, weight=1)
            else:
                weight = weight_function(point1, point2)
                graph.add_edge(node1, node2, weight=weight)

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


def epsilon_weight_function(point1, point2):
    eps = 0.25
    d = len(point1)
    euc_dist = np.linalg.norm(point1 - point2)
    return ((eps ** d) / euc_dist) * euc_dist




# Example target node function (you can customize this based on your requirements)
def is_target_node(row):
    # Example: Nodes where the X coordinate is greater than 2 are target nodes
    return row['X'] > 0.2


def get_target_nodes(dataset, y_pred, y_instance, confidence_threshold):
    return dataset.index[y_pred[y_instance] < confidence_threshold]


def face_algorithm(dataset, y_pred, instance, y_instance, weight_function=None, distance_threshold=0.25,
                   confidence_threshold=0.25, verbose=True):
    coordinate_columns = dataset.columns

    resulting_graph = build_weighted_graph(dataset, coordinate_columns, weight_function,
                                           distance_threshold)

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
