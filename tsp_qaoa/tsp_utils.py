"""
Contains functions for handling instances of the travelling salesman problem
 (TSP) and retrieving their details:
- Loading a TSP
- Creating a TSP instance
- Calculating the distance matrix
- Getting the distance matrix
- Getting the number of cities
- Converting the binary solution from a np array to a str
- Creating intial seed permutation for the whole (large) TSP by expanding
   solution for clustered (small) TSP

Lara Stroh and Dan Forbes - 2023
"""

from itertools import combinations
from functools import reduce

import numpy as np
import numpy.typing as npt
import networkx as nx
from qiskit_optimization.applications import Tsp
from python_tsp.distances import great_circle_distance_matrix


def load_tsp_problem(filename: str, weight_scale: float = 1.5) -> Tsp:
    """
    Loads the .tsp problem, converts it into the format of the TSP class in
     Qiskit, and modifies the weights of the edges by weight_scale.
    It returns the TSP instance.

    Arguments
    ----------
    filename:
        A string specifying the name of the file containing the travelling
        salesman problem (TSP).

    weight_scale:
        The float number specifying the factor by which the edges' weights get
        multiplied (for better road approximation).

    Returns
    ----------
    Tsp:
        The given TSP as an instance in the format of the TSP class by Qiskit.

    Notes
    ----------
    The file containing the TSP needs to be written in the TSPLIB format.

    The distances between the cities are multiplied by a factor so that the
     weights give a better road approximation than the Euclidean distances.
    """
    # Load from file (needs to be of TSPLIB format)
    tsp = Tsp.parse_tsplib_format(filename)

    # Update weights of edges inplace
    for u, v, data in tsp.graph.edges(data=True):
        data['weight'] *= weight_scale

    return tsp


def create_tsp_problem(
        locations: npt.NDArray[np.float64],
        distance_matrix: npt.NDArray[np.float64]
        ) -> Tsp:
    """
    Creates a TSP instance from the given locations and distance matrix.

    Arguments
    ----------
    locations:
        A 2D numpy array containing the coordinates of the cities.
        E.g. (latitude, longitude)

    distance_matrix:
        A 2D numpy array containing the distances between the cities.

    Returns
    ----------
    Tsp:
        The given TSP as an instance in the format of the TSP class by Qiskit.
    """
    # Create an empty graph
    graph = nx.Graph()

    # Add nodes with positions of the cluster centres
    graph.add_nodes_from(
        (i, {"pos": centre}) for i, centre in enumerate(locations))

    # Create an iterable of (u, v, w) tuples
    edges = (
        (i, j, distance_matrix[i, j])
        for i, j in combinations(range(len(distance_matrix)), 2))

    # Add edges with weights from the distance matrix
    graph.add_weighted_edges_from(edges)

    return Tsp(graph)


def calc_distance_matrix(
        latlon_locations: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
    """
    Calculates the distance matrix from the given latitude-longitude locations,
     returning the results in km.

    This should be replaced in the future by a method which calculates accurate
     distances from e.g. google maps API.

    This matrix is also known as an adjacency matrix.

    Arguments
    ----------
    latlon_locations:
        A 2D numpy ndarray with col 0 containing lats,
        and col 1 containing longs.

    Returns
    ----------
        A 2D np.ndarray containing np.float64 numbers:
        The distance matrix giving the distances (in km) for the routes
        between each of the cities.
    """
    distance_matrix = great_circle_distance_matrix(latlon_locations)
    distance_matrix /= 1000  # Convert to km
    return distance_matrix


def get_distance_matrix(tsp: Tsp) -> npt.NDArray[np.float64]:
    """
    Returns the distance matrix of the TSP.

    The distance matrix represents the weights of the edges between the nodes
     (cities).

    Arguments
    ----------
    tsp:
        The TSP instance given in the format of the TSP class by Qiskit.

    Returns
    ----------
    np.NDArray containing np.float64 numbers:
        The distance matrix giving the weights between the cities.
    """
    return nx.to_numpy_array(tsp.graph, dtype=np.float64)


def get_number_cities(tsp: Tsp) -> int:
    """
    Returns the number of cities in the given TSP instance.

    Arguments
    ----------
    tsp:
        The TSP instance given in the format of the TSP class by Qiskit.

    Returns
    ----------
    int:
        The integer which specifies the number of cities in the TSP instance.
    """
    return len(tsp.graph.nodes)


def format_x_array(x_array: np.ndarray) -> str:
    """
    Converts binary solution stored in x_array from a numpy array,
     e.g. [0., 1., 0., 1.] into a str e.g. '0101'.

    Arguments
    ----------
    np.ndarray:
        A numpy array which stores the binary solution.

    Returns
    ----------
    str:
        A string which stores the binary solution.
    """
    x_int = x_array.astype(np.uint8)
    return reduce(lambda x, y: x + str(y), x_int, "")


def create_inital_permutation(
        cluster_solution: str,
        n_clusters: int,
        location_ids: npt.NDArray[np.uint64],
        cluster_labels: npt.NDArray[np.uint64]
        ) -> npt.NDArray[np.uint64]:
    """
    Returns an intial permutation for the whole TSP instance, i.e., it expands
     the solution given for the cluster to the larger problem. The order of the
     clusters is kept and the cities within the clusters get randomly ordered.

    Arguments
    ----------
    cluster_solution:
        A string which gives the binary solution of the clustered TSP instance.

    n_clusters:
        An integer which specifies the number of clusters.

    location_ids:
        A np.ndarray which contains the unique identifiers of the locations.

    cluster_labels:
        A np.ndarray which contains the labels of the clusters for each
        location which identifies for each city to which cluster it belongs to.

    Returns
    ----------
    np.NDArray containing np.int64 numbers:
        The initial permutation for the large TSP instance obtained by
         expanding the solution for the clustered TSP instance.
    """
    # Convert string to array, with the timeline for each city as a row
    bitstring_array = np.array(
        [int(s) for s in cluster_solution], dtype=np.bool_
        ).reshape(-1, n_clusters)
    print(bitstring_array)

    # Sort the clusters by the order in which they appear in the bitstring
    # This only works for bitstrings with a single 1 in each row and column
    assert (sum := bitstring_array.sum()) == n_clusters, (
        f"Bitstring has an incorrect number of 1s, {sum} != {n_clusters}")

    clusters, order = np.nonzero(bitstring_array)
    sorted_clusters = clusters[np.argsort(order)]

    # Create a permutation from the sorted clusters
    initial_permutation = np.empty_like(location_ids)
    start = 0

    # Go through each cluster in the sorted order, shuffle the locations inside
    #  each one randomly, and add the locations to the permutation
    for cluster in sorted_clusters:
        cluster_locations = location_ids[cluster_labels == cluster]
        np.random.shuffle(cluster_locations)
        end = start + len(cluster_locations)
        initial_permutation[start:end] = cluster_locations
        start = end

    return initial_permutation
