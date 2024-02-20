"""
Contains functions for plotting the graph for a travelling salesman problem
 (TSP) and for creating bar plots for the solutions and their associated costs.

Lara Stroh - 2023
"""

from operator import itemgetter

import networkx as nx
import matplotlib.pyplot as plt
from qiskit_optimization.applications import Tsp


def draw_tsp_graph(tsp: Tsp, node_colors: str):
    """
    Visualizes the TSP as a graph.

    Arguments
    ----------
    tsp:
        The TSP instance given in the format of the TSP class by Qiskit.

    node_colors:
        A string specifying the color of the nodes in the graph.
    """
    # Get graph information from Tsp
    graph = tsp.graph
    node_pos = graph.nodes(data="pos")  # type: ignore

    # Create figure frame
    fig_axes = plt.axes(frameon=True)

    # Draw Graph
    nx.draw_networkx(
        graph,
        node_color=node_colors,
        node_size=600,
        ax=fig_axes,
        pos=node_pos)

    # Modify edges and round their weights
    edge_weights = nx.get_edge_attributes(graph, "weight")

    for edge, edge_value in edge_weights.items():
        edge_weights[edge] = round(edge_value, 3)

    nx.draw_networkx_edge_labels(
        graph,
        pos=node_pos,
        edge_labels=edge_weights)

    plt.show()


def draw_probabilities_bar_plot(
        solutions_set: dict[str, int],
        n_display: int | None = None,
        n_shots: int = 1024):
    """
    Visualizes the solutions given by the circuit in a bar plot in descending
     order of occurrence probabilities.

    Arguments
    ----------
    solutions_set:
        A dictionary containing the returned solutions from the circuit as
        keys given as strings and the number of their occurrences as values
        given as integers. It is sorted to display the items in descending
        order with regards to the occurrences.

    n_display:
        An integer specifying how many of the solutions should be displayed.
        If None is passed, show all data.

    n_shots:
        An integer which specifies how often the quantum circuit is run. The
        default value, if none is given, is 1024.
    """
    if n_display is None:
        n_display = len(solutions_set)

    elif n_display > len(solutions_set):
        raise ValueError(
            ("The number of solutions to be displayed must be "
             f"<= {len(solutions_set)}, not {n_display}."))

    # Create list of the solutions
    solutions = [bitstring for bitstring in solutions_set.keys()]

    # Create list of the occurrence probabilities
    probabilities = [
        occurrences/n_shots for occurrences in solutions_set.values()]

    plt.bar(range(n_display), probabilities[:n_display], width=0.7)
    plt.xticks(range(n_display), solutions[:n_display], rotation='vertical')

    plt.show()


def draw_costs_bar_plot(
        solutions_costs: dict[str, float],
        n_display: int | None = None):
    """
    Visualizes the solutions given by the circuit and their associated costs
     in a bar plot, ordered by ascending cost values.

    Arguments
    ----------
    solutions_set:
        A dictionary containing the returned solutions from the circuit as
        keys given as strings and the costs associated with them as values.

    n_display:
        An integer specifying how many of the solutions should be displayed.
        If None is passed, show all data.
    """
    if n_display is None:
        n_display = len(solutions_costs)

    elif n_display > len(solutions_costs):
        raise ValueError(
            ("The number of solutions to be displayed must be "
             f"<= {len(solutions_costs)}, not {n_display}."))

    # Sort dictionary of solutions and costs in ascending order of the costs
    sorted_solutions_costs = dict(
        sorted(solutions_costs.items(), key=itemgetter(1)))

    # Create separate tuples of the solutions and costs
    solutions, cost_values = zip(*sorted_solutions_costs.items())

    plt.bar(range(n_display), cost_values[:n_display], width=0.7)
    plt.xticks(range(n_display), solutions[:n_display], rotation='vertical')

    plt.show()
