"""
Contains functions for ordering binary solutions to the travelling salesman
 problem (TSP) and for calculating the cost associated with a given solution.

Lara Stroh - 2023
"""

import numpy as np


def counts_to_binary_result(
        counts: dict[str, int]
        ) -> dict[tuple[int, ...], int]:
    """
    Reverses the order of the binary solutions given by running and
     measuring a qiskit.QuantumCircuit. Converts the keys from a string to a
     tuple of integers.

    E.g. "1100" -> (0, 0, 1, 1).

    We need this function since Qiskit's ordering of the bits in the solutions
     is reversed compared to the order used in the encoding of the solutions.

    Arguments
    ----------
    counts:
        A dictionary with strings as keys and integers as values where the
        keys present the measured binary solutions and the values specify how
        often these solutions were measured.

    Returns
    ----------
    dict[tuple[int, ...], int]:
        The dictionary's keys (tuples of integers) are the binary solutions
        with the bits in the same order as in the encodings of the solutions.
        Their values specify the number of times the specific solution was
        measured.
    """
    return {
        tuple(map(int, key[::-1])): value
        for key, value in counts.items()}


def calculation_cost_Hamiltonian(
        TSP_distance_matrix: np.ndarray,
        number_cities: int,
        penalty: float,
        binary_solution: tuple[int, ...] | str
        ) -> float:
    """
    Returns the calculated value of the cost Hamiltonian for a certain
     binary solution.

    Arguments
    ----------
    TSP_distance_matrix:
        A np.ndarray which contains the weights of the edges between the nodes
        (cities).

    number_cities:
        An integer which specifies the number of cities.

    penalty:
        A float which specifies the factor by which the penalty terms of the
        constraints get multiplied.

    binary_solution:
        A tuple of integers or a string describing a binary solution where
        each integer/character specifies one bit of the solution.

    Returns
    ----------
    float:
        The float gives the calculated cost for the given binary solution.

    Note
    ----------
    The binary_solution needs to consist of 0s and 1s only.
    If it is a string, then the characters get converted into integers.
    """
    # Convert binary_solution to correct format e.g. "101" -> (1, 0, 1)
    if isinstance(binary_solution, str):
        try:
            binary_solution = tuple(map(int, binary_solution))
        except Exception:
            raise ValueError(
                ("Given binary_solution cannot be converted to tuple."
                 "It should have the form e.g. '101' or (1, 0, 1)"))

    cost_Hamiltonian: float = 0.

    for start_city in range(number_cities):

        # Add the objective function to the cost Hamiltonian
        for next_city in range(0, number_cities):
            weight: float = TSP_distance_matrix[start_city, next_city]

            for time_step in range(number_cities):
                # If the weight is not equal to 0 there exists a route
                if weight != 0:
                    # Define the qubits corresponding to the binary
                    # variables for each step from one to the next city
                    qubit1 = (
                        start_city * number_cities + time_step)
                    qubit2 = (
                        next_city * number_cities
                        + ((time_step+1) % number_cities))

                    # Add interaction term to cost_Hamiltonian
                    cost_Hamiltonian += (
                        weight
                        * binary_solution[qubit1]
                        * binary_solution[qubit2])

        # Include the penalty terms for visiting a city more than once
        visits_to_city: int = 0

        for time_step in range(number_cities):
            # Define the qubit corresponding to the current city and time step
            repetition_penalty_qubit = (
                start_city * number_cities + time_step)

            # Add binary value to the total number of visits to this city
            visits_to_city += binary_solution[repetition_penalty_qubit]

        # Add the penalty of no/repeated visits to a city to cost_Hamiltonian
        cost_Hamiltonian += penalty * (1 - visits_to_city)**2

    # Include the penalty terms for visiting multiple cities at the same time
    first_city: int = 0

    for time_step in range(number_cities):
        locations_same_time: int = 0

        # Define the qubit corresponding to the first city at the current time
        bilocation_penalty_qubit1 = time_step

        # Add binary value to the total number of cities visited at this time
        locations_same_time += binary_solution[
            bilocation_penalty_qubit1]

        # Go through the other cities at that time step
        for next_city in range(first_city + 1, number_cities):
            bilocation_penalty_qubit2 = (
                next_city * number_cities + time_step)
            locations_same_time += binary_solution[
                bilocation_penalty_qubit2]

        # Add penalty of multiple cities at the same time to cost_Hamiltonian
        cost_Hamiltonian += penalty * (1 - locations_same_time)**2

    return cost_Hamiltonian
