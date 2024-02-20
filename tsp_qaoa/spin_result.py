"""
Contains functions which calculate the cost in terms of spin variables for
 given bitstring solutions.

Lara Stroh - 2023
"""

from itertools import combinations, chain, islice

import numpy as np
from qiskit_optimization.applications import Tsp

from . import tsp_utils
from . import quantum_circuits


def values_spins(spin_list: list[int]) -> list[int]:
    """
    Takes a bitstring given in a list and changes the bit values into spin
     values in place.

    Arguments
    ----------
    spin_list:
        A list of integers which specifies the bitstring solution.

    Returns
    ----------
    list[int]:
        The list of integers that has been modified in place is returned, now
        containing spin values instead of bit values.
    """
    # Replace in place all the bit values in the list with spin values
    for element in range(len(spin_list)):
        if spin_list[element] == 0:
            spin_list[element] = 1
        elif spin_list[element] == 1:
            spin_list[element] = -1

    return spin_list


def cost_spin_solution(
        distance_matrix: np.ndarray,
        number_cities: int,
        penalty_factor: float,
        bitstring_solution: list[int] | str
        ) -> float:
    """
    Calculates the cost of a given bitstring using spin variables.

    Arguments
    ----------
    distance_matrix:
        A np.ndarray which contains the weights of the edges between the nodes
        (cities).

    number_cities:
        An integer which specifies the number of cities.

    penalty_factor:
        A float which specifies the factor by which the penalty terms of the
        constraints get multiplied.

    bitstring_solution:
        A list of integers or a string describing a binary solution where
        each integer/character specifies one bit of the solution.

    Returns
    ----------
    float:
        The float returns the calculated cost for the given bitstring
        solution.

    Note
    ----------
    The bitstring_solution needs to consist of 0s and 1s only.
    If it is a string, then the characters get converted into integers.
    """
    # Convert binary_solution to correct format e.g. "101" -> (1, 0, 1)
    if isinstance(bitstring_solution, str):
        try:
            bitstring_solution = list(map(int, bitstring_solution))
        except Exception:
            raise ValueError(
                ("Given bitstring_solution cannot be converted to list."
                 "It should have the form e.g. '101' or (1, 0, 1)"))

    spin_solution_cost: float = 0.
    spin_constant: float = 0.
    number_qubits: int = number_cities**2

    # Convert the solution of bits into a solution of spins
    spin_solution: list[int] = values_spins(bitstring_solution)

    # Costs of routes between cities
    for start_city in range(number_cities):
        # Total of weights of all the edges leaving start_city
        city_weights: float = 0.

        # Step to next city
        for next_city in range(number_cities):
            spin_weight = distance_matrix[start_city, next_city]
            city_weights += spin_weight

            # Go through all the time steps
            for time_step in range(number_cities):
                # If the weight is not equal to 0, there exists a route
                if spin_weight != 0:
                    # Define the spins corresponding to the cities and
                    # time steps
                    spin1 = (
                        start_city * number_cities + time_step)
                    spin2 = (
                        next_city * number_cities
                        + ((time_step+1) % number_cities))

                    # Add to cost term
                    spin_solution_cost += (
                        (spin_weight/4.)
                        * spin_solution[spin1]
                        * spin_solution[spin2])

                    spin_constant += (spin_weight/4)

        # Costs of the individual spins
        for spin in range(
                start_city * number_cities,
                start_city * number_cities + number_cities):

            spin_solution_cost += (
                ((-city_weights/2.)
                 - ((number_cities - 2.) * penalty_factor))
                * spin_solution[spin])

    # Spin combinations for visiting a city at multiple time steps
    repetition_spins = list(chain(*(
        combinations(r, 2)
        for r in quantum_circuits.batched(range(number_qubits), number_cities))
    ))

    # Spin combinations for visiting mutiple cities at the same time step
    bilocation_spins = []
    for timestep in range(number_cities):
        # Create list of spins that correspond to timestep
        times = list(
            islice(range(number_qubits), timestep, None, number_cities))
        bilocation_spins += (list(combinations(times, 2)))

    # Add penalty terms for repetitions and bilocations
    for spin_pair in chain(repetition_spins, bilocation_spins):
        penalty_spin1, penalty_spin2 = spin_pair

        spin_solution_cost += (
            (penalty_factor/2.)
            * spin_solution[penalty_spin1]
            * spin_solution[penalty_spin2])

    # Add interaction terms of spins with themselves for both penalty terms
    spin_solution_cost += ((2 * number_qubits) * (penalty_factor/4.))

    # Add constant from penalty terms
    spin_constant += (
        ((penalty_factor * number_cities)
         * ((number_cities**2) - (4 * number_cities) + 4))/2.)

    spin_solution_cost += spin_constant

    return spin_solution_cost


def costs_spins(
        bitstring_solutions: dict[str, int],
        tsp: Tsp,
        penalty_factor: float
        ) -> dict[str, float]:
    """
    Gathers the calculated costs of all the bitstring solutions passed to
     this function and returns the dictionary of the bitstrings with their
     respective costs calculated using spin variables.

    Arguments
    ----------
    bitstring_solutions:
        A dictionary with the binary solutions as strings as the key variables
        and the number of their occurrence as the values.

    tsp:
        The TSP instance given in the format of the TSP class by Qiskit.

    penalty_factor:
        A float which specifies the factor by which the penalty terms of the
        constraints get multiplied.

    Returns
    ----------
    dict[str, float]:
        A dictionary with the binary solutions as strings as the key variables
        and their respective costs as the values.
    """
    number_cities: int = tsp_utils.get_number_cities(tsp)
    distance_matrix: np.ndarray = tsp_utils.get_distance_matrix(tsp)

    # Calculate the cost for each solution
    solutions_spin_cost = [
        cost_spin_solution(
            distance_matrix,
            number_cities,
            penalty_factor,
            bitstring)
        for bitstring in bitstring_solutions.keys()]

    # Collect the binary solutions and their respective costs in a dictionary
    spin_costs: dict[str, float] = dict(
        zip(bitstring_solutions.keys(), solutions_spin_cost, strict=True))

    return spin_costs
