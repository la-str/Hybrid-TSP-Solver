"""
Contains a function to return the given solutions and costs of the optimised
 quantum circuit.

Lara Stroh - 2023
"""

from operator import itemgetter

import numpy as np
from qiskit_optimization.applications import Tsp
from qiskit import (
    QuantumCircuit,
    Aer, transpile
)

from . import cost
from . import tsp_utils


def results_calculations(
        optimised_circuit: QuantumCircuit,
        tsp: Tsp,
        penalty_factor: float,
        number_shots: int = 1024,
        ) -> tuple[dict[str, int], dict[str, float]]:
    """
    Runs the quantum circuit, gets the costs for the given solutions, and
     then returns the bitstring solutions with the number of their occurrences
     and their respective costs.

    Arguments
    ----------
    optimised_circuit:
        The optimal quantum circuit which describes the QAOA circuit for the
        given TSP.

    number_shots:
        An integer which specifies how often the quantum circuit is run. The
        default value, if none is given, is 1024.

    tsp:
        The TSP instance given in the format of the TSP class by Qiskit.

    penalty_factor:
        A float which specifies the factor by which the penalty terms of the
        constraints get multiplied.

    Returns
    ----------
    dict[str, int]:
        A dictionary with the binary solutions as strings as the key variables
        and the number of their occurrence as the values. The keys are sorted
        by which solution occurs most often.

    dict[str, float]:
        A dictionary with the binary solutions as strings as the key variables
        and their respective costs as the values. The keys are sorted by which
        solution has the lowest cost value.

    Note
    ----------
    The two returns are packed into a tuple.
    """
    simulator_backend = Aer.get_backend('qasm_simulator')
    run_cicruit = simulator_backend.run(
        transpile(optimised_circuit, simulator_backend), shots=number_shots)

    # Order binary solutions by descending number of occurrences
    sorted_counts: dict[str, int] = dict(sorted(
        run_cicruit.result().get_counts().items(),
        key=itemgetter(1), reverse=True))

    # Reverse the order of the elements of the strings in the keys to change
    # Qiskit's ordering of the bits in the solutions given by running the
    # circuit to the order used in the encoding of the solutions
    bitstring_solutions: dict[str, int] = {
        bitstring[::-1]: occurrences
        for bitstring, occurrences in sorted_counts.items()}

    number_cities: int = tsp_utils.get_number_cities(tsp)
    distance_matrix: np.ndarray = tsp_utils.get_distance_matrix(tsp)

    # Calculate the individual costs of the solutions
    solutions_costs = [
        cost.calculation_cost_Hamiltonian(
            distance_matrix,
            number_cities,
            penalty_factor,
            bitstring)
        for bitstring in bitstring_solutions.keys()]

    # Create dictionary of the bitstrings and costs sorted in ascending order
    # of the cost values
    bitstring_costs: dict[str, float] = dict(sorted(
        zip(bitstring_solutions.keys(), solutions_costs, strict=True),
        key=itemgetter(1)))

    return bitstring_solutions, bitstring_costs
