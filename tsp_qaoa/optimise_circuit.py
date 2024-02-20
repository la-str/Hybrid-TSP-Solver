"""
Contains a function for the optimisation of the quantum circuit.

Lara Stroh - 2023
"""

from functools import partial

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize, OptimizeResult
from qiskit_optimization.applications import Tsp
from qiskit import (
    QuantumCircuit,
    Aer, transpile
)

from . import cost
from . import tsp_utils
from .quantum_circuits import prepare_circuit


def _validate_optimise_for_tsp_inputs(
        number_layers: int,
        penalty_factor: float,
        largest_weight: float,
        number_shots: int,
        initial_gammas: np.ndarray | None,
        initial_betas: np.ndarray | None
        ) -> None:
    """
    Performs a series of checks on the values that are/can be specified by
     the user.

    Arguments
    -----------
    number_layers:
        An integer which specifies of how many cost-mixer layers the quantum
        circuit should consist.

    penalty_factor:
        A float which specifies the factor by which the penalty terms of the
        constraints get multiplied.

    largest_weight:
        A float which gives the value of the largest weight/distance between
        nodes/cities in the given TSP.

    number_shots:
        An integer which specifies how often the quantum circuit is run for
        certain values for the betas and gammas.

    initial_gammas:
        An optional argument specifying the initial values for the gamma
        angles in a np.NDArray containing np.float64 numbers describing the
        angles' values.

    initial_betas:
        An optional argument specifying the initial values for the beta
        angles in a np.NDArray containing np.float64 numbers describing the
        angles' values.
    """
    if number_layers < 1:
        raise ValueError(
            f"The number of layers must be >= 1, not {number_layers}.")

    if penalty_factor < largest_weight:
        raise ValueError(
            ("The penalty factor must be greater than the largest edge weight."
             f" But {penalty_factor} < {largest_weight}."))

    if number_shots < 1:
        raise ValueError(
            f"The number of shots must be >= 1, not {number_shots}.")

    if (initial_gammas is not None) and (len(initial_gammas) != number_layers):
        raise ValueError(
                ("initial_gammas must have the same length as number_layers"
                 f"{len(initial_gammas)} != number_cities"))

    if (initial_betas is not None) and (len(initial_betas) != number_layers):
        raise ValueError(
                ("initial_betas must have the same length as number_layers"
                 f"{len(initial_betas)} != number_cities"))


def optimise_for_tsp(
        tsp: Tsp,
        number_layers: int,
        penalty_factor: float,
        number_shots: int = 1024,
        initial_gammas: npt.NDArray[np.float64] | None = None,
        initial_betas: npt.NDArray[np.float64] | None = None
        ) -> tuple[QuantumCircuit, OptimizeResult]:
    """
    Returns the optimised quantum circuit for a given Tsp instance together
     with the optimised results which include the values of the betas and
     gammas used in that optimised circuit.

    Arguments
    ----------
    tsp:
        The TSP instance given in the format of the TSP class by Qiskit.

    number_layers:
        An integer which specifies of how many cost-mixer layers the quantum
        circuit should consist.

    penalty_factor:
        A float which specifies the factor by which the penalty terms of the
        constraints get multiplied.

    number_shots:
        An integer which specifies how often the quantum circuit is run for
        certain values for the betas and gammas. The default value, if none is
        given, is 1024.

    initial_gammas:
        An optional argument specifying the initial values for the gamma
        angles in a np.NDArray containing np.float64 numbers describing the
        angles' values. If none are given, then random values for the gamma
        angles are created in this function here.

    initial_betas:
        An optional argument specifying the initial values for the beta
        angles in a np.NDArray containing np.float64 numbers describing the
        angles' values. If none are given, then random values for the beta
        angles are created in this function here.

    Returns
    -----------
    QuantumCircuit:
        The optimal quantum circuit which describes the QAOA circuit for the
        given TSP.

    OptimizeResult:
        A class containing the optimization results returned by the
        scipy.optimize minimize module.

    Note
    ----------
    The optimised beta and gamma results can be gotten using:
    >>> circuit, result = optimise_for_tsp(...)
    >>> betas = result.betas
    >>> gammas = result.gammas

    For more information for the optimisation result see:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    """
    number_cities: int = tsp_utils.get_number_cities(tsp)
    distance_matrix: np.ndarray = tsp_utils.get_distance_matrix(tsp)

    largest_weight: float = np.max(distance_matrix)

    # Raise ValueError on invalid inputs
    _validate_optimise_for_tsp_inputs(
        number_layers,
        penalty_factor,
        largest_weight,
        number_shots,
        initial_gammas,
        initial_betas
    )

    # Create random intial_gammas if not given
    if initial_gammas is None:
        initial_gammas = np.random.uniform(0., np.pi, size=number_layers)

    # Create random intial_betas if not given
    if initial_betas is None:
        initial_betas = np.random.uniform(0., np.pi, size=number_layers)

    # Prefill arguments for circuit creation
    prepare_this_circuit = partial(
        prepare_circuit,
        number_cities=number_cities,
        number_layers=number_layers,
        penalty_factor=penalty_factor,
        distance_matrix=distance_matrix
    )

    # Combine into 1D-array to be compatible with minimize
    initial_values = np.concatenate((initial_betas, initial_gammas))

    # Ojective to be optimised, returns the average cost value
    def objective(
            x: np.ndarray,
            dist_matrix: np.ndarray,
            n_cities: int,
            penalty: float,
            n_shots: int
            ) -> float:
        betas, gammas = np.split(x, 2)

        circuit: QuantumCircuit = prepare_this_circuit(
            betas=betas,
            gammas=gammas)

        # Execute the circuit on a simulator
        simulator_backend = Aer.get_backend('qasm_simulator')
        job = simulator_backend.run(
            transpile(circuit, simulator_backend), shots=n_shots)

        # Results of the execution with solutions in binary encoding
        counts: dict[tuple[int, ...], int] = cost.counts_to_binary_result(
            job.result().get_counts())

        # Combine the number of occurrences of the returned distinct solutions
        occurrences = np.array(list(counts.values()))

        # Combine the costs of the returned distinct solutions
        costs = np.array([cost.calculation_cost_Hamiltonian(
            dist_matrix,
            n_cities,
            penalty,
            solution_id)
            for solution_id in counts.keys()])

        # Multiply the cost of each solution and the number of its occurrences
        costs *= occurrences

        return float(np.average(costs))

    # Optimize the angles' values by minimizing the objective
    result: OptimizeResult = minimize(
        objective,
        x0=initial_values,
        args=(
            distance_matrix,
            number_cities,
            penalty_factor,
            number_shots
        ),
        method='COBYLA'
    )

    if not result.success:
        raise OptimizationFailed(result)

    # Getting the optimised values for the betas and gammas
    optimised_betas, optimised_gammas = np.split(result.x, 2)

    print(f"""The optimised angles are:
    Betas: {optimised_betas}
    Gammas: {optimised_gammas}""")

    # Assign the optimised values to result variables for direct calls
    result.betas = optimised_betas
    result.gammas = optimised_gammas

    # Create the optimized circuit
    optimised_circuit: QuantumCircuit = prepare_this_circuit(
        betas=optimised_betas,
        gammas=optimised_gammas)

    return optimised_circuit, result


class OptimizationFailed(Exception):
    def __init__(self, result: OptimizeResult):
        self.message = result.message
        self.result = result
        super().__init__(self.message)
