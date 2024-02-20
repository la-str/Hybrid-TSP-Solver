"""
Contains the functions which create the quantum circuit for the application of
 the quantum approximate optimisation algorithm (QAOA) to the travelling
 salesman problem (TSP).

Lara Stroh - 2023
"""

from itertools import combinations, chain, islice

import numpy as np
import numpy.typing as npt
from qiskit import (
    QuantumCircuit,
    QuantumRegister, ClassicalRegister
)


def batched(iterable, n):
    """
    Import from itertools in Python 3.12.
    Used as Qiskit is not yet compatible with Python 3.12.
    """
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def cost_layer_interaction_terms(
        qubit_1: int,
        qubit_2: int,
        gamma: float,
        weight: float,
        quantum_register: QuantumRegister,
        classical_register: ClassicalRegister
        ) -> QuantumCircuit:
    """
    Applies the RZZ gate (as two CNOT gates and one RZ gate) to the qubit pair.

    Arguments
    ----------
    qubit_1:
        An integer which specifies which qubit is assigned the role of the
        first qubit for the application of the gates.

    qubit_2:
        An integer which specifies which qubit is assigned the role of the
        second qubit for the application of the gates.

    gamma:
        A float specifying the value of the anlge gamma used in the RZ gate.

    weight:
        A float specifying the value of the distance between the nodes/cities
        associated to the first qubit and second qubit.

    quantum_register:
        A QuantumRegister which consists of a number of qubits and is used as
        part of a quantum circuit.

    classical_register:
        A ClassicalRegister which consists of a number of bits and is used as
        part of a quantum circuit.

    Returns
    ----------
    QuantumCircuit:
        The returned quantum circuit consists of the two CNOT gates and the RZ
        gate applied to the specified two qubits.
    """
    cost_circuit: QuantumCircuit = QuantumCircuit(
        quantum_register, classical_register)

    # Apply the gates
    cost_circuit.cx(qubit_1, qubit_2)
    cost_circuit.rz(gamma * weight, qubit_2)
    cost_circuit.cx(qubit_1, qubit_2)

    return cost_circuit


def circuit_layer(
        gamma: float,
        beta: float,
        penalty: float,
        quantum_register: QuantumRegister,
        classical_register: ClassicalRegister,
        distance_matrix: npt.NDArray[np.float64]
        ) -> QuantumCircuit:
    """
    Creates one layer of the cost and mixer sub-circuitx using some given
     angles beta and gamma.
    The circuit creation works for a symmetric combinatorial problem with a
     symmetric distance_matrix only.

    Arguments
    ----------
    gamma:
        A float specifying the value of the anlge gamma used in the cost
        sub-circuit.

    beta:
        A float specifying the value of the anlge beta used in the mixer
        sub-circuit.

    penalty:
        A float which specifies the factor by which the penalty terms of the
        constraints get multiplied.

    quantum_register:
        A QuantumRegister which consists of a number of qubits and is used as
        part of a quantum circuit.

    classical_register:
        A ClassicalRegister which consists of a number of bits and is used as
        part of a quantum circuit.

    distance_matrix:
        A np.ndarray containing np.float64 numbers and which specifies the
        weights of the edges between the nodes (cities).

    Returns
    ----------
    QuantumCircuit:
        The quantum circuit consists of the cost and mixer sub-circuits.

    Steps
    ----------
    Create the cost sub-circuit:
    1. Add the qubit interaction terms for valid routes
    2. Add the individual qubit terms
    3. Add the qubit interaction terms resulting from penalising the
    repetitions of cities in multiple time steps
    4. Add the qubit interaction terms resulting from penalising the
    bilocation in two cities in one time step

    Create the mixer sub-circuit:
    1. Add RX gates to each qubit

    Note
    ----------
    The circuit creation implemented here is for symmetric combinatorial
     problems. Thus, distance_matrix must be symmetric.
    """
    cost_mixer_circuit = QuantumCircuit(quantum_register, classical_register)

    # Validate the number of qubits
    number_qubits: int = quantum_register.size
    number_cities = np.sqrt(number_qubits)
    if number_cities % 1 != 0:
        raise ValueError(
            ("The sqrt of the number of qubits in the quantum register must "
             "yield a whole number."))
    number_cities = int(number_cities)

    # Cost sub-circuit
    # Routes between cities
    for start_city in range(number_cities):
        # Total of weights of all the edges leaving start_city
        city_weights: float = 0.

        # Step to next city
        for next_city in range(number_cities):
            weight = distance_matrix[start_city, next_city]
            city_weights += weight

            # Go through all the time steps
            for time_step in range(number_cities):
                # If the weight is not equal to 0, there exists a route
                if weight != 0:
                    # Define the qubits corresponding to the cities and
                    # time steps
                    qubit1 = (
                        start_city * number_cities + time_step)
                    qubit2 = (
                        next_city * number_cities
                        + ((time_step+1) % number_cities))

                    # Append QAOA circuit with the qubits interaction terms
                    cost_mixer_circuit.compose(cost_layer_interaction_terms(
                        qubit1, qubit2, gamma, weight/4.,
                        quantum_register, classical_register
                        ), inplace=True)

        # Add the individual terms to the cost layer by adding rotational
        # Z gates to qubits
        for qubit in range(start_city * number_cities,
                           start_city * number_cities + number_cities):
            cost_mixer_circuit.rz(
                ((-gamma)
                 * ((city_weights/2.) + ((number_cities - 2.) * penalty))),
                qubit)

    # Qubit combinations for visiting a city at multiple time steps
    repetition_qubits = list(chain(*(
        combinations(row, 2)
        for row in batched(range(number_qubits), number_cities))
    ))

    # Qubit combinations for visiting mutiple cities at the same time step
    bilocation_qubits = []
    for timestep in range(number_cities):
        # Create list of qubits that correspond to timestep
        times = list(
            islice(range(number_qubits), timestep, None, number_cities))
        bilocation_qubits += (list(combinations(times, 2)))

    # Add penalty qubit interaction terms for repetitions and bilocations
    for qubit_pair in chain(repetition_qubits, bilocation_qubits):
        cost_mixer_circuit.compose(cost_layer_interaction_terms(
            *qubit_pair, gamma, penalty/2.,
            quantum_register, classical_register
            ), inplace=True)

    # Add barrier to divide between the cost and mixer layers
    cost_mixer_circuit.barrier()

    # Mixer sub-circuit
    # Implement the mixer layer by adding an RX gate to each qubit
    for qubit in range(number_qubits):
        cost_mixer_circuit.rx(beta, qubit)

    # Add barrier to end this layer
    cost_mixer_circuit.barrier()

    return cost_mixer_circuit


def prepare_circuit(
        number_cities: int,
        number_layers: int,
        gammas: np.ndarray,
        betas: np.ndarray,
        penalty_factor: float,
        distance_matrix: npt.NDArray[np.float64]
        ) -> QuantumCircuit:
    """
    Returns a qiskit.QuantumCircuit for a travelling salesman problem (TSP)
     with the given number of cities and distance matrix.

    Arguments
    ----------
    number_cities:
        An integer which specifies the number of cities.

    number_layers:
        An integer which specifies of how many cost-mixer layers the quantum
        circuit should consist.

    gammas:
        A np.ndarray which contains the values of the gamma angles for the
        cost sub-circuits of all of the layers.

    betas:
        A np.ndarray which contains the values of the beta angles for the
        mixer sub-circuits of all of the layers.

    penalty_factor:
        A float which specifies the factor by which the penalty terms of the
        constraints get multiplied.

    distance_matrix:
        A np.NDArray containing np.float64 numbers which describe the weights
        of the edges, that is the distances between the cities.

    Returns
    ----------
    QuantumCircuit:
        The quantum circuit which describes the QAOA circuit for the given TSP.

    Steps
    ----------
    1. Creating intial state by applying Hadamard gate to each qubit.
    2. Creating the cost and mixer sub-circuits for each layer
    3. Adding a state measurement to each qubit
    """
    if number_cities < 3:
        raise ValueError(
            f"Number cities must be at least 3, not {number_cities}.",
            "A Hamiltonian cylce needs at least three vertices.")

    number_qubits: int = number_cities**2

    # Create the registers and circuit
    quantum_register = QuantumRegister(number_qubits, name='q')
    classical_register = ClassicalRegister(number_qubits, name='classical')
    quantum_circuit = QuantumCircuit(quantum_register, classical_register)

    # Apply one Hadamard gate for each qubit in quantum_register to create the
    # initial state, then partition the circuit with a barrier
    quantum_circuit.h(range(quantum_register.size))
    quantum_circuit.barrier()

    # Create the cost and mixer sub-circuits for each layer
    for layer in range(number_layers):
        quantum_circuit.compose(
            circuit_layer(
                gammas[layer],
                betas[layer],
                penalty_factor,
                quantum_register,
                classical_register,
                distance_matrix),
            inplace=True)

    # Add qubit state measurements
    quantum_circuit.measure(range(number_qubits), classical_register)

    return quantum_circuit
