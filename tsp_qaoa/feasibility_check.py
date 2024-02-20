"""
Contains functions for evaluating the feasibility of the possible solutions
 given by the QAOA algorithm.

Lara Stroh and Dan Forbes - 2024
"""

from typing import Mapping

import numpy as np
import numpy.typing as npt


def convert_key(key: str, n_cities: int) -> npt.NDArray[np.bool_]:
    """
    Converts a key from the QAOA algorithm to a numpy array.

    Arguments
    ----------
    key:
        A string which represents the key from the QAOA algorithm.

    n_cities:
        An integer which indicates the number of cities of the TSP.

    Returns
    ----------
    np.ndarray:
        The numpy array.
    """
    return np.array(
        [int(x) for x in key], dtype=np.bool_
        ).reshape(n_cities, n_cities)


def find_first_feasible(
        solutions_dict: Mapping[str, int | float],
        n_cities: int
        ) -> str:
    """
    Finds the first feasible solution in the dictionary of solutions.

    Arguments
    ----------
    solutions_dict (Mapping[str, int | float]):
        Dictionary of solutions.
        The keys are e.g. "01100101..."

     n_cities:
        An integer which indicates the number of cities of the TSP.

    Returns
    ----------
    str:
        The first feasible solution.
    """
    for key in solutions_dict:
        # Validate length of solutions_dict keys with n_cities
        if len(key) != n_cities**2:
            raise ValueError("Invalid key length.")

        # Validate that each city is visited exactly once
        solution_array = convert_key(key, n_cities)
        cols_valid = np.sum(solution_array, axis=0) == 1
        rows_valid = np.sum(solution_array, axis=1) == 1
        if np.all(cols_valid) and np.all(rows_valid):
            return key

    raise ValueError("No feasible solution found.")
