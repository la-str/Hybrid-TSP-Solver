"""
Modifed from the python-tsp package.
https://github.com/fillipe-gsm/python-tsp

Contains a class and functions to solve and log the process of solving a TSP:
- Class setting up a timer and returning the elapsed time
- Function which returns a solution (permutation and cost value) to a TSP,
   logging information/steps of the solution finding process
- Function calculating the cost value to a given permutation

Dan Forbes and Lara Stroh - 2024
"""

import random
import logging
from pathlib import Path
from time import perf_counter
from datetime import datetime

import numpy as np
from python_tsp.heuristics.perturbation_schemes import neighborhood_gen


class TimeSince:
    """
    Returns time since this object was created in seconds
     when the instance is called.

    Example
    -------
    >>> timer = TimeSince()  # Start timer
    >>> time_elapsed: float = timer()
    """
    def __init__(self):
        self.start_time = perf_counter()

    def __call__(self):
        return perf_counter() - self.start_time


DEFAULT_TIMEOUT_S: float = 50


def solve_tsp_local_search(
        distance_matrix: np.ndarray,
        initial_permutation: list[int] | None = None,
        perturbation_scheme: str = "two_opt",
        max_processing_time: float | None = None,
        logger_name: str | None = None,
        make_logger: bool = False,
        log_dir: Path | None = None
        ) -> tuple[list[int], float]:
    """
    Returns a solution to the TSP (permutation and cost value).
    The function logs the process, that is, it prints new improved solutions
    every time they change and also the time at which these improvements occur.

    Arguments
    ----------
    distance_matrix:
        A numpy array containing the distances between the cities.

    inital_permutation:
        An optional argument which is a list containing integers from 0 to
        n - 1, which represent the cities, given in any order. This represents
        a possible solution to the TSP, which is used as start to the solution
        finding process from.
        If None is provided, a random starting permutation will be generated.

    perturbation_scheme:
        Determines method which is used to generate the new solutions.
        The default method is set to "two_opt".

    max_processing_time:
        An optional float which can be used to set a maximum time (in seconds)
        for the algorithm to run before it stops its solution finding process
        and outputs the best solution found so far. If None is given, the
        algortihm continues until it finds a local minimum.

    logger_name:
        An optional str which gives the name of the logger where the
        information should be stored in.

    make_logger:
        A boolean which indicates whether a logger should be created or not.

    log_dir:
        An optional Path which gives the location of the directory where the
        log file should be stored in.

    Returns
    ----------
    list[int]:
        A list of integers which gives the best permutation found at the point
        when the algorithm stops. That is, the permutation that gives the
        lowest cost value found.

    float:
        The float outputs the cost value of the best permutation found by the
        end of the algorithm.
    """
    if not make_logger:
        if logger_name is None:
            logger_name = __name__
            logger = logging.getLogger(logger_name)
        else:
            logger = logging.getLogger(logger_name+".local_search")
    elif make_logger and (log_dir is not None):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        # Define logger formatting
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p")

        # Define log location
        timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        log_filename = log_dir / f"{timestamp}_geosolve_{logger_name}.txt"

        # Define file log handler
        file_handler = logging.FileHandler(log_filename, "w")
        file_handler.setFormatter(formatter)
        # file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    else:
        raise ValueError("If make_logger is True, log_dir must be provided.")

    # Getting the number of cities in the TSP to be solved
    n_cities_large: int = distance_matrix.shape[0]
    logger.info(f"The TSP to be solved consists of {n_cities_large} cities.")

    # Check if initial_permutation is provided, if not generate random one
    if initial_permutation is None:
        n_cities: int = distance_matrix.shape[0]  # number of nodes
        initial_permutation: list[int] = [
            0,  # ensure 0 is the first node
            *random.sample(range(1, n_cities), n_cities - 1)
        ]
        logger.info("Running with Seed randomly generated.")
    else:
        logger.info("Running with Seed provided.")

    logger.info(f"Initial permutation: {initial_permutation}")

    best_cost = calc_cost_value(distance_matrix, initial_permutation)

    # Get a generator for the neighborhood
    neighborhood_generator = neighborhood_gen[perturbation_scheme]

    if max_processing_time is None:
        max_processing_time = DEFAULT_TIMEOUT_S

    logger.info(f"Initial cost value: {best_cost}")
    timer = TimeSince()

    # Set-up local search variables
    best_permutation = initial_permutation
    stop_early = False
    improvement = True

    logger.info("Beginning local search")
    while improvement and (not stop_early):
        improvement = False
        for next_permutation in neighborhood_generator(best_permutation):
            if timer() > max_processing_time:
                logger.warning("Stopping as time limit has been reached")
                stop_early = True
                break

            current_cost = calc_cost_value(distance_matrix, next_permutation)

            if current_cost < best_cost:
                logger.info(
                    (f"Found improved permutation:\n  {next_permutation}\n"
                     f"  solution cost: {current_cost}\n"
                     f"  time: {timer()}"))
                improvement = True
                best_permutation, best_cost = next_permutation, current_cost
                break  # Early stop due to first improvement local search

    logger.info(f"Finished local search after {timer()} seconds")
    logger.info(f"Best permutation cost: {best_cost}")
    logger.info(f"Best permutation: {best_permutation}")

    return best_permutation, best_cost


def calc_cost_value(
        distance_matrix: np.ndarray,
        permutation: list[int]
        ) -> float:
    """
    Calculates the total cost value (route distance) of a given permutation
     which represents a solution to the TSP.

    Arguments
    ----------
    distance_matrix:
        A numpy array containing the distances between the cities.

    permutation:
        A list containing integers from 0 to n - 1, which represent the cities,
        and they are in any order, giving a possible solution to the TSP.

    Returns
    ----------
    float:
        The total cost (distance) of the possible solution path described in
        the given permutation for the provided distance_matrix.
    """
    return distance_matrix[permutation, np.roll(permutation, -1)].sum()
