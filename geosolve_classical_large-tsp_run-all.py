"""
This script contains a function that runs through the following steps:
  1. Reads in a file with locations of a large TSP instance
  2. Solves the TSP instance using a classical solver (a heuristic local search
      algorithm). It repeats this step for a given number of n_iterations

The script then calls this function and runs it for each of the files without
 as well as with initial seed permutations which are extracted from the files.

Multi-processing is used to run the function for each file in parallel.

Note:
----------
- The input file must have 2 columns (comma, separated),
   the 0th column is the latitude and the 1st column is the longitude.
   The first row is a header row and is ignored.

- Some of the outputs are logged in a file that is stored in a separate
   directory within the project folder.

David Corne, Dan Forbes, and Lara Stroh - 2024
"""

import multiprocessing as mp
import os
import pickle
import sys
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from solvers import solve_tsp_local_search
from tsp_qaoa.tsp_utils import calc_distance_matrix


###############################################################################
# Configure number of digits in print statements and logs.
###############################################################################

np.set_printoptions(precision=3)


def main(
        llfile: Path | str,
        nclusters: int,
        seed: list[int] | None = None,
        n_iterations: int = 100,
        max_processing_time: float = 10.,
        pool: Any | None = None
        ) -> None:

    ###########################################################################
    # Configure Logging.
    ###########################################################################

    llfile_stem = Path(llfile).stem  # E.g. latlonlists/cn366.csv -> cn366

    # Define log location
    log_dir = (
        Path(__file__).parent  # Project folder
        / "logs_large_run"
        / llfile_stem
        / f"{nclusters}_clusters")

    # Create log directory if needed
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get logger for this script
    logger_name = f"{llfile_stem}_{nclusters}"
    log = logging.getLogger(logger_name)
    log.setLevel(logging.INFO)

    # Define logger formatting
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p")

    # Define terminal handler
    terminal_handler = logging.StreamHandler(sys.stdout)  # Log to terminal
    terminal_handler.setFormatter(formatter)
    log.addHandler(terminal_handler)

    ###########################################################################
    # Read in the latitudes and longitudes of the locations and get the
    #  distance matrix.
    ###########################################################################

    # Read input latlon list
    latlongs = np.loadtxt(llfile, delimiter=",", skiprows=1)  # Skip header row
    log.info(f"Read in {llfile}.")

    # Calculate a square distance matrix containing the distances between
    #  every city and every other city.
    distmat = calc_distance_matrix(latlongs)

    # Get the number of cities in the full TSP.
    n_cities_large: int = distmat.shape[0]
    log.info(f"The full TSP consists of {n_cities_large} cities.")
    log.info(f"The distance matrix of the full TSP:\n {distmat}")

    ###########################################################################
    # Solve the full TSP.
    ###########################################################################

    time_before = perf_counter()

    jobs = (
        (distmat,
         seed,
         "two_opt",
         max_processing_time,
         f"{logger_name}_{i}",
         True,
         log_dir)
        for i in range(n_iterations)
    )

    if pool is not None:
        pool.starmap(solve_tsp_local_search, jobs)
    else:
        for job in jobs:
            solve_tsp_local_search(**job)

    time_after = perf_counter() - time_before

    log.info(f"The solver time using a seed: {time_after} seconds.")


if __name__ == "__main__":
    clusters = (3, 4, 5)
    files = Path(__file__).parent.glob("latlonlists/*.csv")

    # Avoid bug with high core count CPUs
    n_cpu: int | None = None if os.cpu_count() < 62 else 60

    # Load the dictionary of permutations
    with open("permutations.pickle", "rb") as file:
        permutations = pickle.load(file)

    with mp.Pool(n_cpu) as pool:
        for file in files:

            # Run once without a seed
            main(
                llfile=file,
                nclusters=0,  # Flag to indicate completely random start
                pool=pool)

            # Run once for each cluster
            for _nclusters in clusters:
                # If there is no seed permutation for this instance, skip it
                if permutations[(file.stem, _nclusters)] is None:
                    break

                main(
                    llfile=file,
                    nclusters=_nclusters,
                    seed=permutations[(file.stem, _nclusters)],
                    pool=pool)
