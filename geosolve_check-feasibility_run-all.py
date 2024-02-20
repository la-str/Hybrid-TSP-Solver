"""
This script contains a function that runs through the following steps:
  1. Reads in a file with locations of a large TSP instance
  2. Clusters the large TSP instance into a smaller TSP instance
  3. Runs the QAOA on the smaller TSP instance
  4. Checks for feasibility of the given solution(s)
  5. Expands the feasible solution of the clustered TSP to the full TSP
  6. Runs a classical solver on the full TSP with a random start
  7. Runs a classical solver on the full TSP with the starting seed obtained
      by expanding the solution given by the QAOA for the clustered TSP

The script then calls this function for all files in the latlonlists folder and
 for all cluster sizes in the clusters tuple. So we obtain data logs for all
 the possible combinations of the files and the given cluster sizes.

Note:
----------
- The input file must have 2 columns (comma, separated),
   the 0th column is the latitude and the 1st column is the longitude.
   The first row is a header row and is ignored.

- If none of the solutions given by the QAOA is found to be feasible in step 4,
   the optimized circuit of the QAOA is run again and the feasibility is
   checked again. This is repeated until either a feasible solution is found or
   a timeout of 50 seconds is reached.

- Some of the outputs are logged in a file that is stored in a separate
   directory within the project folder.

- The TSP instance needs to be symmetric as the QAOA implementation is based on
   the assumption that the distance matrix is symmetric.

David Corne, Dan Forbes, and Lara Stroh - 2024
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from time import perf_counter

import numpy as np
from qiskit_optimization.applications import Tsp
from sklearn.cluster import AgglomerativeClustering

from tsp_qaoa.optimise_circuit import optimise_for_tsp
from tsp_qaoa.bitstring_results import results_calculations
from tsp_qaoa.tsp_utils import (
    create_tsp_problem, calc_distance_matrix, create_inital_permutation)
from tsp_qaoa.feasibility_check import find_first_feasible
from solvers import solve_tsp_local_search


###############################################################################
# Configure number of digits in print statements and logs.
###############################################################################

np.set_printoptions(precision=3)


def main(
        llfile: Path | str,
        nclusters: int
        ) -> None:

    # Use the arguments
    sa_time_step1: float = 5.
    sa_time_step2: float = 5.
    quantum_circuit_layers: int = 1
    penalty_factor: float = 300.
    number_shots: int = 1024
    result_number_shots: int = 1024

    ###########################################################################
    # Configure Logging.
    ###########################################################################

    llfile_stem = Path(llfile).stem  # E.g. latlonlists/cn366.csv -> cn366

    # Define log location
    log_dir = (
        Path(__file__).parent  # Project folder
        / "logs"
        / llfile_stem
        / f"{nclusters}_clusters")

    # Create log directory if needed
    log_dir.mkdir(parents=True, exist_ok=True)

    # Define unique log filename
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    log_filename = log_dir / f"{timestamp}_geosolve_{llfile_stem}.txt"

    print(f"Logging to {log_filename}")

    # Get logger for this script
    logger_name = f"{llfile_stem}_{nclusters}"
    log = logging.getLogger(logger_name)
    log.setLevel(logging.INFO)

    # Define logger formatting
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p")

    # Define file log handler
    file_handler = logging.FileHandler(log_filename, "w")
    file_handler.setFormatter(formatter)
    # file_handler.setLevel(logging.INFO)
    log.addHandler(file_handler)

    # Define terminal handler
    terminal_handler = logging.StreamHandler(sys.stdout)  # Log to terminal
    terminal_handler.setFormatter(formatter)
    log.addHandler(terminal_handler)

    ###########################################################################
    # Read in the latitudes and longitudes of the locations, create IDs for
    #  the locations, and get the distance matrix.
    ###########################################################################

    # Read input latlon list
    latlongs = np.loadtxt(llfile, delimiter=",", skiprows=1)  # Skip header row
    log.info(f"Read in {llfile}.")

    # Formal ID for each location
    location_ids = np.arange(latlongs.shape[0], dtype=np.uint64)

    # Calculate a square distance matrix containing the distances between every
    #  city and every other city.
    distmat = calc_distance_matrix(latlongs)

    # Get the number of cities in the full TSP.
    n_cities_large: int = distmat.shape[0]
    log.info(f"The full TSP consists of {n_cities_large} cities.")
    log.info(f"The distance matrix of the full TSP:\n {distmat}")

    ###########################################################################
    # Cluster the larger TSP into a smaller TSP instance.
    ###########################################################################

    # Cluster the latlons into nclusters clusters
    time_before_clustering = perf_counter()
    clustering = AgglomerativeClustering(nclusters).fit(latlongs)
    cluster_labels = clustering.labels_
    log.info(f"The locations are assigned to clusters:\n {cluster_labels}")
    log.info(f"Requested {nclusters}, found {clustering.n_clusters_}.")

    # Overwrite input nclusters with the actual number of clusters found
    nclusters = int(clustering.n_clusters_)

    # Initialize arrays to store the average lat and lon for each cluster
    cluster_centres = np.zeros((nclusters, 2))

    log.info("Calculating cluster centres")
    for c in range(nclusters):
        # Where the cluster label is c
        cluster_latlongs = latlongs[cluster_labels == c]

        # Calculate the mean of lat and lon for the cluster
        cluster_centres[c] = np.mean(cluster_latlongs, axis=0)

    log.info(f"The cluster centres:\n {cluster_centres}")

    log.info("Calculate distance matrix of cluster centres")
    clustered_distance_matrix = calc_distance_matrix(cluster_centres)
    log.info(clustered_distance_matrix)
    time_clustering = perf_counter() - time_before_clustering

    log.info(f"The clustering time: {time_clustering} sec.")

    ###########################################################################
    # Update the penalty factor if necessary.
    ###########################################################################

    # Check if the penalty factor is smaller than the largest weight in the
    #  distance matrix
    largest_weight = float(np.max(clustered_distance_matrix))
    if penalty_factor < largest_weight:
        print(f"Penalty factor of {penalty_factor} too small,", end="")
        penalty_factor = largest_weight * 100
        print(f" using {penalty_factor} instead.")

    ###########################################################################
    # Create the clustered TSP instance.
    ###########################################################################

    # Create TSP problem instance
    time_before_instance_creation = perf_counter()
    tsp_problem: Tsp = create_tsp_problem(
        cluster_centres, clustered_distance_matrix)
    time_instance_creation = perf_counter() - time_before_instance_creation

    log.info(f"The time creating the TSP: {time_instance_creation} sec.")

    ###########################################################################
    # Run the quantum approximate optimization algorithm on the clustered TSP.
    ###########################################################################

    # Generate an optimised quantum circuit for the given Tsp problem
    time_before_quantum_calculation = perf_counter()
    optimised_circuit, optimised_result = optimise_for_tsp(
        tsp_problem,
        quantum_circuit_layers,
        penalty_factor,
        number_shots)

    # Obtain the results from the optimized circuit
    possible_solutions, solutions_costs = results_calculations(
        optimised_circuit,
        tsp_problem,
        penalty_factor,
        result_number_shots)
    time_quantum_calculation = perf_counter() - time_before_quantum_calculation

    log.info(f"The time running the QAOA: {time_quantum_calculation} sec.")

    ###########################################################################
    # Display the results of the optimized circuit.
    ###########################################################################

    # Display best result ordered by occurrence probabilities
    solution_max_prob = list(possible_solutions)[0]
    log.info((
        f'The most often occuring solution is {solution_max_prob}.\n'
        f'It occurs {possible_solutions[solution_max_prob]} time(s) '
        f'and has cost {solutions_costs[solution_max_prob]}.'))

    # Display best result ordered by cost values
    solution_min_cost_str: str = list(solutions_costs)[0]
    log.info((
        f'The solution with the lowest cost is {solution_min_cost_str}.\n'
        f'It occurs {possible_solutions[solution_min_cost_str]} time(s) '
        f'and has cost {solutions_costs[solution_min_cost_str]}.'))

    ###########################################################################
    # Check for feasible solution in the possible solutions given by the QAOA.
    # If no feasible solution found, run the optimised circuit again and check.
    ###########################################################################

    time_before_feasibility_check = perf_counter()
    start_time = perf_counter()
    while True:
        try:
            feasible_cluster_sol = find_first_feasible(
                solutions_costs, nclusters)
            break  # Found a feasible solution

        except ValueError:
            if (perf_counter() - start_time) >= 50:
                log.info("No feasible solution found in 50 sec.")
                raise TimeoutError("No feasible solution found in 50 sec.")
            # Retry if no feasible solution found
            possible_solutions, solutions_costs = results_calculations(
                optimised_circuit,
                tsp_problem,
                penalty_factor,
                result_number_shots)
            continue

    time_feasibility_check = perf_counter() - time_before_feasibility_check

    log.info(f"Time to find a valid solution: {time_feasibility_check} sec.")
    log.info(f"The feasible solution for the clusters: {feasible_cluster_sol}")

    ###########################################################################
    # Expand the solution for the clustered TSP to the full TSP instance to
    #  obtain an initial permutation to use as seed.
    ###########################################################################

    time_before_solution_expansion = perf_counter()
    initial_permutation = create_inital_permutation(
        feasible_cluster_sol,
        nclusters,
        location_ids,
        cluster_labels)
    time_solution_expansion = perf_counter() - time_before_solution_expansion

    log.info(f"Time to expand the solution: {time_solution_expansion} sec.")
    log.info(f"The final seed permutation is\n {initial_permutation.tolist()}")

    ###########################################################################
    # Solve the full TSP from a random start.
    ###########################################################################

    time_before_classical_random = perf_counter()
    permutation2, distance2 = solve_tsp_local_search(
        distmat,
        max_processing_time=sa_time_step1+sa_time_step2,
        logger_name=logger_name)
    time_classical_random = perf_counter() - time_before_classical_random

    log.info((f'The result solution is {permutation2}.\n'
              f'No seed distance: {distance2} km'))

    log.info(f"Time of solver with random start: {time_classical_random} sec.")

    ###########################################################################
    # Solve the full TSP using initial_permutation as the seed.
    ###########################################################################

    time_before_classical_seed = perf_counter()
    permutation3, distance3 = solve_tsp_local_search(
        distmat,
        initial_permutation=initial_permutation.tolist(),
        max_processing_time=sa_time_step2,
        logger_name=logger_name)
    time_classical_seed = perf_counter() - time_before_classical_seed

    log.info((f'The result solution is {permutation3}.\n'
              f'With seed distance: {distance3} km'))

    log.info(f"The solver time using a seed: {time_classical_seed} sec.")

    time_total_seed = (
        time_clustering
        + time_instance_creation
        + time_quantum_calculation
        + time_feasibility_check
        + time_solution_expansion
        + time_classical_seed)

    log.info(f"Total time of the algorithm with seed: {time_total_seed} sec.")


if __name__ == "__main__":
    clusters = (3, 4, 5)
    files = Path(__file__).parent.glob("latlonlists/*.csv")

    for file in files:
        for nclusters in clusters:
            # If there is a TimeoutError for an instance, continue to next one
            try:
                main(file, nclusters)
            except TimeoutError:
                pass
