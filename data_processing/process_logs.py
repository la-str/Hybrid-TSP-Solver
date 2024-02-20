"""
Converts the log files from the optimization runs into a single Excel file.

Dan Forbes and Lara Stroh - 2024
"""

import os
import re

import pandas as pd

# Initialize an empty list to store the data
data = []

# Walk through the logs directory
for root, dirs, files in os.walk("logs"):
    for file in files:
        if file.endswith(".txt"):
            # Construct the full file path
            file_path = os.path.join(root, file)

            # Extract latlon_filename and n_clusters from the file path
            latlon_filename = re.search(r'logs/([^/]*)/', file_path).group(1)
            n_clusters = int(re.search(r'(\d+)_clusters', file_path).group(1))

            print(f"Processing {file_path}...")
            print(f"latlon_filename: {latlon_filename}")
            print(f"n_clusters: {n_clusters}")

            # Open the log file and read its lines
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Initialize uses_seed as False for the first optimization run
            uses_seed = False

            # Initialize variables for cost and time
            cost = None
            time = None

            # Iterate over the lines of the log file
            for i, line in enumerate(lines):

                # If the line conatins 'The full TSP consists of',
                #  extract n_cities
                if 'The full TSP consists of' in line:
                    n_cities = int(re.search(r'of (\d+)', line).group(1))
                    continue

                # If the line contains 'Best permutation:',
                #  switch uses_seed to True
                if 'Best permutation:' in line:
                    uses_seed = True
                    continue

                # If the line contains 'Initial cost value:',
                #  extract the cost from the next line
                if "Initial cost value:" in line:
                    cost = float(re.search(
                        r'Initial cost value: (\d+\.\d+)', line).group(1))
                    time = 0.

                # If the line contains 'Found improved permutation:',
                #  extract the cost and time from the next lines
                if 'Found improved permutation:' in line:
                    cost = float(re.search(
                        r'solution cost: (\d+\.\d+)', lines[i+2]).group(1))
                    time = float(re.search(
                        r'time: (\d+\.\d+)', lines[i+3]).group(1))

                # If both cost and time have been found,
                #  append the data to the list and reset cost and time
                if cost is not None and time is not None:
                    data.append([latlon_filename, n_cities, n_clusters,
                                uses_seed, time, cost])
                    cost = None
                    time = None

# Create a DataFrame from the data
df = pd.DataFrame(
    data,
    columns=["latlon_filename", "n_cities",
             "n_clusters", "uses_seed", "time", "cost"])

# Write the DataFrame to an Excel file
df.to_excel("results.xlsx", index=False)
