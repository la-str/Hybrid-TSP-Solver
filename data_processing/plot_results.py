"""
Contains functions to read in the results from a specified Excel file and to
 plot them (cost value VS time) using multiprocessing.

Dan Forbes and Lara Stroh - 2024
"""

import os
from pathlib import Path
from multiprocessing import Pool

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_results(filename_to_plot: str, df: pd.DataFrame, plot_dir: Path):
    """
    Plots the results for a given filename and saves the graphs to a given
     directory.

    Arguments
    ----------
    filename_to_plot:
        A string which specifies the filename to plot.

    df:
        A pandas DataFrame which contains the data to plot.

    plot_dir:
        A pathlib.Path which specifies the directory to save the plots to.
    """
    # Create a sub-DataFrame for the filename to plot
    sub_df = df[df["latlon_filename"] == filename_to_plot]

    # Plot the data
    sns.lmplot(
        data=sub_df,
        x="Time (ms)" if sub_df["Time (ms)"].max() < 5000 else "Time (s)",
        y="cost",
        hue="uses_seed",
        col="n_clusters",
        fit_reg=False,
        legend=True,
        palette="Set1",
        scatter_kws={"s": 10}  # Adjust the marker size here (e.g., s=10)
    )
    plt.suptitle(filename_to_plot)
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig(plot_dir / filename_to_plot, dpi=300)


def main():
    # Avoid bug with high core count CPUs
    NCPU: int | None = None if os.cpu_count() < 62 else 60

    # Specify the file path
    file_path = "results.xlsx"

    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(file_path)

    # Convert time from seconds to milliseconds
    df["Time (ms)"] = df["time"] * 1000

    # Rename the time column
    df["Time (s)"] = df["time"]

    # Create output directory
    plot_dir = Path(__file__).parent.parent / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Get the unique latlon_filenames
    filenames = df["latlon_filename"].unique()

    # Create a pool of workers
    with Pool(NCPU) as pool:
        # Plot the results for each latlon_filename
        pool.starmap(
            plot_results,
            [(filename, df, plot_dir) for filename in filenames])


if __name__ == "__main__":
    main()
