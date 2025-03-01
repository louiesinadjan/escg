import os
import csv
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from os.path import splitext

def visualise(directory=None):

    """
    For each CSV file in 'directory' (default: current script's folder):
      - Reads as a 2D matrix of integers (0 to 5)
      - Creates a separate figure
      - Saves as a PNG named after the CSV (e.g., 'matrix1.csv' -> 'matrix1.png')
    """

    # If directory is not specified, default to this script's folder
    if directory is None:
        directory = os.path.dirname(os.path.abspath(__file__))

    # Gather all CSV files in the directory
    csv_files = sorted(
        [f for f in os.listdir(directory) if f.lower().endswith(".csv")]
    )

    if not csv_files:
        print("No CSV files found in directory: {}".format(directory))
        return

    # Prepare a discrete colormap for values 0..5
    colours = ["black", "yellow", "orange", "red", "green", "blue"]
    cmap = ListedColormap(colours)

    for csv_file in csv_files:
        # Build full path to the CSV
        filepath = os.path.join(directory, csv_file)
        
        # Read this CSV into a 2D NumPy array of integers
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            data_list = list(reader)
        matrix = np.array(data_list, dtype=int)

        # Create a new figure for this matrix
        fig, ax = plt.subplots(figsize=(6, 4))

        # Display the matrix with imshow
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=5)
        ax.set_title("Matrix from {}".format(csv_file), fontsize=12)
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Row Index")

        # Add a colorbar for this figure
        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.05)
        cbar.set_ticks(range(6))  # 0 through 5
        cbar.set_label("Matrix Values")

        plt.tight_layout()

        # Derive an output filename based on CSV name, replacing .csv with .png
        base, _ = splitext(csv_file)
        output_png = base + ".png"
        output_path = os.path.join(directory, output_png)

        # Save the figure as a PNG
        plt.savefig(output_path)
        print("Saved matrix plot to {}".format(output_path))

        # Close the figure to free memory before loading the next CSV
        plt.close(fig)

if __name__ == "__main__":
    # By default, use the directory where the script is located
    visualise()