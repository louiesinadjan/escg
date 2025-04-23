import os
import csv
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from os.path import splitext
import argparse

def plot_densities(csv_file, directory):
    """
    Reads 'densities.csv' (with header: MCS,s0,s1,s2,...) and produces a semilog-x plot of each density vs. MCS.
    Saves the figure to 'densities.png'.
    """

    filepath = os.path.join(directory, csv_file)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or len(header) < 2:
            print("Invalid densities.csv header.")
            return

        rows = list(reader)
        rows = rows[:-1]  # Skip the last line which is just the MCS number
        species_names = header[1:]
        steps = []
        densities = [[] for _ in species_names]

        for row in rows:
            row = [val for val in row if val.strip() != '']  # Strip empty trailing entries
            if len(row) < len(species_names) + 1:
                continue
            steps.append(float(row[0]))
            for i, density in enumerate(row[1:]):
                densities[i].append(float(density))

    plt.figure(figsize=(10, 5))
    plt.tight_layout()

    for i, species_density in enumerate(densities):
        plt.semilogx(steps, species_density, label=species_names[i])

    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel(r"$\rho_i$")
    plt.title("Density Evolution Over Time")

    output_path = os.path.join(directory, "densities.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved densities plot to {output_path}")


def plot_matrix_snapshot(csv_file, directory):
    """
    Reads a CSV as a 2D array of integers (0..5) and saves a PNG snapshot.
    """
    filepath = os.path.join(directory, csv_file)
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        data_list = list(reader)
        data_list = data_list[:-1]  # Skip the last line containing the MCS number
        matrix = np.array(data_list, dtype=int)

    fig, ax = plt.subplots(figsize=(6, 4))

    im = ax.imshow(matrix, cmap="tab20b", vmin=matrix.min(), vmax=matrix.max())
    ax.set_xlabel("Column Index")
    ax.set_ylabel("Row Index")
    ax.vmin = 0
    ax.vmax = 10
    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.05)
    # cbar.set_ticks(range(6))  # 0 through 5
    cbar.set_label("Matrix Values")

    plt.tight_layout()

    base, _ = splitext(csv_file)
    mcs_number = base.split('_')[-1]
    output_png = f"ss_{mcs_number}.png"
    ax.set_title(mcs_number, fontsize=12)
    output_path = os.path.join(directory, output_png)
    plt.savefig(output_path)
    plt.close(fig)

    print(f"Saved matrix plot to {output_path}")


def visualise(directory=None):
    """
    For each CSV file in 'directory':
      - If it's named "densities.csv", plot semilog-x density lines.
      - Otherwise, treat it as a matrix snapshot of values 0..5.
    """

    if directory is None:
        directory = os.path.dirname(os.path.abspath(__file__))

    # Gather all CSV files in the directory
    csv_files = sorted(
        [f for f in os.listdir(directory) if f.lower().endswith(".csv")]
    )

    if not csv_files:
        print("No CSV files found in directory: {}".format(directory))
        return

    for csv_file in csv_files:
        if csv_file.lower() == "params.csv" or csv_file.lower() == "dominance.csv":
            continue
        elif csv_file.lower() == "densities.csv":
            plot_densities(csv_file, directory)
        else:
            plot_matrix_snapshot(csv_file, directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise CSV matrix snapshots in a directory")
    parser.add_argument(
        "directory",
        nargs="?",
        default=None,
        help="Directory containing CSV files (default: current script directory)"
    )
    args = parser.parse_args()

    visualise(args.directory)