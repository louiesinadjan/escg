import os
import csv
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from os.path import splitext

def plot_densities(csv_file, directory):
    """
    Reads 'densities.csv' (with header: MCS, ROCK, PAPER, SCISSORS, LIZARD, SPOCK)
    and produces a semilog-x plot of each density vs. MCS.
    Saves the figure to 'densities.png'.
    """

    filepath = os.path.join(directory, csv_file)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    steps = []
    rock = []
    paper = []
    scissors = []
    lizard = []
    spock = []

    # Read the CSV
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip the header row, e.g. MCS,ROCK,PAPER,...
        for row in reader:
            if len(row) < 6:
                continue  # skip malformed rows
            steps.append(float(row[0]))
            rock.append(float(row[1]))
            paper.append(float(row[2]))
            scissors.append(float(row[3]))
            lizard.append(float(row[4]))
            spock.append(float(row[5]))

    # Create the figure
    plt.figure(figsize=(10, 5))
    plt.tight_layout()

    # Plot each density on a semilog x-axis
    plt.semilogx(steps, rock, label="Rock", color='b')
    plt.semilogx(steps, paper, label="Paper", color='cyan')
    plt.semilogx(steps, scissors, label="Scissors", color='g')
    plt.semilogx(steps, lizard, label="Lizard", color='y')
    plt.semilogx(steps, spock, label="Spock", color='r')

    # Add legend, labels, and title
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel(r"$\rho_i$")  # LaTeX-style label for densities
    plt.title("Density Evolution Over Time")

    # Save the figure
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

    matrix = np.array(data_list, dtype=int)

    # Prepare a discrete colormap for values 0..5
    colours = ["white", "blue", "cyan", "green", "yellow", "red"]
    cmap = ListedColormap(colours)

    fig, ax = plt.subplots(figsize=(6, 4))

    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=5)
    ax.set_title(csv_file, fontsize=12)
    ax.set_xlabel("Column Index")
    ax.set_ylabel("Row Index")

    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.05)
    cbar.set_ticks(range(6))  # 0 through 5
    cbar.set_label("Matrix Values")

    plt.tight_layout()

    base, _ = splitext(csv_file)
    output_png = base + ".png"
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
        if csv_file.lower() == "densities.csv":
            # Plot densities
            plot_densities(csv_file, directory)
        else:
            # Plot as matrix snapshot
            plot_matrix_snapshot(csv_file, directory)


if __name__ == "__main__":
    # By default, use the directory where the script is located
    visualise()