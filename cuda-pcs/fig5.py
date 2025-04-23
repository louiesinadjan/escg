import pandas as pd
import matplotlib.pyplot as plt
import os

# Load your CSV
df = pd.read_csv("pcs.csv")

# Ensure output directory exists
os.makedirs("plots_per_mcs", exist_ok=True)

# Define visual attributes
markers = ['^', 'o', 'o', 's', 's']
colors = ['red', 'green', 'blue', 'grey', 'black']

# Iterate through each unique MCS value
for mcs_val in sorted(df['mcs'].unique()):
    mcs_df = df[df['mcs'] == mcs_val]

    # Group by system size (length) and alpha
    grouped = mcs_df.groupby(['length', 'alpha'])['s5'].mean().reset_index()
    grouped['1-rho_5'] = 1 - grouped['s5']
    system_sizes = sorted(grouped['length'].unique())

    # Plot
    plt.figure(figsize=(10, 6))
    for i, size in enumerate(system_sizes):
        data = grouped[grouped['length'] == size]
        plt.plot(data['alpha'], data['1-rho_5'],
                 label=str(size),
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 linestyle='-')

    # Annotate solution phases
    # plt.text(0.02, 0.99, "all", fontsize=12, ha='left', va='bottom')
    # plt.text(0.2, 1.01, r"$A^7_{\{0,1,2,3,4,6,7\}}$", fontsize=12, ha='center')
    # plt.text(0.385, 1.01, r"$A^4_{\{1,3,5,7\}}$", fontsize=12, ha='right')

    # Labels and formatting
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$1 - \rho_5$")
    plt.legend(title="System size")
    plt.ylim(0.75, 1.01)
    plt.grid(True)
    plt.title(f"Absence of Species 5 vs Alpha at MCS={mcs_val}")
    plt.tight_layout()
    plt.savefig(f"plots_per_mcs/absence_species5_mcs_{mcs_val}.png")
    plt.close()