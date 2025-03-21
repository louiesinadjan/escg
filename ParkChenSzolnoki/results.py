import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import results.csv
results = pd.read_csv('results3.csv')

# Clean column names (remove leading/trailing spaces)
results.columns = results.columns.str.strip()

# Group by alpha, beta, and mcs, computing survival probability (fraction of nonzero values)
def compute_survival_probability(df):
    # survival_prob = (df > 1e-6).mean()  # Use a small threshold for numerical stability
    survival_prob = (df > 0).mean()  # Use a small threshold for numerical stability
    return survival_prob.clip(0, 1)  # Ensure values remain within [0, 1]

results_grouped = results.groupby(["alpha", "beta", "mcs"], as_index=False).agg(compute_survival_probability)

# Get unique MCS values
unique_mcs = results_grouped["mcs"].unique()

# Species columns
species_columns = [col for col in results_grouped.columns if col.startswith("s")]

# Generate heatmaps for each MCS separately
for mcs in unique_mcs:
    subset = results_grouped[results_grouped["mcs"] == mcs]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2 rows, 4 columns

    for ax, species in zip(axes.flat, species_columns):
        heatmap_data = subset.pivot(index="beta", columns="alpha", values=species)  # Swap axes
        sns.heatmap(heatmap_data, cmap="magma", ax=ax, annot=False, vmin=0, vmax=1)  # Set color bar range
        ax.set_title(f"Survival Probability of {species} (MCS={mcs})")
        ax.set_xlabel("Alpha")  # Swap label
        ax.set_ylabel("Beta")  # Swap label
        ax.invert_yaxis()  # Ensure beta values are increasing from bottom to top

    plt.tight_layout()
    plt.savefig(f"survival_probability_mcs_{mcs}.png")
    # plt.show()
