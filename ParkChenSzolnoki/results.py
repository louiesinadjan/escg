import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import results.csv
results = pd.read_csv('results2.csv')

# Clean column names (remove leading/trailing spaces)
results.columns = results.columns.str.strip()

# Define species columns as s0 to s7
species_columns = [f"s{i}" for i in range(8)]

# Compute number of surviving species (species with value > 0)
def count_surviving_species(row):
    return (row[species_columns] > 0).sum()

results["S"] = results.apply(count_surviving_species, axis=1)

# Group by alpha, beta, mcs and count how often each S occurs
grouped = results.groupby(["alpha", "beta", "mcs"])["S"].value_counts().unstack(fill_value=0).reset_index()

# Convert S columns to strings for consistency
grouped.columns = ["alpha", "beta", "mcs"] + [str(col) for col in grouped.columns[3:]]

print("Grouped columns:", grouped.columns)

# Ensure columns for survival counts S=1 to 8 exist, add missing with 0
species_labels = [str(i) for i in range(1, 9)]
for label in species_labels:
    if label not in grouped.columns:
        grouped[label] = 0

# Reorder columns so that species survival counts are in order
grouped = grouped[["alpha", "beta", "mcs"] + species_labels]

# Cast species columns to float to avoid dtype warnings
grouped[species_labels] = grouped[species_labels].astype(float)

# Normalize to get probabilities (divide by total runs per group)
def normalize_group(df):
    total = df[species_labels].sum(axis=1)
    # Avoid division by zero; if total==0, values remain 0
    df.loc[total != 0, species_labels] = df.loc[total != 0, species_labels].div(total[total != 0], axis=0)
    return df

grouped = normalize_group(grouped)

# For each unique mcs value, generate a separate heatmap figure
unique_mcs = sorted(grouped["mcs"].unique())
for mcs_val in unique_mcs:
    # Filter for the current mcs value
    grouped_mcs = grouped[grouped["mcs"] == mcs_val]

    # Melt data for the current mcs value
    melted = grouped_mcs.melt(id_vars=["alpha", "beta", "mcs"], value_vars=species_labels,
                              var_name="S", value_name="probability")

    # Generate a 2x4 grid of heatmaps for S = 1 to 8
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2 rows, 4 columns
    for ax, s in zip(axes.flat, species_labels):
        subset = melted[melted["S"] == s]
        heatmap_data = subset.pivot(index="beta", columns="alpha", values="probability")
        sns.heatmap(heatmap_data, cmap="jet", ax=ax, annot=False, vmin=0, vmax=1)
        ax.set_title(f"S={s}")
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Beta")
        ax.invert_yaxis()

    plt.suptitle(f"mcs = {mcs_val}", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"species_survival_by_S_mcs_{mcs_val}.png")
    plt.close(fig)