import pandas as pd
import matplotlib.pyplot as plt
import os

# Load your CSV
df = pd.read_csv("pcs_fig5.csv")

# Ensure output directory exists
os.makedirs("plots_per_mcs", exist_ok=True)

# Define visual attributes
markers = ['^', 'o', 'o', 's', 's']
colors = ['red', 'green', 'blue', 'grey', 'black']

print("Rows where MCS = 0:")
print(df[df['mcs'] == 0])
print()
# Iterate through each unique MCS value
for mcs_val in sorted(df['mcs'].unique()):
    mcs_df = df[df['mcs'] == mcs_val]

    # Group by system size (length) and alpha
    grouped = mcs_df.groupby(['length', 'alpha'])
    ext_prob = grouped['s5'].apply(lambda x: (x == 0).sum() / len(x)).reset_index()
    ext_prob.rename(columns={'s5': 'extinction_prob'}, inplace=True)
    system_sizes = sorted(ext_prob['length'].unique())

    # Plot
    plt.figure(figsize=(10, 6))
    for i, size in enumerate(system_sizes):
        data = ext_prob[ext_prob['length'] == size]
        plt.plot(data['alpha'], data['extinction_prob'],
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
    plt.ylabel("Extinction probability of Species 5")
    plt.legend(title="System size")
    plt.ylim(-0.01, 1.01)
    plt.grid(True)
    plt.title(f"Extinction Probability of Species 5 at MCS={mcs_val}")
    plt.tight_layout()
    plt.savefig(f"plots_per_mcs/fig5_mcs_{mcs_val}.png")
    plt.close()