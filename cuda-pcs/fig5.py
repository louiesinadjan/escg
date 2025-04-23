import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("your_data.csv")

# Group by system size (length) and alpha
grouped = df.groupby(['length', 'alpha'])

# Calculate 1 - average density of species 5 for each group
density = grouped['s5'].mean().reset_index()
density['1-rho_5'] = 1 - density['s5']

# Get unique system sizes and assign markers/colors
system_sizes = sorted(density['length'].unique())
markers = ['^', 'o', 'o', 's', 's']
colors = ['red', 'green', 'blue', 'grey', 'black']

# Plot
plt.figure(figsize=(10, 6))
for i, size in enumerate(system_sizes):
    data = density[density['length'] == size]
    plt.plot(data['alpha'], data['1-rho_5'],
             label=str(size),
             marker=markers[i],
             color=colors[i],
             linestyle='-')

# Add vertical solution phase labels manually
plt.text(0.02, 0.99, "all", fontsize=12, ha='left', va='bottom')
plt.text(0.2, 1.01, r"$A^7_{\{0,1,2,3,4,6,7\}}$", fontsize=12, ha='center')
plt.text(0.385, 1.01, r"$A^4_{\{1,3,5,7\}}$", fontsize=12, ha='right')

# Labels and legend
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$1 - \rho_5$")
plt.legend(title="System size")
plt.ylim(0.75, 1.01)
plt.grid(True)
plt.title("Absence of Species 5 vs Alpha for Various System Sizes")

plt.tight_layout()
plt.show()