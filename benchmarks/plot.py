import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Define a consistent colour palette based on the original plot
palette = {
    'cuda': '#1f77b4',
    'cuda-max': '#ff7f0e',
    'metal': '#2ca02c',
    'metal-max': '#d62728',
    'single-threaded': '#9467bd'
}

# -------------------Combined Results-------------------

df = pd.read_csv("csvs/combined-escgs.csv")

# Convert 'time_seconds' to numeric, forcing errors to NaN
df['time_seconds'] = pd.to_numeric(df['time_seconds'], errors='coerce')

# Drop rows with NaN in 'time_seconds' (conversion failures)
df = df.dropna(subset=['time_seconds'])

# Convert 'length' to numeric, forcing errors to NaN
df['length'] = pd.to_numeric(df['length'], errors='coerce')
# Filter out lengths greater than 400
df = df[df['length'] <= 400]

# Group by implementation and lattice length, then take mean time
grouped = df.groupby(['implementation', 'length']).agg({'time_seconds': 'mean'}).reset_index()

# Plot using seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(data=grouped, x='length', y='time_seconds', hue='implementation', marker='o')

plt.title('Execution Time vs Lattice Size by Implementation')
plt.xlabel('Lattice Size (length)')
plt.ylabel('Execution Time (seconds)')
plt.grid(True)
plt.legend(title='Implementation')
plt.tight_layout()
plt.savefig("figs/comparisons.png")

# -------------------Implementation-------------------

df = pd.read_csv("csvs/combined-escgs.csv")
df['time_seconds'] = pd.to_numeric(df['time_seconds'], errors='coerce')
df['length'] = pd.to_numeric(df['length'], errors='coerce')

df = df.dropna(subset=['time_seconds', 'length'])

grouped = df.groupby(['implementation', 'length']).agg({'time_seconds': 'mean'}).reset_index()
for impl in grouped['implementation'].unique():
    data = grouped[grouped['implementation'] == impl]

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='length', y='time_seconds', marker='o', color=palette.get(impl, 'gray'))

    # # Prepare table data: headers + rows
    # table_data = [["Length", "Seconds"]]
    # table_data += [[int(row['length']), f"{row['time_seconds']:.1f}"] for _, row in data.iterrows()]

    # # Dynamically calculate bbox height based on row count
    # n_rows = len(table_data)
    # bbox_height = 0.05 * n_rows  # 5% of figure per row

    # # Create table with dynamic height, fully over the grid
    # table = plt.table(cellText=table_data,
    #                   colLabels=None,
    #                   cellLoc='center',
    #                   loc='upper left',
    #                   bbox=[0.1, 0.9 - bbox_height, 0.3, bbox_height],
    #                   zorder=5)

    # table.auto_set_font_size(False)
    # table.set_fontsize(9)

    # # Style table cells
    # for (row, col), cell in table.get_celld().items():
    #     cell.set_facecolor("white")
    #     cell.set_edgecolor("black")
    #     cell.set_text_props(weight="bold")
    #     cell.set_width(0.1)

    plt.title(f'Execution Time vs Lattice Size ({impl})')
    plt.xlabel('Lattice Size (length)')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.subplots_adjust(top=0.85)  # Move title slightly down to avoid overlap
    plt.tight_layout()
    plt.savefig(f"figs/{impl}.png")
    plt.close()
