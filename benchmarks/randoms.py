import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Load and clean the data
df = pd.read_csv("csvs/randoms.csv")
df.columns = df.columns.str.strip()

df['time_seconds'] = pd.to_numeric(df['time_seconds'], errors='coerce')
df['numRandoms'] = pd.to_numeric(df['numRandoms'], errors='coerce')
df['length'] = pd.to_numeric(df['length'], errors='coerce')
df = df.dropna(subset=['time_seconds'])

# Group by both numRandoms and length
grouped = df.groupby(['numRandoms', 'length']).agg({'time_seconds': 'mean'}).reset_index()

# Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=grouped, x='numRandoms', y='time_seconds', hue='length', marker='o')

plt.title('Execution Time with Different --numRandoms (Grouped by Length)')
plt.xlabel('numRandoms')
plt.ylabel('Execution Time (seconds)')
plt.grid(True)

# Format x-axis with commas
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout()
plt.savefig("figs/numRandoms.png")
plt.show()
