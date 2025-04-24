import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("randoms.csv")

df['time_seconds'] = pd.to_numeric(df['time_seconds'], errors='coerce')
df['numRandoms'] = pd.to_numeric(df['numRandoms'], errors='coerce')
df = df.dropna(subset=['time_seconds'])

# Group by numRandoms only
grouped = df.groupby(['numRandoms']).agg({'time_seconds': 'mean'}).reset_index()

from matplotlib.ticker import FuncFormatter

plt.figure(figsize=(10, 6))
sns.lineplot(data=grouped, x='numRandoms', y='time_seconds', marker='o')

plt.title('Execution Time with Different --numRandoms')
plt.xlabel('numRandoms')
plt.ylabel('Execution Time (seconds)')
plt.grid(True)

# Apply comma formatting to x-axis
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout()
plt.savefig("numRandoms.png")
