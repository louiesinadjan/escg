import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
df = pd.read_csv("csvs/metal-optimising.csv")

# Convert relevant columns to numeric
df['time_seconds'] = pd.to_numeric(df['time_seconds'], errors='coerce')
df['length'] = pd.to_numeric(df['length'], errors='coerce')
df['run'] = pd.to_numeric(df['run'], errors='coerce')

# Filter for 'metal' and 'metal-max' implementations
df_filtered = df[df['implementation'].isin(['metal', 'metal-max'])]

# Plot using seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_filtered, x='run', y='time_seconds', hue='implementation', marker='o')

# Add titles and labels
plt.title('Metal Execution Time for L = 300')
plt.xlabel('Trial')
plt.ylabel('Execution Time (seconds)')
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig("figs/metal-warmup.png")

df_filtered = df[df['implementation'].isin(['cuda', 'cuda-max'])]

# Plot using seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_filtered, x='run', y='time_seconds', hue='implementation', marker='o')

# Add titles and labels
plt.title('CUDA Execution Time for L = 300')
plt.xlabel('Trial')
plt.ylabel('Execution Time (seconds)')
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig("figs/cuda-no-warmup.png")

