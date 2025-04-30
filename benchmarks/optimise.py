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
df_metal = df[df['implementation'].isin(['metal', 'metal-max'])]

# Filter for 'cuda' and 'cuda-max' implementations
df_cuda = df[df['implementation'].isin(['cuda', 'cuda-max'])]

# Create side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot metal data
sns.scatterplot(data=df_metal, x='run', y='time_seconds', hue='implementation', ax=ax1)
ax1.set_title('Metal Execution Time for L = 300', fontsize=14)
ax1.set_xlabel('Trial', fontsize=12)
ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
ax1.grid(True)

# Plot cuda data
sns.scatterplot(data=df_cuda, x='run', y='time_seconds', hue='implementation', ax=ax2)
ax2.set_title('CUDA Execution Time for L = 300', fontsize=14)
ax2.set_xlabel('Trial', fontsize=12)
ax2.set_ylabel('Execution Time (seconds)', fontsize=12)
ax2.grid(True)

plt.tight_layout()
plt.savefig("figs/metal_cuda_warmup.png")
plt.show()
