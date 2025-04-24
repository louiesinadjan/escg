import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



df = pd.read_csv("combined-escgs.csv")

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
plt.show()
