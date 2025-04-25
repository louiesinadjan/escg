import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("csvs/combined-numbers.csv")

# Compute average time per implementation
avg_times = df.groupby("implementation")["time_seconds"].mean().sort_values()

# Plotting
plt.figure(figsize=(10, 6))
bar_colour = "#4A90E2"
bars = avg_times.plot(kind="bar", color=bar_colour)

plt.title("Average Time per Implementation")
plt.ylabel("Time (seconds)")
plt.xlabel("Implementation")
plt.xticks(rotation=45, ha="right")

# Padding the y-axis limit
y_max = avg_times.max()
plt.ylim(top=y_max * 1.2)  # Add 20% headroom

# Add value labels
for i, value in enumerate(avg_times):
    plt.text(i, value * 1.05, f"{value:.6f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("figs/mersenne.png")
plt.show()
plt.close()
