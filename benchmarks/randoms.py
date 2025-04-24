import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("combined-numbers.csv")

# Compute average time per implementation
avg_times = df.groupby("implementation")["time_seconds"].mean().sort_values()

# Plotting with custom bar colour
plt.figure(figsize=(10, 6))
bar_colour = "#4A90E2"  # Soft blue
bars = avg_times.plot(kind="bar", color=bar_colour)

plt.yscale("log")
plt.title("Average Time per Implementation (Log Scale)")
plt.ylabel("Time (seconds, log scale)")
plt.xlabel("Implementation")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Add decimal-formatted labels
for i, value in enumerate(avg_times):
    plt.text(i, value * 1.1, f"{value:.6f}", ha="center", va="bottom", fontsize=9)

plt.show()
