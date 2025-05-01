import pandas as pd
import matplotlib.pyplot as plt
import os

# Load your CSV
df = pd.read_csv("fig5.csv")

df['mcs'] = pd.to_numeric(df['mcs'], errors='coerce')
df = df.dropna(subset=['mcs'])
df['mcs'] = df['mcs'].astype(int)


df['length'] = pd.to_numeric(df['length'], errors='coerce')
df = df.dropna(subset=['length'])
df['length'] = df['length'].astype(int)

# Convert s0–s7 columns to float
sensor_cols = ['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7']
df[sensor_cols] = df[sensor_cols].apply(pd.to_numeric, errors='coerce')

# Ensure output directory exists
os.makedirs("plots_per_mcs", exist_ok=True)

# Define visual attributes
markers = ['^', 'o', 'o', 's', 's']
colors = ['red', 'green', 'blue', 'grey', 'black']

# Iterate through each unique MCS value
for mcs_val in sorted(df['mcs'].unique()):
    mcs_df = df[df['mcs'] == mcs_val]
    extinction_prob = (
        mcs_df.groupby(['alpha', 'length'])['s5']
        .apply(lambda x: (x == 0).sum() / len(x))
        .reset_index(name='extinction_prob')
    )
    print(f"MCS = {mcs_val}")
    print(extinction_prob)

    # Plot extinction probability grouped by length
    plt.figure()
    for length_val in sorted(extinction_prob['length'].unique()):
        subset = extinction_prob[extinction_prob['length'] == length_val]
        plt.plot(subset['alpha'], subset['extinction_prob'], label=f'Length {length_val}', marker='o')

    plt.xlabel('Alpha')
    plt.ylabel('Extinction Probability')
    plt.title(f'Extinction Probability vs Alpha (MCS = {mcs_val})')
    plt.legend(title='Length')
    plt.grid(True)
    plt.title(f"Extinction Probability of Species 5 at MCS={mcs_val}")
    plt.tight_layout()
    plt.savefig(f"plots_per_mcs/fig5_mcs_{mcs_val}.png")
    plt.close()
    
# Print standard deviation of extinction probabilities for each system size and MCS
for mcs_val in sorted(df['mcs'].unique()):
    mcs_df = df[df['mcs'] == mcs_val]
    grouped = mcs_df.groupby(['length', 'alpha'])
    ext_prob = grouped['s5'].apply(lambda x: (x == 0).sum() / len(x)).reset_index()
    ext_prob.rename(columns={'s5': 'extinction_prob'}, inplace=True)
    system_sizes = sorted(ext_prob['length'].unique())

    print(f"Standard deviations at MCS={mcs_val}:")
    for size in system_sizes:
        data = ext_prob[ext_prob['length'] == size]
        std_dev = data['extinction_prob'].std()
        print(f"  System size {size}: {std_dev}")
    print()
    
count = len(df[(df['length'] == 100) & (df['mcs'] == 1000)])
print(f"Number of trials where length=100 and mcs=1000: {count}")