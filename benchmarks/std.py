import pandas as pd


df = pd.read_csv("csvs/combined-escgs.csv")

# Ensure relevant columns are correctly typed
df['time_seconds'] = pd.to_numeric(df['time_seconds'], errors='coerce')
df['length'] = pd.to_numeric(df['length'], errors='coerce')

# Drop rows with missing values
df = df.dropna(subset=['time_seconds', 'length'])

# Group by implementation and length and calculate standard deviation
stds = df.groupby(['implementation', 'length'])['time_seconds'].std().reset_index()

# Print the results
for impl in stds['implementation'].unique():
    print(f"Implementation: {impl}")
    impl_data = stds[stds['implementation'] == impl]
    for _, row in impl_data.iterrows():
        print(f"  Length {int(row['length'])}: Std Dev = {row['time_seconds']:.4f} seconds")
    print()
