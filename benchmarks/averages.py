import pandas as pd

df = pd.read_csv("csvs/combined-escgs.csv")

df['time_seconds'] = pd.to_numeric(df['time_seconds'], errors='coerce')
df['length'] = pd.to_numeric(df['length'], errors='coerce') 
df = df.dropna(subset=['time_seconds', 'length'])

grouped = df.groupby(['implementation', 'length']).agg({'time_seconds': 'mean'}).reset_index()

# Print to escgs.csv
grouped.to_csv("escgs.csv", index=False)

#-------------------------------------------------------------------------------------------------------

df = pd.read_csv("csvs/randoms.csv")
df['time_seconds'] = pd.to_numeric(df['time_seconds'], errors='coerce')
df['numRandoms'] = pd.to_numeric(df['numRandoms'], errors='coerce')
df = df.dropna(subset=['time_seconds'])
grouped = df.groupby(['numRandoms']).agg({'time_seconds': 'mean'}).reset_index()
# Print to randoms.csv
grouped.to_csv("randoms.csv", index=False)

#-------------------------------------------------------------------------------------------------------
df = pd.read_csv("csvs/combined-numbers.csv")
df['time_seconds'] = pd.to_numeric(df['time_seconds'], errors='coerce')
grouped = df.groupby(['implementation']).agg({'time_seconds': 'mean'}).reset_index()
grouped.to_csv("numbers.csv", index=False)



