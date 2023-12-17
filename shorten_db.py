import pandas as pd

df = pd.read_csv("data/dataset.csv")
df = df.sample(frac=0.04)
df.to_csv("data/dataset_min.csv", index=False)