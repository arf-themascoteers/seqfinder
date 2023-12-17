import pandas as pd
import os

for f in os.listdir("data"):
    d = os.path.join("data",f)
    df = pd.read_csv(d)
    print(d,len(df),len(df.columns))