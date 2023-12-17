import pandas as pd

# Assuming 'df' is your DataFrame with the dataset
df = pd.read_csv("data/dataset.csv")
x = list(df.columns).index("400")
df = df.iloc[:,x:]
correlation_matrix = df.corr()

# Display the correlation matrix
print(correlation_matrix)
correlation_matrix.to_csv('correlation_matrix.csv', index=True)