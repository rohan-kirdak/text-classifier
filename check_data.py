import pandas as pd

df = pd.read_csv("data/spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]

print(df.shape)
print(df.head())
print(df["label"].value_counts())
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())