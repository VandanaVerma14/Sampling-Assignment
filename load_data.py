import pandas as pd

def load_dataset():
    df = pd.read_csv("../data/creditcard_data.csv")
    print("Original Class Distribution:")
    print(df["Class"].value_counts())
    return df
