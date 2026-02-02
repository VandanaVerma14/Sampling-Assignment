from imblearn.over_sampling import SMOTE
import pandas as pd

def balance_dataset(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)

    balanced_df = pd.concat([X_bal, y_bal], axis=1)

    print("\nBalanced Class Distribution:")
    print(balanced_df["Class"].value_counts())

    return balanced_df
