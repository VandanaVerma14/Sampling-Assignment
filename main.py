import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from load_data import load_dataset
from balance_data import balance_dataset
from sampling import *
from models import get_models

# Step 1: Load dataset
df = load_dataset()

# Step 2: Balance dataset
balanced_df = balance_dataset(df)

# Step 3: Create five samples
# Step 3: Create five samples
samples = {
    "Sampling1": simple_random_sampling(balanced_df),
    "Sampling2": systematic_sampling(balanced_df),
    "Sampling3": stratified_sampling(balanced_df),
    "Sampling4": cluster_sampling(balanced_df),
    "Sampling5": bootstrap_sampling(balanced_df)
}

models = get_models()
results = []

# Step 4: Train models and calculate accuracy
for sampling_name, sample_df in samples.items():
    X = sample_df.drop("Class", axis=1)
    y = sample_df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100

        results.append([model_name, sampling_name, round(acc, 2)])

# Convert to DataFrame (long format)
results_df = pd.DataFrame(
    results, columns=["Model", "Sampling", "Accuracy"]
)

# Convert to wide format (LIKE ASSIGNMENT TABLE)
final_table = results_df.pivot(
    index="Model",
    columns="Sampling",
    values="Accuracy"
)

# Optional: sort rows and columns nicely
final_table = final_table.sort_index()
final_table = final_table[["Sampling1", "Sampling2", "Sampling3", "Sampling4", "Sampling5"]]

# Save and display
final_table.to_csv("../results/accuracy_table.csv")

print("\nFinal Accuracy Table (Assignment Format):\n")
print(final_table)

