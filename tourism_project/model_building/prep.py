
import pandas as pd
import os

from sklearn.model_selection import train_test_split

# Load the raw dataset
df = pd.read_csv("tourism_project/data/tourism.csv")

print("Original Dataset Shape:", df.shape)

# Remove exact duplicate records
df = df.drop_duplicates().copy()

# Standardize Gender
df["Gender"] = (
    df["Gender"]
    .str.strip()
    .replace({
        "Fe Male": "Female"
    })
)

# Standardize Marital Status
df["MaritalStatus"] = (
    df["MaritalStatus"]
    .str.strip()
    .replace({
        "Unmarried": "Single"
    })
)

# Remove unnecessary and post-interaction columns
# CustomerID and Unnamed: 0 are identifiers
# Sales interaction variables are excluded to prevent data leakage

columns_to_drop = [
    "Unnamed: 0",
    "CustomerID",
    "PitchSatisfactionScore",
    "ProductPitched",
    "NumberOfFollowups",
    "DurationOfPitch"
]

df.drop(
    columns=columns_to_drop,
    errors="ignore",
    inplace=True
)

# Separate independent and dependent variables
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Split dataset into training and testing sets
# Stratification preserves the target class distribution

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create output directory
OUTPUT_PATH = "tourism_project/model_building/data_splits"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Save training and testing datasets
Xtrain.to_csv(f"{OUTPUT_PATH}/Xtrain.csv", index=False)
Xtest.to_csv(f"{OUTPUT_PATH}/Xtest.csv", index=False)
ytrain.to_csv(f"{OUTPUT_PATH}/ytrain.csv", index=False)
ytest.to_csv(f"{OUTPUT_PATH}/ytest.csv", index=False)

# Print data preparation summary
print("\nData preparation completed successfully.")

print(f"Cleaned Dataset Shape: {df.shape}")

print(f"Xtrain Shape: {Xtrain.shape}")
print(f"Xtest Shape: {Xtest.shape}")
print(f"ytrain Shape: {ytrain.shape}")
print(f"ytest Shape: {ytest.shape}")

print("\nGender values retained:")
print(df["Gender"].value_counts())

print("\nMarital Status values retained:")
print(df["MaritalStatus"].value_counts())

print("\nTraining target distribution:")
print(ytrain.value_counts(normalize=True).round(3))

print("\nTesting target distribution:")
print(ytest.value_counts(normalize=True).round(3))

print("\nSaved files:")
print(os.listdir(OUTPUT_PATH))
