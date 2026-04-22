"""
Handling Missing Values in Machine Learning
Dataset: Titanic (Seaborn Built-in Dataset)

This script demonstrates how to:

1. Detect missing values
2. Calculate missing-value percentage
3. Drop columns with excessive missing values
4. Apply mean imputation (numerical feature)
5. Apply mode imputation (categorical feature)
6. Visualize missing values before and after cleaning
7. Perform advanced KNN imputation

Author: Hamna
Project: Machine-Learning-Roadmap
"""

# =============================
# Import Required Libraries
# =============================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer


# =============================
# Load Dataset
# =============================

df = sns.load_dataset("titanic")

print("\nFirst 5 rows of dataset:\n")
print(df.head())


# =============================
# Detect Missing Values
# =============================

print("\nMissing values in each column:\n")
print(df.isnull().sum())


# =============================
# Missing Value Percentage
# =============================

missing_percentage = (df.isnull().sum() / len(df)) * 100

print("\nMissing value percentage:\n")
print(missing_percentage)


# =============================
# Visualize Missing Values
# =============================

plt.figure(figsize=(10, 6))

sns.heatmap(df.isnull(), cbar=False, cmap="viridis")

plt.title("Missing Values Before Handling")

plt.show()


# =============================
# Drop Column with Too Many Missing Values
# =============================

df.drop("deck", axis=1, inplace=True)

print("\nDropped column: deck")


# =============================
# Mean Imputation (Age Column)
# =============================

df["age"].fillna(df["age"].mean(), inplace=True)

print("\nMissing values in 'age' after mean imputation:")
print(df["age"].isnull().sum())


# =============================
# Mode Imputation (Embarked Column)
# =============================

df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)

print("\nMissing values in 'embarked' after mode imputation:")
print(df["embarked"].isnull().sum())


# =============================
# Mode Imputation (Embark Town Column)
# =============================

df["embark_town"].fillna(df["embark_town"].mode()[0], inplace=True)

print("\nMissing values in 'embark_town' after mode imputation:")
print(df["embark_town"].isnull().sum())


# =============================
# Verify Remaining Missing Values
# =============================

print("\nRemaining missing values:\n")
print(df.isnull().sum())


# =============================
# Visualize After Handling Missing Values
# =============================

plt.figure(figsize=(10, 6))

sns.heatmap(df.isnull(), cbar=False, cmap="viridis")

plt.title("Missing Values After Handling")

plt.show()


# =============================
# Advanced Technique: KNN Imputation
# =============================

print("\nApplying KNN Imputation on numeric columns...")

df_knn = sns.load_dataset("titanic")

numeric_df = df_knn.select_dtypes(include=np.number)

imputer = KNNImputer(n_neighbors=5)

numeric_imputed = imputer.fit_transform(numeric_df)

numeric_imputed = pd.DataFrame(
    numeric_imputed,
    columns=numeric_df.columns
)

print("\nMissing values after KNN imputation:\n")
print(numeric_imputed.isnull().sum())


# =============================
# Visualization Before vs After KNN
# =============================

plt.figure(figsize=(10, 5))

sns.heatmap(numeric_df.isnull(), cbar=False)

plt.title("Before KNN Imputation")

plt.show()


plt.figure(figsize=(10, 5))

sns.heatmap(numeric_imputed.isnull(), cbar=False)

plt.title("After KNN Imputation")

plt.show()


# =============================
# Script Completed
# =============================

print("\nMissing value handling completed successfully.")
