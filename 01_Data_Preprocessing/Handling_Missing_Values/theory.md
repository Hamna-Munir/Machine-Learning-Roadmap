# Handling Missing Values

## Overview

Handling missing values is an essential step in **data preprocessing**. Real-world datasets often contain incomplete observations due to data entry errors, system failures, or unavailable information.

If missing values are not handled properly, they can negatively impact model performance and lead to incorrect predictions.

This section explains the types of missing data and commonly used techniques to handle them in Machine Learning workflows.

---

## Why Missing Values Occur

Missing values may appear due to:

- Errors during data collection
- Equipment malfunction
- Human mistakes in data entry
- Privacy restrictions
- Incomplete survey responses
- Data merging from multiple sources

Understanding the reason behind missing data helps select the correct handling strategy.

---

## Types of Missing Data

### 1. Missing Completely at Random (MCAR)

Missing values have no relationship with any variable in the dataset.

Example:
A sensor randomly fails to record temperature values.

Impact:
Removing rows usually does not introduce bias.

---

### 2. Missing at Random (MAR)

Missing values depend on another feature but not on the missing feature itself.

Example:
Income data missing more frequently for younger individuals.

Impact:
Can be handled using statistical imputation methods.

---

### 3. Missing Not at Random (MNAR)

Missing values depend on the feature itself.

Example:
People with high income choosing not to disclose their salary.

Impact:
More complex handling techniques are required.

---

## Identifying Missing Values

Common ways to detect missing values:

- Null value inspection
- Summary statistics
- Dataset information overview
- Visualization techniques

Typical indicators:

- NaN
- NULL
- None
- Blank cells


### Example in Python:

```python
df.isnull().sum()
```
---

## Techniques for Handling Missing Values

### 1. Removing Missing Values

Rows or columns containing missing values can be removed.

#### Removing Rows

Useful when missing values are very few.

#### Example:
```python
df.dropna(axis=0)
```
#### Removing Columns

Useful when a column contains too many missing values.

#### Example:
```python
df.dropna(axis=1)
```
Limitations:

- May reduce dataset size
- Risk of losing important information

---

 
### 2. Mean Imputation

Replace missing values with the mean of the column.

Suitable for:

- Numerical features
- Normally distributed data

#### Example:
```python
df["Age"].fillna(df["Age"].mean(), inplace=True)
```
---

### 3. Median Imputation

Replace missing values with the median value.

Suitable for:

- Numerical features
- Data with outliers

#### Example:
```python
df["Salary"].fillna(df["Salary"].median(), inplace=True)
```

---

### 4. Mode Imputation

Replace missing values with the most frequent value.

Suitable for:

- Categorical features

#### Example:
```python
df["City"].fillna(df["City"].mode()[0], inplace=True)
```
---

### 5. Forward Fill

Fill missing values using previous row values.

#### Example:
```python
df.fillna(method="ffill", inplace=True)
```
Useful for:

Time-series datasets

---

### 6. Backward Fill

Fill missing values using next row values.

#### Example:
```python
df.fillna(method="bfill", inplace=True)
```
---

### 7. Interpolation

Estimate missing values using surrounding observations.

#### Example:
```python
df.interpolate(method="linear", inplace=True)
```
Useful for:

Continuous numerical data

---

### 8. Using Machine Learning Models

Missing values can be predicted using regression or classification algorithms.

#### Examples:

- KNN Imputation
- Regression Imputation
- Iterative Imputation

These methods improve accuracy compared to simple statistical replacements.

---

### Choosing the Right Strategy

Selection depends on:

- Percentage of missing data
- Type of feature (categorical or numerical)
- Dataset size
- Presence of outliers
- Business context

General guideline:

Small missing percentage → remove rows

Moderate missing percentage → statistical imputation

Large missing percentage → advanced imputation techniques

---

### Advantages of Handling Missing Values Properly

- Improves model accuracy
- Prevents biased predictions
- Maintains dataset consistency
- Supports better feature learning

---
### Summary

Handling missing values is a critical preprocessing step in Machine Learning pipelines. Selecting an appropriate strategy ensures reliable model training and improves prediction performance.

Common approaches include:

- Removing rows or columns
- Mean imputation
- Median imputation
- Mode imputation
- Forward fill and backward fill
- Interpolation
Machine learning–based imputation
