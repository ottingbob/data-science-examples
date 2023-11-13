# Based on the work in this guide:
# https://www.kaggle.com/code/pmarcelino/comprehensive-data-exploration-with-python/notebook

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

df_train = pd.read_csv(
    "./scripts/house-prices-advanced-regression-techniques/train.csv"
)

# Check columns
print(df_train.columns)

# Get the descriptive statistics
print(df_train["SalePrice"].describe())

# Print histogram
# sns.distplot(df_train["SalePrice"])
# plt.show()

# Skewness and kurtosis
print(f"Skewness: {df_train['SalePrice'].skew()}")
print(f"Kurtosis: {df_train['SalePrice'].kurt()}")


def plot_features_vs_sale_price():
    # Scatter plot grlivarea / saleprice
    data = pd.concat([df_train["SalePrice"], df_train["GrLivArea"]], axis=1)
    data.plot.scatter(x="GrLivArea", y="SalePrice", ylim=(0, 800_000))
    plt.show()

    # Scatter plot totalbsmtsf / saleprice
    data = pd.concat([df_train["SalePrice"], df_train["TotalBsmtSF"]], axis=1)
    data.plot.scatter(x="TotalBsmtSF", y="SalePrice", ylim=(0, 800_000))
    plt.show()

    # Box plot overallqual / saleprice
    data = pd.concat([df_train["SalePrice"], df_train["OverallQual"]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x="OverallQual", y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800_000)
    plt.show()

    data = pd.concat([df_train["SalePrice"], df_train["YearBuilt"]], axis=1)
    f, ax = plt.subplots(figsize=(16, 8))
    fig = sns.boxplot(x="YearBuilt", y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800_000)
    plt.xticks(rotation=90)
    plt.show()


def plot_correlations():
    # Correlation matrix heatmap style
    corr_mat = df_train.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr_mat, vmax=0.8, square=True)
    plt.show()

    # Saleprice correlation matrix
    # Number of variables for heatmap
    k = 10
    cols = corr_mat.nlargest(k, "SalePrice")["SalePrice"].index
    cm = np.corrcoef(df_train[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(
        cm,
        cbar=True,
        annot=True,
        square=True,
        fmt="0.2f",
        annot_kws={"size": 10},
        yticklabels=cols.values,
        xticklabels=cols.values,
    )
    plt.tight_layout()
    plt.show()

    # Scatter plots
    sns.set()
    cols = [
        "SalePrice",
        "OverallQual",
        "GrLivArea",
        "TotalBsmtSF",
        "FullBath",
        "YearBuilt",
    ]
    sns.pairplot(df_train[cols], size=2.5)
    plt.show()


# Treat missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(
    ascending=False
)
missing_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
print(missing_data.head(20))

# Consider more than 15% of missing data we remove the related variable
df_train = df_train.drop((missing_data[missing_data["Total"] > 1]).index, 1)
df_train = df_train.drop(df_train.loc[df_train["Electrical"].isnull()].index)
# Double check that missing data is removed
print(df_train.isnull().sum().max())

# Standardize the data
sale_price_scaled = StandardScaler().fit_transform(df_train["SalePrice"][:, np.newaxis])
low_range = sale_price_scaled[sale_price_scaled[:, 0].argsort()][:10]
high_range = sale_price_scaled[sale_price_scaled[:, 0].argsort()][-10:]
print("Outer range (low) of the distribution:")
print(low_range)
print("Outer range (high) of the distribution:")
print(high_range)

# Bivariate analysis saleprice / grlivarea
data = pd.concat([df_train["SalePrice"], df_train["GrLivArea"]], axis=1)
# data.plot.scatter(x="GrLivArea", y="SalePrice", ylim=(0, 800_000))
# plt.show()

# Remove the outliers
df_train.sort_values(by="GrLivArea", ascending=False)[:2]
df_train = df_train.drop(df_train[df_train["Id"] == 1299].index)
df_train = df_train.drop(df_train[df_train["Id"] == 524].index)

# Bivariate analysis saleprice / totalbsmtsf
data = pd.concat([df_train["SalePrice"], df_train["TotalBsmtSF"]], axis=1)
# data.plot.scatter(x="TotalBsmtSF", y="SalePrice", ylim=(0, 800_000))
# plt.show()

# Now test the 4 assumptions:
# 1) Normality
# 2) Homoscedasticity
# 3) Linearity
# 4) Absence of correlated errors


def plot_hist_and_norm(df_train: pd.DataFrame, col_name: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    # Plot the values before the log transformation
    sns.distplot(df_train[col_name], fit=stats.norm, ax=axes[0][0])
    stats.probplot(df_train[col_name], plot=axes[0][1])

    # Plot values after log transform
    df_train[col_name] = np.log(df_train[col_name])

    sns.distplot(df_train[col_name], fit=stats.norm, ax=axes[1][0])
    stats.probplot(df_train[col_name], plot=axes[1][1])
    plt.show()


def verbose_plots():
    # Histogram and normal probability plot
    sns.distplot(df_train["SalePrice"], fit=stats.norm)
    fig = plt.figure()
    res = stats.probplot(df_train["SalePrice"], plot=plt)
    plt.show()

    # Sale price is not normal and it shows `peakedness` and positive skewness
    # and does not follow the diagonal line on the probability plot.

    # With that being said we can apply log transformations when we see
    # positive skewness and that should help out

    # Applying the log transformation
    df_train["SalePrice"] = np.log(df_train["SalePrice"])

    # Transformed histogram and normal probability plot
    sns.distplot(df_train["SalePrice"], fit=stats.norm)
    fig = plt.figure()
    res = stats.probplot(df_train["SalePrice"], plot=plt)
    plt.show()

    # Now check the same with the grlivarea
    sns.distplot(df_train["GrLivArea"], fit=stats.norm)
    fig = plt.figure()
    res = stats.probplot(df_train["GrLivArea"], plot=plt)
    plt.show()

    # We see skewness again so apply a log transformation
    df_train["GrLivArea"] = np.log(df_train["GrLivArea"])

    # Transformed histogram and normal probability plot
    sns.distplot(df_train["GrLivArea"], fit=stats.norm)
    fig = plt.figure()
    res = stats.probplot(df_train["GrLivArea"], plot=plt)
    plt.show()


# Histogram and normal probability plots
plot_hist_and_norm(df_train=df_train, col_name="SalePrice")
plot_hist_and_norm(df_train=df_train, col_name="GrLivArea")

# Now check the totalbsmtsf
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
sns.distplot(df_train["TotalBsmtSF"], fit=stats.norm, ax=axes[0][0])
stats.probplot(df_train["TotalBsmtSF"], plot=axes[0][1])

# This case is a little different. Although it deals with skewness we
# also have a lot of observations with zero values and zero doesn't
# allow us to use log transformations

# So we create a column to identify if there is a basement, and if there
# is we will apply the transformation on that observation

# Create a column for new variable
df_train["HasBsmt"] = pd.Series(len(df_train["TotalBsmtSF"]), index=df_train.index)
df_train["HasBsmt"] = 0
df_train.loc[df_train["TotalBsmtSF"] > 0, "HasBsmt"] = 1

# Transform the data
df_train.loc[df_train["HasBsmt"] == 1, "TotalBsmtSF"] = np.log(df_train["TotalBsmtSF"])

# Histogram and normal probability plot
sns.distplot(
    df_train[df_train["TotalBsmtSF"] > 0]["TotalBsmtSF"], fit=stats.norm, ax=axes[1][0]
)
stats.probplot(df_train[df_train["TotalBsmtSF"] > 0]["TotalBsmtSF"], plot=axes[1][1])
plt.show()

# Now test for `homoscedasticity`
# Start with a scatter plot between saleprice / grlivarea
plt.scatter(df_train["GrLivArea"], df_train["SalePrice"])
plt.show()

# The shape is no longer conical since we applied the log transformations
# so we fixed the homoscedasticity problem !!

# Check saleprice / totalbsmtsf
obs_with_basement = df_train[df_train["TotalBsmtSF"] > 0]
plt.scatter(obs_with_basement["TotalBsmtSF"], obs_with_basement["SalePrice"])
plt.show()

# Convert categorical values with dummies
df_train = pd.get_dummies(df_train)
