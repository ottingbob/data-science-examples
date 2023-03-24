# Adapted from this medium article:
# https://towardsdatascience.com/tips-and-tricks-for-working-with-strings-in-polars-ec6bb74aeec2

from pathlib import Path

import numpy as np
import polars as pl

# All headers in Polars must be of string type
df = pl.DataFrame(np.random.randint(0, 1, size=(10, 4)), schema=list("CDAB"))
print(df)

try:
    df = pl.DataFrame(np.random.randint(0, 1, size=(10, 4)), schema=list(range(1, 5)))
except TypeError:
    print("you tried to make numeric column headers huh..?")

# Sorting columns alphabetically
print(df.select(sorted(df.columns)))
# Sorting columns reversed
print(df.select(sorted(df.columns, reverse=True)))
# Sorting using array notation is not recommended if you want to evaluate
# using lazy evaluation:
# df[sorted(df.columns, reverse=True)]

names_csv = """
name,age
Kristopher Ruch,23
Lakiesha Halton,45
Yun Frei,23
Sharyn Llanos,76
Lorretta Herren,21
Merrilee Akana,67
Boyd Gilman,89
Heidy Smalling,11
Leta Batalla,45
Siu Wayne,67
Sammie Gurule,23
Jayne Whetzel,11
Byron Doggett,67
Luke Alcazar,90
Petra Doutt,12
Tula Parkhurst,67
Davina Hess,26
Enda Cornelius,78
Merlyn Cora,89
Jeanett Hardeman,34
"""
names_csv_file = Path("/tmp/names.csv")
with open(names_csv_file, "w") as csv_file:
    csv_file.write(names_csv)
q = pl.scan_csv(names_csv_file)
print(q.collect())

# Use the `lengths()` function to get the length of each name and store it in a new column
q = pl.scan_csv(names_csv_file).select(
    [
        "name",
        "age",
        pl.col("name").str.lengths().alias("length_of_name"),
    ]
)
print(q.collect())

# Select columns based on headers
# Here we use the titanic dataset from:
# https://www.kaggle.com/datasets/tedllh/titanic-train
q = pl.scan_csv("./scripts/titanic_train.csv")
print(q.collect())

# Only retrieve the `Name` and `Age` columns
q = pl.scan_csv("./scripts/titanic_train.csv").select(["Name", "Age"])
print(q.collect())

# All columns EXCEPT the `PassengerId` column
q = pl.scan_csv("./scripts/titanic_train.csv").select(pl.exclude("PassengerId"))
print(q.collect())

# Exclude supports regex so find all columns except those that start with `S`
q = pl.scan_csv("./scripts/titanic_train.csv").select(pl.exclude(r"^S.*$"))
print(q.collect())

# Get specific columns that start with `S`
q = pl.scan_csv("./scripts/titanic_train.csv").select(pl.col(r"^S.*$"))
print(q.collect())

# Filter rows with regex. Find all names ending with `William`
q = (
    pl.scan_csv("./scripts/titanic_train.csv")
    .filter(pl.col("Name").str.contains(r"William$"))
    .select(["Name"])
)
print(q.collect())

# More regex queries:
# Upper or lower case `w`
q = (
    pl.scan_csv("./scripts/titanic_train.csv")
    .filter(pl.col("Name").str.contains(r"[Ww]illiam"))
    .select(["Name"])
)
print(q.collect())

# multiple `i`s
q = (
    pl.scan_csv("./scripts/titanic_train.csv")
    .filter(pl.col("Name").str.contains(r"(?i)illiam"))
    .select(["Name"])
)
print(q.collect())

# Starts with `William`
q = (
    pl.scan_csv("./scripts/titanic_train.csv")
    .filter(pl.col("Name").str.contains(r"^William"))
    .select(["Name"])
)
print(q.collect())

# Names that do NOT end with William
# Polars does not support look-around / look-ahead / look-behind for regex
q = (
    pl.scan_csv("./scripts/titanic_train.csv")
    .filter(pl.col("Name").str.contains(r"William$").is_not())
    .select(["Name"])
)
print(q.collect())

# Splitting string columns
# Split the name column based on the space
q = pl.scan_csv(names_csv_file).select(
    [
        "name",
        pl.col("name").str.split(" ").alias("splitname"),
        "age",
    ]
)
print(q.collect())

# Convert list of strings into multiple columns representing first and last name
q = (
    pl.scan_csv(names_csv_file)
    .select(
        [
            "name",
            pl.col("name").str.split(" ").alias("split_name"),
            "age",
        ]
    )
    .with_columns(
        pl.struct(
            [
                pl.col("split_name")
                .arr.get(i)
                .alias("first_name" if i == 0 else "last_name")
                for i in range(2)
            ]
        ).alias("split_name")
    )
    # This splits the struct into multiple columns
    .unnest("split_name")
)
print(q.collect())

# Names in the titanic dataset come in the form of:
# `Last Name, Title. First Name`
# We will use a regex to split the related values but split does not allow for
# regex values so we will perform the split multiple times
q = (
    pl.scan_csv("./scripts/titanic_train.csv")
    .select([pl.col("Name").str.split(r", ").alias("split_name")])
    .with_column(
        [
            pl.struct(
                [
                    pl.col("split_name")
                    .arr.get(i)
                    .alias("Last Name" if i == 0 else "First Name")
                    for i in range(2)
                ]
            ).alias("split_name")
        ]
    )
    .unnest("split_name")
    # Second split for title
    .select(
        [
            # All columns except first name
            pl.exclude("First Name"),
            pl.col("First Name").str.split(r". ").alias("split_name"),
        ]
    )
    .with_column(
        pl.struct(
            [
                pl.col("split_name")
                .arr.get(i)
                .alias("Title" if i == 0 else "First Name")
                for i in range(2)
            ]
        ).alias("split_name")
    )
    .unnest("split_name")
)
print(q.collect())

# Replacing values in a dataframe
q = pl.scan_csv("./scripts/titanic_train.csv").select(
    [pl.col("Name").str.replace("Mlle.", "Miss.")]
)
print(q.collect())

# Replacing with regex
q = pl.scan_csv("./scripts/titanic_train.csv").select(
    [pl.col("Name").str.replace("Mlle.|Ms.|Mme.|Mrs.", "Miss.")]
)
print(q.collect())
