import pandas as pd
import matplotlib.pyplot as plt

# import numpy as np

# Built from this example
# https://github.com/rsanimesh/udemy-complete-data-science-bootcamp/blob/master/Movie%20Recommender%20System-Filming.ipynb
#
# With these datasources
# https://github.com/99x/scikit-recommender-api/blob/master/db/Movie_Id_Titles

# Import the datasets
movie_titles_df = pd.read_csv("./movie_id_titles.csv")
movie_rating_df = pd.read_csv(
    "./movie_ratings.tsv", sep="\t", names=["user_id", "item_id", "rating", "timestamp"]
)

movie_rating_df.drop(["timestamp"], axis="columns", inplace=True)
movie_rating_df = pd.merge(movie_rating_df, movie_titles_df, on="item_id")

# Visualize the dataset
print(movie_rating_df.groupby("title")["rating"].describe())

movie_mean_ratings = movie_rating_df.groupby("title")["rating"].describe()["mean"]
print(movie_mean_ratings.head())

movie_rating_counts = movie_rating_df.groupby("title")["rating"].describe()["count"]
print(movie_rating_counts.head())

rating_mean_count_df = pd.concat(
    [movie_mean_ratings, movie_rating_counts], axis="columns"
)
rating_mean_count_df.reset_index()
print(rating_mean_count_df.head())

# rating_mean_count_df["mean"].plot(bins=100, kind="hist", color="r")
# rating_mean_count_df["count"].plot(bins=100, kind="hist", color="r")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

# Plot the first histogram on the first subplot (ax1)
ax1.hist(rating_mean_count_df["mean"], bins=100, alpha=0.5, color="blue")
ax1.set_xlabel("Movie Rating")
ax1.set_ylabel("Frequency of rating")
ax1.set_title("Moving Rating Means")

# Plot the second histogram on the second subplot (ax2)
ax2.hist(rating_mean_count_df["count"], bins=100, alpha=0.5, color="red")
ax2.set_xlabel("Rating Count")
ax2.set_ylabel("Movies with Rating Count")
ax2.set_title("Moving Rating Counts")

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
