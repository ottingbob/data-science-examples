import pandas as pd
import matplotlib.pyplot as plt

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

print(rating_mean_count_df[rating_mean_count_df["mean"] == 5])
print(rating_mean_count_df.sort_values("count", ascending=False).head(100))
print(rating_mean_count_df.sort_values("count", ascending=True).head(100))

# Perform item based collaborative filtering on one movie sample
userid_movie_title_matrix = movie_rating_df.pivot_table(
    index="user_id", columns="title", values="rating"
)
print(userid_movie_title_matrix.head())

titanic = userid_movie_title_matrix["Titanic (1997)"]
print(titanic)

star_wars = userid_movie_title_matrix["Star Wars (1977)"]
star_wars_correlations = pd.DataFrame(
    userid_movie_title_matrix.corrwith(star_wars),
    columns=["Correlation"],
)
star_wars_correlations = star_wars_correlations.join(rating_mean_count_df["count"])
star_wars_correlations.dropna(inplace=True)
star_wars_correlations = star_wars_correlations[
    star_wars_correlations["count"] > 100
].sort_values("Correlation", ascending=False)
print(star_wars_correlations)

titanic_correlations = pd.DataFrame(
    userid_movie_title_matrix.corrwith(titanic), columns=["Correlation"]
)
titanic_correlations = titanic_correlations.join(rating_mean_count_df["count"])
titanic_correlations.dropna(inplace=True)
titanic_correlations = titanic_correlations[
    titanic_correlations["count"] > 80
].sort_values("Correlation", ascending=False)
print(titanic_correlations)

# Create an item-based collaborative filter on the entire dataset
movie_correlations = userid_movie_title_matrix.corr(method="pearson", min_periods=80)
print(movie_correlations)

my_ratings = pd.DataFrame(
    [["Liar Liar (1997)", 5], ["Star Wars (1977)", 1]],
    columns=["Movie Name", "Ratings"],
)

similar_movies_list = pd.Series()
for i in range(my_ratings.shape[1]):
    similar_movie = movie_correlations[my_ratings["Movie Name"][i]].dropna()
    similar_movie = similar_movie.map(lambda x: x * my_ratings["Ratings"][i])
    similar_movies_list = similar_movies_list.append(similar_movie)

similar_movies_list.sort_values(inplace=True, ascending=False)
print(similar_movies_list.head(10))
