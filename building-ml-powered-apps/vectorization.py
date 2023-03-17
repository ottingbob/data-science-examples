from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from matplotlib.patches import Rectangle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the data from the example file
df = pd.read_csv(Path("./ml-powered-applications/data/writers.csv"))

# Format data with changing types to make processing easier
df["AnswerCount"] = df["AnswerCount"].fillna(-1)
df["AnswerCount"] = df["AnswerCount"].astype(int)
df["PostTypeId"] = df["PostTypeId"].astype(int)
df["Id"] = df["Id"].astype(int)
df.set_index("Id", inplace=True, drop=False)
# Add measure of the length of a post
df["full_text"] = df["Title"].str.cat(df["body_text"], sep=" ", na_rep="")
df["text_len"] = df["full_text"].str.len()
# Label a question as a post id of 1
df["is_question"] = df["PostTypeId"] == 1

# See some examples
print(df)

questions_with_accepted_answers = df[
    df["is_question"] & ~(df["AcceptedAnswerId"].isna())
]
q_and_a = questions_with_accepted_answers.join(
    df[["body_text"]], on="AcceptedAnswerId", how="left", rsuffix="_answer"
)

# Display all the data in the DataFrame
pd.options.display.max_colwidth = 500
print(q_and_a[["body_text", "body_text_answer"]][:3])

has_accepted_answer = df[df["is_question"] & ~(df["AcceptedAnswerId"].isna())]
received_answers = df[df["is_question"] & (df["AnswerCount"] != 0)]
no_answers = df[
    df["is_question"] & (df["AcceptedAnswerId"].isna()) & (df["AnswerCount"] == 0)
]
print(
    f"{len(df[df['is_question']])} total questions\n"
    + f"{len(received_answers)} received at least one answer\n"
    + f"{len(has_accepted_answer)} received an accepted answer"
)

# Now plot some summary statistics
high_score = df["Score"] > df["Score"].median()
# Filter out really long questions
normal_length = df["text_len"] < 2000
high_score_questions = df[df["is_question"] & high_score & normal_length]["text_len"]
question_plot = high_score_questions.hist(
    bins=60,
    density=True,
    histtype="step",
    color="orange",
    linewidth=3,
    grid=False,
    figsize=(8, 5),
)

non_high_score_questions = df[df["is_question"] & ~high_score & normal_length][
    "text_len"
]
non_high_score_questions.hist(
    bins=60,
    density=True,
    histtype="step",
    color="purple",
    linewidth=3,
    grid=False,
    figsize=(8, 5),
)

handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in ["orange", "purple"]]
labels = ["High score", "Low score"]
plt.legend(handles, labels)
question_plot.set_xlabel("Sentence length (characters)")
question_plot.set_ylabel("Percentage of sentences")
# plt.show()


def get_normalized_series(df: pd.DataFrame, col: str) -> pd.Series:
    return (df[col] - df[col].mean()) / df[col].std()


tabular_df = df.copy()
tabular_df["NormComment"] = get_normalized_series(tabular_df, "CommentCount")
tabular_df["NormScore"] = get_normalized_series(tabular_df, "Score")

# Extract relevant information from the date
tabular_df["date"] = pd.to_datetime(tabular_df["CreationDate"])
# Extract meaningful features from the datetime object
tabular_df["year"] = tabular_df["date"].dt.year
tabular_df["month"] = tabular_df["date"].dt.month
tabular_df["day"] = tabular_df["date"].dt.day
tabular_df["hour"] = tabular_df["date"].dt.hour

# Only get top tags that have a high count (> 500)
tags = tabular_df["Tags"]
clean_tags = tags.str.split("><").apply(
    lambda x: [a.strip("<").strip(">") for a in x] if type(x) == list else x
)
# get dummy values and select tags that appear over 500 times
tag_columns = pd.get_dummies(clean_tags.apply(pd.Series).stack()).sum(level=0)
all_tags = tag_columns.astype(bool).sum(axis=0).sort_values(ascending=False)
top_tags = all_tags[all_tags > 500]
top_tag_columns = tag_columns[top_tags.index]

# Add tags back into original DataFrame
final = pd.concat([tabular_df, top_tag_columns], axis=1)
# Keep only the vectorized features
cols_to_keep = ["year", "month", "day", "hour", "NormComment", "NormScore"] + list(
    top_tags.index
)
final_features = final[cols_to_keep]
print(final_features)

# TODO: Work on the vectorizer next...
# Term Frequency -- Inverse document frequency
vectorizer = TfidfVectorizer(
    strip_accents="ascii", min_df=5, max_df=0.5, max_features=10_000
)
bag_of_words = vectorizer.fit_transform(df[df["is_question"]]["full_text"])
print(bag_of_words)

# Load a large model and disable pipeline unnecessary parts for our task to speed
# up the vectorization process
# FIXME: Double check that this model name is correct...
#   OR check if the model needs to be downloaded prior to using it...
nlp = spacy.load("en_core_web_lg", disable=["parser", "tagger", "ner", "textcat"])

# Get the vector for each of the questions. The default vectorizer returned is the
# average of all vectors in the sentence
spacy_emb = df[df["is_question"]]["full_text"].apply(lambda x: nlp(x).vector)
embeddings = np.vstack(spacy_emb)
print(embeddings)
