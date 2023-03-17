from pathlib import Path
from typing import Any, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, vstack
from sklearn.ensemble import RandomForestClassifier

# TODO: NEED TO LEARN HOW TO JUMP TO DEFINITIONS USING LSP...
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit


def format_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    # Fix types and set index
    df["AnswerCount"].fillna(-1, inplace=True)
    df["AnswerCount"] = df["AnswerCount"].astype(int)
    df["Id"] = df["Id"].astype(int)
    df["OwnerUserId"].fillna(-1, inplace=True)
    df["OwnerUserId"] = df["OwnerUserId"].astype(int)
    df["PostTypeId"] = df["PostTypeId"].astype(int)
    df.set_index("Id", inplace=True, drop=False)

    # Add measure of the length of a post
    df["full_text"] = df["Title"].str.cat(df["body_text"], sep=" ", na_rep="")
    df["text_len"] = df["full_text"].str.len()

    # Label a question as a post id of 1
    df["is_question"] = df["PostTypeId"] == 1

    # Link questions and answers
    df = df.join(
        df[["Id", "Title", "body_text", "Score", "AcceptedAnswerId"]],
        on="ParentId",
        how="left",
        rsuffix="_question",
    )
    return df


def add_features_to_df(df: pd.DataFrame) -> pd.DataFrame:
    if "Title" in df.columns:
        df["full_text"] = df["Title"].str.cat(df["body_text"], sep=" ", na_rep="")

    df["action_verb_full"] = (
        df["full_text"].str.contains("can", regex=False)
        | df["full_text"].str.contains("What", regex=False)
        | df["full_text"].str.contains("should", regex=False)
    )
    df["language_question"] = (
        df["full_text"].str.contains("punctuate", regex=False)
        | df["full_text"].str.contains("capitalize", regex=False)
        | df["full_text"].str.contains("abbreviate", regex=False)
    )
    df["question_mark_full"] = df["full_text"].str.contains("?", regex=False)
    df["text_len"] = df["full_text"].str.len()
    return df


def get_split_by_author(
    posts: pd.DataFrame,
    author_id_column: str = "OwnerUserId",
    test_size: float = 0.3,
    random_state: int = 40,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    splits = splitter.split(posts, groups=posts[author_id_column])
    train_idx, test_idx = next(splits)
    return posts.iloc[train_idx, :], posts.iloc[test_idx, :]


def train_vectorizer(df: pd.DataFrame) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        strip_accents="ascii", min_df=5, max_df=0.5, max_features=10000
    )

    vectorizer.fit(df["full_text"].copy())
    return vectorizer


def get_vectorized_series(
    text_series: pd.Series, vectorizer: TfidfVectorizer
) -> pd.Series:
    vectors = vectorizer.transform(text_series)
    vectorized_series = [vectors[i] for i in range(vectors.shape[0])]
    return vectorized_series


# TODO: What does `hstack` return && fill it in as the first arg in the tuple below
def get_feature_vector_and_label(
    df: pd.DataFrame, feature_names: List[str]
) -> Tuple[Any, pd.DataFrame]:
    vec_features = vstack(df["vectors"])
    num_features = df[feature_names].astype(float)
    features = hstack([vec_features, num_features])
    labels = df["Score"] > df["Score"].median()
    return features, labels


def get_metrics(
    y_test: pd.DataFrame, y_predicted: pd.DataFrame
) -> Tuple[float, float, float, float]:
    # true positives / (true positives + false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None, average="weighted")
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None, average="weighted")
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average="weighted")
    # true positives + true negatives / total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1


def get_model_probabilities_for_input_texts(
    text_array: List[str],
    feature_list: List[str],
    model: RandomForestClassifier,
    vectorizer: TfidfVectorizer,
):
    vectors = vectorizer.transform(text_array)
    text_ser = pd.DataFrame(text_array, columns=["full_text"])
    text_ser = add_features_to_df(text_ser)
    vec_features = vstack(vectors)
    num_features = text_ser[feature_list].astype(float)
    features = hstack([vec_features, num_features])
    return model.predict_proba(features)


df = pd.read_csv(Path("./ml-powered-applications/data/writers.csv"))
df = format_raw_df(df.copy())
# Only use the questions ??
df = df.loc[df["is_question"]].copy()

# Add features, vectorize, and create train / test split
df = add_features_to_df(df.copy())
train_df, test_df = get_split_by_author(df, test_size=0.2, random_state=40)
vectorizer = train_vectorizer(train_df)
train_df["vectors"] = get_vectorized_series(train_df["full_text"].copy(), vectorizer)
test_df["vectors"] = get_vectorized_series(test_df["full_text"].copy(), vectorizer)

# Define features and get related features from test and train dataframes
features = ["action_verb_full", "question_mark_full", "text_len", "language_question"]
x_train, y_train = get_feature_vector_and_label(train_df, features)
x_test, y_test = get_feature_vector_and_label(test_df, features)

# Train the model
clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", oob_score=True)
clf.fit(x_train, y_train)
y_predicted = clf.predict(x_test)
y_predicted_probs = clf.predict_proba(x_test)

print(y_train.value_counts())

# Now that the model is trained evaluate the results starting with aggregate metrics

# Training accuracy:
# Thanks to https://datascience.stackexchange.com/questions/13151/randomforestclassifier-oob-scoring-method
y_train_pred = np.argmax(clf.oob_decision_function_, axis=1)
accuracy, precision, recall, f1 = get_metrics(y_train, y_train_pred)
print(
    f"Training accuracy = {accuracy:.3f}, precision = {precision:.3f}, recall = {recall:.3f}, f1 = {f1:.3f}"
)

# Validation accuracy:
accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted)
print(
    f"Validation accuracy = {accuracy:.3f}, precision = {precision:.3f}, recall = {recall:.3f}, f1 = {f1:.3f}"
)

# Save the trained model and vectorizer to disk for future use:
model_path = Path("./building-ml-powered-apps/models/model_1.pkl")
vectorizer_path = Path("./building-ml-powered-apps/models/vectorizer_1.pkl")
joblib.dump(clf, model_path)
joblib.dump(vectorizer, vectorizer_path)

# To use it on unseen data, define and use an inference function against our trained model.
# Check the probability of an arbitrary question receiving a high score according to our model
# Inference function expects an array of questions so make an array of length 1
test_q = ["bad question"]
probs = get_model_probabilities_for_input_texts(test_q, features, clf, vectorizer)
# Index 1 corresponds to the positive class here
print(
    f"{probs[0][1]} probability of the question receiving a high score according to our model"
)
