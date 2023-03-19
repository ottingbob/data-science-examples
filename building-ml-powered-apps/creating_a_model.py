import itertools
from pathlib import Path
from typing import Any, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack, vstack
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier

# TODO: NEED TO LEARN HOW TO JUMP TO DEFINITIONS USING LSP...
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
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


def get_confusion_matrix_plot(
    predicted_y: pd.DataFrame,
    true_y: pd.DataFrame,
    classes: List[str] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    cmap=plt.get_cmap("binary"),
    figsize: Tuple[int, int] = (10, 10),
):
    # Inspired by sklearn example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    if not classes:
        classes = ["Low quality", "High quality"]

    cm = confusion_matrix(true_y, predicted_y)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=figsize)
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)

    title_obj = plt.title(title, fontsize=30)
    title_obj.set_position([0.5, 1.15])

    plt.colorbar(im)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = ".2f" if normalize else "d"
    threshold = (cm.max() - cm.min()) / 2.0 + cm.min()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > threshold else "black",
        )
    plt.tight_layout()
    plt.ylabel("True label", fontsize=20)
    plt.xlabel("Predicted label", fontsize=20)


def get_roc_plot(
    predicted_proba_y,
    true_y: pd.DataFrame,
    tpr_bar: int = -1,
    fpr_bar: int = -1,
    figsize: Tuple[int, int] = (10, 10),
):
    # Inspired by sklearn example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    fpr, tpr, thresholds = roc_curve(true_y, predicted_proba_y)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=figsize)
    plt.plot(
        fpr, tpr, lw=1, alpha=1, color="black", label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot(
        [0, 1], [0, 1], linestyle="--", lw=2, color="grey", label="Chance", alpha=1
    )

    # Cheating on position to make plot more readable
    plt.plot(
        [0.01, 0.01, 1],
        [0.01, 0.99, 0.99],
        linestyle=":",
        lw=2,
        color="green",
        label="Perfect model",
        alpha=1,
    )

    if tpr_bar != -1:
        plt.plot(
            [0, 1],
            [tpr_bar, tpr_bar],
            linestyle="-",
            lw=2,
            color="red",
            label="TPR requirement",
            alpha=1,
        )
        plt.fill_between([0, 1], [tpr_bar, tpr_bar], [1, 1], alpha=0, hatch="\\")

    if fpr_bar != -1:
        plt.plot(
            [fpr_bar, fpr_bar],
            [0, 1],
            linestyle="-",
            lw=2,
            color="red",
            label="FPR requirement",
            alpha=1,
        )
        plt.fill_between([fpr_bar, 1], [1, 1], alpha=0, hatch="\\")

    plt.legend(loc="lower right")
    plt.ylabel("True positive rate", fontsize=20)
    plt.xlabel("False positive rate", fontsize=20)
    plt.xlim(0, 1)
    plt.ylim(0, 1)


def get_calibration_plot(predicted_proba_y, true_y: pd.DataFrame, figsize=(10, 10)):
    # Inspired by sklearn example
    # https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
    plt.figure(figsize=figsize)
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    clf_score = brier_score_loss(true_y, predicted_proba_y, pos_label=true_y.max())
    print(f"\tBrier: {clf_score:1.3f}")

    fraction_of_positives, mean_predicted_value = calibration_curve(
        true_y, predicted_proba_y, n_bins=10
    )

    ax1.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        color="black",
        label=f"{clf_score:1.3f} Brier score (0 is best, 1 is worst)",
    )
    ax2.hist(
        predicted_proba_y, range=(0, 1), bins=10, histtype="step", lw=2, color="black"
    )

    ax1.set_ylabel("Fraction of positives")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.legend(loc="lower right")
    ax1.set_title("Calibration plot")

    ax2.set_title("Probability distribution")
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()


def get_top_k(
    df: pd.DataFrame,
    # column name of the predicted probabilies
    proba_col: str,
    # column name of the true labels
    true_label_col: str,
    # number of examples for each category
    k: int = 5,
    # classifier decision boundary to classify as positive
    decision_threshold: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # For binary classification problems return the k most correct / incorrect /
    # unsure examples for each class
    #
    # Get correct and incorrect predictions
    correct = df[(df[proba_col] > decision_threshold) == df[true_label_col]].copy()
    incorrect = df[(df[proba_col] > decision_threshold) != df[true_label_col]].copy()

    # `nlargest` will order in descending order
    # `nsmallest` will order in ascending order
    top_correct_positive = correct[correct[true_label_col]].nlargest(k, proba_col)
    top_correct_negative = correct[~correct[true_label_col]].nsmallest(k, proba_col)

    top_incorrect_positive = incorrect[incorrect[true_label_col]].nsmallest(
        k, proba_col
    )
    top_incorrect_negative = incorrect[~incorrect[true_label_col]].nlargest(
        k, proba_col
    )

    # Get closest examples to decision threshold
    most_uncertain = df.iloc[(df[proba_col] - decision_threshold).abs().argsort()[:k]]

    return (
        top_correct_positive,
        top_correct_negative,
        top_incorrect_positive,
        top_incorrect_negative,
        most_uncertain,
    )


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

# Now explore the data:
# We start by investigating a confusion matrix which helps us compare a models predictions
# with the true classes for each class
get_confusion_matrix_plot(y_predicted, y_test)

# A ROC (Receiver operating characteristic) plots true positive rate as a function of
# the false positive rate
# This uses a decision threshold to determine whether an example is of a certain class
# if the probability reported by the model is above a given amount
# get_roc_plot(y_predicted_probs[:, 1], y_test)

# Plot with a specific false positive rate in mind:
get_roc_plot(y_predicted_probs[:, 1], y_test, fpr_bar=0.1, figsize=(10, 10))
# For the given FPR of `0.1` our model has around a `0.2` true positive rate. In an application
# where maintaining this GPR constraint is important, we will continue to track the metric in
# following changes / experiments

# The final test we look at is a calibration curve. This plots a fraction of actual positive
# examples as a function of a models probability score. The curve measures the quality of a
# models probability estimate (when a model says the prob is X% is that really the case??)
get_calibration_plot(y_predicted_probs[:, 1], y_test, figsize=(9, 9))

# plt.show()

test_analysis_df = test_df.copy()
test_analysis_df["predicted_proba"] = y_predicted_probs[:, 1]
test_analysis_df["true_label"] = y_test
threshold = 0.5

top_pos, top_neg, worst_pos, worst_neg, unsure = get_top_k(
    test_analysis_df, "predicted_proba", "true_label", k=2
)
pd.options.display.max_colwidth = 500

to_display = [
    "predicted_proba",
    "true_label",
    "text_len",
    "Title",
    "body_text",
    "action_verb_full",
    "question_mark_full",
    "language_question",
]

# Most confident correct positive predictions
print(top_pos[to_display])
# Most confident correct negative predictions
print(top_neg[to_display])

# Most confident incorrect negative predictions
print(worst_pos[to_display])
# Most confident incorrect positive predictions
print(worst_neg[to_display])

# Unsure cases, where the model porbability is closes to equal for all classes
# In this case with 2 classes it would just be `0.5`
print(unsure[to_display])
