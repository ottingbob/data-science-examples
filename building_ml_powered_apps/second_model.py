# Here is a second model using features in order to address the first model's shortcomings

from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import spacy
from building_ml_powered_apps.creating_a_model import (
    format_raw_df,
    get_calibration_plot,
    get_confusion_matrix_plot,
    get_feature_importance,
    get_feature_vector_and_label,
    get_metrics,
    get_roc_plot,
    get_split_by_author,
    get_vectorized_series,
    train_vectorizer,
)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.sparse import hstack, vstack
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

POS_NAMES = {
    "ADJ": "adjective",
    "ADP": "adposition",
    "ADV": "adverb",
    "AUX": "auxiliary verb",
    "CONJ": "coordinating conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other",
}

FEATURE_ARR = [
    "num_questions",
    "num_periods",
    "num_commas",
    "num_exclam",
    "num_quotes",
    "num_colon",
    "num_stops",
    "num_semicolon",
    "num_words",
    "num_chars",
    "num_diff_words",
    "avg_word_len",
    "polarity",
]
FEATURE_ARR.extend(POS_NAMES.keys())

SPACY_MODEL = spacy.load("en_core_web_sm")
tqdm.pandas()


# Add counts of punctuation chars to a DataFrame
def add_char_count_features(df: pd.DataFrame) -> pd.DataFrame:
    df["num_chars"] = df["full_text"].str.len()

    def num_specific_char(char: str) -> pd.Series:
        return 100 * df["full_text"].str.count(char) / df["num_chars"]

    df["num_questions"] = num_specific_char("\?")
    df["num_periods"] = num_specific_char("\.")
    df["num_commas"] = num_specific_char(",")
    df["num_exclam"] = num_specific_char("!")
    df["num_quotes"] = num_specific_char('"')
    df["num_colon"] = num_specific_char(":")
    df["num_semicolon"] = num_specific_char(";")
    return df


# Average word length for a list of words
def _get_avg_word_len(tokens: List[str]) -> float:
    if len(tokens) < 1:
        return 0
    lens = [len(x) for x in tokens]
    return float(sum(lens) / len(lens))


# Count occurences of each part of speech, and add it to an input DataFrame
def count_each_pos(df: pd.DataFrame) -> pd.DataFrame:
    pos_list = df["spacy_text"].apply(lambda doc: [token.pos_ for token in doc])
    for pos_name in POS_NAMES.keys():
        df[pos_name] = (
            pos_list.apply(lambda x: len([match for match in x if match == pos_name]))
            / df["num_chars"]
        )
    return df


# Add statistical features such as word counts to a DataFrame
def get_word_stats(df: pd.DataFrame) -> pd.DataFrame:
    df["spacy_text"] = df["full_text"].progress_apply(lambda x: SPACY_MODEL(x))

    df["num_words"] = df["spacy_text"].apply(lambda x: 100 * len(x)) / df["num_chars"]
    df["num_diff_words"] = df["spacy_text"].apply(lambda x: len(set(x)))
    df["avg_word_len"] = df["spacy_text"].apply(lambda x: _get_avg_word_len(df))
    df["num_stops"] = (
        df["spacy_text"].apply(
            # What is `stop.is_stop` lol...
            lambda x: 100
            * len([stop for stop in x if stop.is_stop])
        )
        / df["num_chars"]
    )

    df = count_each_pos(df.copy())
    return df


def get_sentiment_score(df: pd.DataFrame) -> pd.DataFrame:
    sid = SentimentIntensityAnalyzer()
    df["polarity"] = df["full_text"].progress_apply(
        lambda x: sid.polarity_scores(x)["pos"]
    )
    return df


# Add multiple features used by v2 model DataFrame
def add_v2_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_char_count_features(df)
    df = get_word_stats(df)
    df = get_sentiment_score(df)
    return df


# Return an array of probability scores representing the likelihood of a question
# receiving a high score format:
# [
#   [prob_low_score1, prob_high_score_1], ...
# ]
def get_model_probabilities_for_input_texts(
    vectorizer, model, text_array: List[str]
) -> np.ndarray:
    if not vectorizer:
        vectorizer = joblib.load("./building_ml_powered_apps/models/vectorizer_2.pkl")
    vectors = vectorizer.transform(text_array)
    text_ser = pd.DataFrame(text_array, columns=["full_text"])
    text_ser = add_v2_text_features(text_ser)
    vec_features = vstack(vectors)
    num_features = text_ser[FEATURE_ARR].astype(float)
    features = hstack([vec_features, num_features])

    return model.predict_proba(features)


# Get the probability for the positive class for one example question
def get_question_score_from_input(vectorizer, model, question: str) -> pd.DataFrame:
    preds = get_model_probabilities_for_input_texts(vectorizer, model, [question])
    positive_proba = preds[0][1]
    return positive_proba


if __name__ == "__main__":
    np.random.seed(35)
    # Check if model exists
    MODEL_PATH = Path("./building_ml_powered_apps/models/model_2.pkl")
    VECTORIZER_PATH = Path("./building_ml_powered_apps/models/vectorizer_2.pkl")
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        data_path = Path("./ml-powered-applications/data/writers.csv")
        df = pd.read_csv(data_path)
        df = format_raw_df(df.copy())
        df = df.loc[df["is_question"]].copy()

        df["full_text"] = df["Title"].str.cat(df["body_text"], sep=" ", na_rep="")
        train_df, test_df = get_split_by_author(df, test_size=0.2, random_state=40)
        vectorizer = train_vectorizer(train_df)
        df["vectors"] = get_vectorized_series(df["full_text"].copy(), vectorizer)
        df = add_char_count_features(df)
        df = get_word_stats(df)
        df = get_sentiment_score(df)
        print(df)

        # Now train a new model with the new features
        train_df, test_df = get_split_by_author(df, test_size=0.2, random_state=40)

        X_train, y_train = get_feature_vector_and_label(train_df, FEATURE_ARR)
        X_test, y_test = get_feature_vector_and_label(test_df, FEATURE_ARR)
        print(y_train.value_counts())
        print(X_test.shape)

        clf = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", oob_score=True
        )
        clf.fit(X_train, y_train)

        y_predicted = clf.predict(X_test)
        y_predicted_proba = clf.predict_proba(X_test)
        y_train_pred = np.argmax(clf.oob_decision_function_, axis=1)
        accuracy, precision, recall, f1 = get_metrics(y_train, y_train_pred)
        print(
            f"Training accuracy = {accuracy:.3f}, precision = {precision:.3f}, recall = {recall:.3f}, f1 = {f1:.3f}"
        )

        accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted)
        print(
            f"Validation accuracy = {accuracy:.3f}, precision = {precision:.3f}, recall = {recall:.3f}, f1 = {f1:.3f}"
        )

        model_path = Path("./building_ml_powered_apps/models/model_2.pkl")
        vectorizer_path = Path("./building_ml_powered_apps/models/vectorizer_2.pkl")
        joblib.dump(clf, model_path)
        joblib.dump(vectorizer, vectorizer_path)
    else:
        clf = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)

    # Now evaluate the model performance
    k = 20
    w_indices: np.ndarray = vectorizer.get_feature_names_out()
    w_indices = np.append(w_indices, FEATURE_ARR)
    feature_importances = get_feature_importance(clf, w_indices)
    print(f"Top {k} importances:")
    print(
        "\n".join(
            [
                # `g` formatting keeps the trailing 0's
                f"{tup[0]}: {tup[1]:.2g}"
                for tup in feature_importances[:k]
            ]
        )
    )
    print(f"Bottom {k} importances:")
    print(
        "\n".join(
            [
                # `g` formatting keeps the trailing 0's
                f"{tup[0]}: {tup[1]:.2g}"
                for tup in feature_importances[-k:]
            ]
        )
    )

    # FIXME: We actually need the dataset with the updated features if we want
    # to be able to graph these...
    # And some plots
    # get_roc_plot(y_predicted_proba[:, 1], y_test, figsize=(10, 10))
    # get_confusion_matrix_plot(y_predicted, y_test, figsize=(9, 9))
    # get_calibration_plot(y_predicted_proba[:, 1], y_test, figsize=(9, 9))

    pos_prob = get_question_score_from_input(
        vectorizer,
        clf,
        "When quoting a person's informal speech, how much liberty do you have to make changes to what they say?",
    )
    print(
        f"{pos_prob} probability of the question receiving a high score according to our model"
    )
