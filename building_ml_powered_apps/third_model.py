# Here is a third model that trims down the number of features and only uses
# generated features VS the bag of words model that explodes the feature size

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from building_ml_powered_apps.creating_a_model import (
    format_raw_df,
    get_feature_importance,
    get_metrics,
    get_split_by_author,
)
from building_ml_powered_apps.second_model import (
    FEATURE_ARR,
    add_v2_text_features,
    get_question_score_from_input,
)
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

tqdm.pandas()

FEATURE_DISPLAY_NAMES = {
    "num_questions": "frequency of question marks",
    "num_periods": "frequency of periods",
    "num_commas": "frequency of commas",
    "num_exclam": "frequency of exclamation points",
    "num_quotes": "frequency of quotes",
    "num_colon": "frequency of colons",
    "num_semicolon": "frequency of semicolons",
    "num_stops": "frequency of stop words",
    "num_words": "word count",
    "num_chars": "word count",
    "num_diff_words": "vocabulary diversity",
    "avg_word_len": "vocabulary complexity",
    "polarity": "positivity of emotional sentiment",
    "ADJ": "frequency of adjectives",
    "ADP": "frequency of adpositions",
    "ADV": "frequency of adverbs",
    "AUX": "frequency of auxiliary verbs",
    "CONJ": "frequency of coordinating conjunctions",
    "DET": "frequency of determiners",
    "INTJ": "frequency of interjections",
    "NOUN": "frequency of nouns",
    "NUM": "frequency of numerals",
    "PART": "frequency of particles",
    "PRON": "frequency of pronouns",
    "PROPN": "frequency of proper nouns",
    "PUNCT": "frequency of punctuation",
    "SCONJ": "frequency of subordinating conjunctions",
    "SYM": "frequency of symbols",
    "VERB": "frequency of verbs",
    "X": "frequency of other words",
}


# Prepare LIME explainer using training data
def get_explainer() -> LimeTabularExplainer:
    df = pd.read_csv(Path("./ml-powered-applications/data/writers_with_features.csv"))
    train_df, test_df = get_split_by_author(df, test_size=0.2, random_state=40)
    explainer = LimeTabularExplainer(
        train_df[FEATURE_ARR].values,
        feature_names=FEATURE_ARR,
        class_names=["low", "high"],
    )
    return explainer


# Simplify signs to make display clearer for users
def _simplify_order_sign(order_sign: str) -> str:
    if order_sign in ["<=", "<"]:
        return "<"
    if order_sign in [">=", ">"]:
        return ">"
    return order_sign


# Generate a recommendation string from an operator and the
# type of impact
def _get_recommended_modification(simple_order: str, impact: float) -> str:
    bigger_than_threshold = simple_order == ">"
    has_positive_impact = impact > 0

    if bigger_than_threshold and has_positive_impact:
        return "No need to decrease"
    if not bigger_than_threshold and not has_positive_impact:
        return "Increase"
    if bigger_than_threshold and not has_positive_impact:
        return "Decrease"
    if not bigger_than_threshold and has_positive_impact:
        return "No need to increase"
    # TODO: What if we are none of these...
    return "WHAT HAPPENED"


# Parse explanations returned by LIME into a user readable format
def parse_explanations(explanations: Tuple[str, float]) -> List[Dict[str, Any]]:
    parsed_exps = []
    for feat_bound, impact in explanations:
        conditions = feat_bound.split(" ")
        # Ignore doubly bound conditions: 1 <= x <= 2 since they will be
        # harder to form as a recommendation
        if len(conditions) == 3:
            feat_name, order, threshold = conditions

            simple_order = _simplify_order_sign(order)
            recommendation = _get_recommended_modification(simple_order, impact)

            parsed_exps.append(
                {
                    "feature": feat_name,
                    "feature_display_name": FEATURE_DISPLAY_NAMES[feat_name],
                    "order": simple_order,
                    "threshold": threshold,
                    "impact": impact,
                    "recommendation": recommendation,
                }
            )
    return parsed_exps


# Generate recommendation text we can display on a flask app
def get_recommendation_string_from_parsed_exps(exp_list: List[Dict[str, Any]]) -> str:
    recommendations = []
    for i, feature_exp in enumerate(exp_list):
        recommendation = (
            f"{feature_exp['recommendation']} {feature_exp['feature_display_name']}"
        )
        font_color = "green"
        if feature_exp["recommendation"] in ["Increase", "Decrease"]:
            font_color = "red"
        rec_str = f"""<font color="{font_color}">{i + 1}) {recommendation}"""
        recommendations.append(rec_str)
    rec_string = "<br/>".join(recommendations)
    return rec_string


# Generated features for an input array of text
def get_features_from_text_array(input_array: List[str]) -> pd.DataFrame:
    text_ser = pd.DataFrame(input_array, columns=["full_text"])
    text_ser = add_v2_text_features(text_ser.copy())
    features = text_ser[FEATURE_ARR].astype(float)
    return features


# Generate features for a unique text input and return a 1 row series
# with v3 model features
def get_features_from_input_text(text_input: str) -> pd.Series:
    arr_features = get_features_from_text_array([text_input])
    return arr_features.iloc[0]


# Get a score and recommenations that can be displayed in a Flask app
def get_recommendation_and_prediction_from_text(
    model, input_text: str, num_feats: int = 10
):
    feats = get_features_from_input_text(input_text)
    pos_score = model.predict_proba([feats])[0][1]
    print("explaining")
    exp = get_explainer().explain_instance(
        feats, model.predict_proba, num_features=num_feats, labels=(1,)
    )
    print("explaining done")
    parsed_exps = parse_explanations(exp.as_list())
    recs = get_recommendation_string_from_parsed_exps(parsed_exps)
    return f"""
    Current score (0 is worst, 1 is best):
     <br/>
     {pos_score}
    <br/>
    <br/>

    Recommendations (ordered by importance):
    <br/>
    <br/>
    {recs}
    """


def get_v3_model():
    MODEL_PATH = Path("./building_ml_powered_apps/models/model_3.pkl")
    clf = joblib.load(MODEL_PATH)
    return clf


if __name__ == "__main__":
    MODEL_PATH = Path("./building_ml_powered_apps/models/model_3.pkl")
    if not MODEL_PATH.exists():
        np.random.seed(35)
        data_path = Path("./ml-powered-applications/data/writers.csv")
        df = pd.read_csv(data_path)
        df = format_raw_df(df.copy())
        df["full_text"] = df["Title"].str.cat(df["body_text"], sep=" ", na_rep="")
        df = add_v2_text_features(df)

        train_df, test_df = get_split_by_author(df, test_size=0.2, random_state=40)
        print(df[FEATURE_ARR].head() * 100)

        # Generate input and output vectors using feature names
        def get_feature_vector_and_label(
            df: pd.DataFrame, feature_names: List[str]
        ) -> Tuple[np.ndarray, np.ndarray]:
            features = df[feature_names].astype(float)
            labels = df["Score"] > df["Score"].median()
            return features, labels

        X_train, y_train = get_feature_vector_and_label(train_df, FEATURE_ARR)
        X_test, y_test = get_feature_vector_and_label(test_df, FEATURE_ARR)
        print(y_train.value_counts())
        print(X_test.shape)

        # Train the model using sklearn
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

        joblib.dump(clf, MODEL_PATH)
    else:
        clf = joblib.load(MODEL_PATH)

    # Validate features that are useful
    k = 20
    all_feature_names = np.array(FEATURE_ARR)
    feature_importances = get_feature_importance(clf, all_feature_names)
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

    # FIXME: This will also not work since we didn't use vectorization here...
    """
    pos_prob = get_question_score_from_input(
        None,
        clf,
        "When quoting a person's informal speech, how much liberty do you have to make changes to what they say?",
    )
    print(
        f"{pos_prob} probability of the question receiving a high score according to our model"
    )
    """
