import os
from functools import cached_property
from pathlib import Path
from unittest import TestCase

import joblib
import pandas as pd
from building_ml_powered_apps.creating_a_model import (
    add_features_to_df,
    format_raw_df,
    get_feature_vector_and_label,
    get_model_predictions_for_input_texts,
    get_vectorized_series,
)
from building_ml_powered_apps.tests.helpers import parse_xml_to_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_PATH = Path("../models/model_1.pkl")
VECTORIZER_PATH = Path("../models/vectorizer_1.pkl")


class ModelTests(TestCase):
    CURR_PATH = Path(os.path.dirname(__file__))

    FEATURE_NAMES = [
        "action_verb_full",
        "question_mark_full",
        "text_len",
        "language_question",
    ]

    @cached_property
    def df_with_features(self) -> pd.DataFrame:
        df = parse_xml_to_csv(self.CURR_PATH)
        df = format_raw_df(df.copy())
        return add_features_to_df(df)

    @cached_property
    def trained_v1_model(self) -> RandomForestClassifier:
        full_model_path = Path(self.CURR_PATH / MODEL_PATH)
        return joblib.load(full_model_path)

    @cached_property
    def trained_v1_vectorizer(self) -> TfidfVectorizer:
        full_vectorizer_path = Path(self.CURR_PATH / VECTORIZER_PATH)
        return joblib.load(full_vectorizer_path)

    @cached_property
    def vectorized_df(self) -> pd.DataFrame:
        df = self.df_with_features
        vectorizer = self.trained_v1_vectorizer
        df["vectors"] = get_vectorized_series(df["full_text"].copy(), vectorizer)
        return df

    def test_model_prediction_dimensions(self):
        df = self.vectorized_df
        features, labels = get_feature_vector_and_label(df, self.FEATURE_NAMES)
        probas = self.trained_v1_model.predict_proba(features)

        # The model makes one prediction per input sample
        assert probas.shape[0] == features.shape[0]
        # The model predicts probabilities for two classes
        assert probas.shape[1] == 2

    def test_model_proba_values(self):
        df = self.vectorized_df
        features, _ = get_feature_vector_and_label(df, self.FEATURE_NAMES)
        probas = self.trained_v1_model.predict_proba(features)

        # the model's probabilities are between 0 and 1
        assert (0 <= probas).all() and (probas <= 1).all()

    def test_model_predicts_no_on_bad_question(self):
        input_text = "This isn't even a question. We should score it poorly"
        is_question_good = get_model_predictions_for_input_texts(
            [input_text],
            self.FEATURE_NAMES,
            self.trained_v1_model,
            self.trained_v1_vectorizer,
        )
        assert not is_question_good[0]
