import os
from functools import cached_property
from pathlib import Path
from unittest import TestCase

import pandas as pd
from building_ml_powered_apps.creating_a_model import add_features_to_df, format_raw_df
from building_ml_powered_apps.tests.helpers import parse_xml_to_csv


class DataProcessingTests(TestCase):
    REQUIRED_FEATURES = [
        "is_question",
        "action_verb_full",
        "language_question",
        "question_mark_full",
        "text_len",
    ]

    @cached_property
    def df_with_features(self):
        curr_path = Path(os.path.dirname(__file__))
        df = parse_xml_to_csv(curr_path)
        df = format_raw_df(df.copy())
        return add_features_to_df(df)

    def test_feature_presence(self):
        for feat in self.REQUIRED_FEATURES:
            assert feat in self.df_with_features.columns

    def test_feature_type(self):
        df = self.df_with_features
        expected = {
            "is_question": bool,
            "action_verb_full": bool,
            "language_question": bool,
            "question_mark_full": bool,
            "text_len": int,
        }
        for expected_col, expected_type in expected.items():
            assert df[expected_col].dtype == expected_type

    def test_text_length(self):
        """
        Verify that the text_len col are in the expected range
        """
        df = self.df_with_features
        ntl_col = "text_len"
        assert df[ntl_col].mean() in pd.Interval(left=200, right=1000)
        assert df[ntl_col].max() in pd.Interval(left=0, right=1015)
        assert df[ntl_col].min() in pd.Interval(left=0, right=1000)
