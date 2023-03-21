import math
import os
from functools import cached_property
from pathlib import Path
from unittest import TestCase

import pandas as pd
from building_ml_powered_apps.creating_a_model import (
    add_features_to_df,
    format_raw_df,
    get_random_train_test_split,
    get_split_by_author,
)
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
    def df_with_features(self) -> pd.DataFrame:
        curr_path = Path(os.path.dirname(__file__))
        df = parse_xml_to_csv(curr_path)
        df = format_raw_df(df.copy())
        return add_features_to_df(df)

    def test_random_split_proportion(self):
        df = self.df_with_features
        test_size = 0.3
        train, test = get_random_train_test_split(df, test_size=test_size)
        expected_train_len = math.floor(len(df) * (1 - test_size))
        expected_test_len = math.ceil(len(df) * test_size)
        assert expected_train_len == len(train)
        assert expected_test_len == len(test)

    def test_author_split_no_leakage(self):
        df = self.df_with_features
        train, test = get_split_by_author(df, test_size=0.3)
        train_owners = set(train["OwnerUserId"].values)
        test_owners = set(test["OwnerUserId"].values)
        assert len(train_owners.intersection(test_owners)) == 0

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
