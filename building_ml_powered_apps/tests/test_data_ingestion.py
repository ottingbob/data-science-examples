import os
from functools import cached_property
from pathlib import Path
from unittest import TestCase

# Import example file from project to ensure that coverage is working on only source
# code imported in test files
import pandas as pd
from building_ml_powered_apps.tests.helpers import parse_xml_to_csv


class DataIngestionTests(TestCase):

    REQUIRED_COLUMNS = [
        "Id",
        "AnswerCount",
        "PostTypeId",
        "AcceptedAnswerId",
        "Body",
        "body_text",
        "Title",
        "Score",
    ]

    @cached_property
    def fixture_df(self) -> pd.DataFrame:
        """
        Use parser to return DataFrame
        """
        curr_path = Path(os.path.dirname(__file__))
        return parse_xml_to_csv(curr_path)

    def test_parser_returns_dataframe(self):
        """
        Tests that our parser runs and returns a DataFrame
        """
        assert isinstance(self.fixture_df, pd.DataFrame)

    def test_feature_columns_exist(self):
        """
        Validate all required columns are present
        """
        for col in self.REQUIRED_COLUMNS:
            assert col in self.fixture_df.columns

    def test_feature_not_all_null(self):
        """
        Validate no features are missing every value
        """
        for col in self.REQUIRED_COLUMNS:
            assert not self.fixture_df[col].isnull().all()

    def test_text_mean(self):
        """
        Validate text mean matches with exploration expectations
        """
        ACCEPTABLE_TEXT_LENGTH_MEANS = pd.Interval(left=20, right=2000)
        df = self.fixture_df.copy()
        df["text_len"] = df["body_text"].str.len()
        assert df["text_len"].mean() in ACCEPTABLE_TEXT_LENGTH_MEANS
