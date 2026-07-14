"""
Dataset utilities for the WorldCup Intelligence Platform.

Responsibilities
----------------
- Load the ML-ready dataset.
- Validate required columns.
- Select model features.
- Perform chronological train/validation/test split.
- Return ready-to-use pandas DataFrames.

No machine learning happens here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_matches.csv"
)

TARGET_COLUMN = "target_code"


FEATURE_COLUMNS = [

    "home_elo",
    "away_elo",
    "elo_diff",

    "home_expected_elo",
    "away_expected_elo",

    "neutral",
    "home_advantage_applied",

    "tournament_k_factor",

    "home_matches_played",
    "away_matches_played",
    "matches_played_diff",

    "home_recent_points_per_match",
    "away_recent_points_per_match",
    "recent_points_diff",

    "home_recent_goals_for",
    "away_recent_goals_for",
    "recent_goals_for_diff",

    "home_recent_goals_against",
    "away_recent_goals_against",
    "recent_goals_against_diff",

    "home_recent_goal_difference",
    "away_recent_goal_difference",
    "recent_goal_difference_diff",

    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",

    "head_to_head_matches",

    "home_head_to_head_points_per_match",
    "away_head_to_head_points_per_match",
    "head_to_head_points_diff",
]


@dataclass(slots=True)
class DatasetSplit:

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame

    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


class MatchDataset:

    def __init__(
        self,
        dataset_path: str | Path = DEFAULT_DATASET,
    ) -> None:

        self.dataset_path = Path(dataset_path)

        self.dataframe: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:

        dataframe = pd.read_csv(self.dataset_path)

        dataframe = dataframe.sort_values(
            "match_index"
        ).reset_index(drop=True)

        self._validate_columns(dataframe)

        self.dataframe = dataframe

        return dataframe

    @staticmethod
    def _validate_columns(
        dataframe: pd.DataFrame,
    ) -> None:

        missing = []

        for column in FEATURE_COLUMNS:

            if column not in dataframe.columns:

                missing.append(column)

        if TARGET_COLUMN not in dataframe.columns:

            missing.append(TARGET_COLUMN)

        if missing:

            raise ValueError(
                f"Dataset missing columns: {missing}"
            )

    def features(self) -> pd.DataFrame:

        if self.dataframe is None:

            self.load()

        return self.dataframe[FEATURE_COLUMNS].copy()

    def target(self) -> pd.Series:

        if self.dataframe is None:

            self.load()

        return self.dataframe[TARGET_COLUMN].copy()

    def split(
        self,
        train_size: float = 0.70,
        validation_size: float = 0.15,
    ) -> DatasetSplit:

        if self.dataframe is None:

            self.load()

        dataframe = self.dataframe

        total_rows = len(dataframe)

        train_end = int(
            total_rows * train_size
        )

        validation_end = int(
            total_rows
            * (train_size + validation_size)
        )

        train = dataframe.iloc[:train_end]

        validation = dataframe.iloc[
            train_end:validation_end
        ]

        test = dataframe.iloc[
            validation_end:
        ]

        return DatasetSplit(

            X_train=train[FEATURE_COLUMNS],

            X_validation=validation[
                FEATURE_COLUMNS
            ],

            X_test=test[
                FEATURE_COLUMNS
            ],

            y_train=train[
                TARGET_COLUMN
            ],

            y_validation=validation[
                TARGET_COLUMN
            ],

            y_test=test[
                TARGET_COLUMN
            ],
        )

    @property
    def number_of_matches(
        self,
    ) -> int:

        if self.dataframe is None:

            self.load()

        return len(self.dataframe)

    @property
    def number_of_features(
        self,
    ) -> int:

        return len(FEATURE_COLUMNS)

    @property
    def class_distribution(
        self,
    ) -> pd.Series:

        if self.dataframe is None:

            self.load()

        return (
            self.dataframe["target"]
            .value_counts()
            .sort_index()
        )