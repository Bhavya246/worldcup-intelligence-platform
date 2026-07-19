"""
XGBoost model for the WorldCup Intelligence Platform.

Follows the same interface as LogisticMatchPredictor
so both models are interchangeable in the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.calibration import CalibratedClassifierCV

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ml_matches.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.joblib"

FEATURE_COLUMNS = [
    "home_elo", "away_elo", "elo_diff",
    "home_expected_elo", "away_expected_elo",
    "neutral", "home_advantage_applied",
    "tournament_k_factor",
    "home_matches_played", "away_matches_played", "matches_played_diff",
    "home_recent_points_per_match", "away_recent_points_per_match", "recent_points_diff",
    "home_recent_goals_for", "away_recent_goals_for", "recent_goals_for_diff",
    "home_recent_goals_against", "away_recent_goals_against", "recent_goals_against_diff",
    "home_recent_goal_difference", "away_recent_goal_difference", "recent_goal_difference_diff",
    "home_rest_days", "away_rest_days", "rest_days_diff",
    "head_to_head_matches",
    "home_head_to_head_points_per_match", "away_head_to_head_points_per_match",
    "head_to_head_points_diff",
]

TARGET_COLUMN = "target_code"


@dataclass
class EvaluationResult:
    accuracy: float
    log_loss: float
    confusion_matrix: Any
    classification_report: dict[str, Any]


class XGBoostMatchPredictor:

    def __init__(self, random_state: int = 42, test_size: float = 0.20) -> None:
        self.random_state = random_state
        self.test_size = test_size
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_trained = False

    @staticmethod
    def load_dataset(dataset_path: Path | str = DATA_PATH) -> pd.DataFrame:
        df = pd.read_csv(dataset_path)
        return df.sort_values("match_index").reset_index(drop=True)

    def chronological_split(self, df: pd.DataFrame):
        split_index = int(len(df) * (1 - self.test_size))
        train = df.iloc[:split_index]
        test = df.iloc[split_index:]
        return (
            train[FEATURE_COLUMNS], test[FEATURE_COLUMNS],
            train[TARGET_COLUMN], test[TARGET_COLUMN],
        )

    def train(self, df: pd.DataFrame) -> None:
        X_train, _, y_train, _ = self.chronological_split(df)
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, features: pd.DataFrame) -> Any:
        if not self.is_trained:
            raise RuntimeError("Model has not been trained.")
        return self.model.predict(features)

    def predict_proba(self, features: pd.DataFrame) -> Any:
        if not self.is_trained:
            raise RuntimeError("Model has not been trained.")
        return self.model.predict_proba(features)

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        home_elo: float,
        away_elo: float,
        neutral: bool = False,
        home_advantage: float = 75.0,
        tournament_k_factor: float = 20.0,
        home_matches_played: int = 50,
        away_matches_played: int = 50,
        home_recent_points_per_match: float = 1.5,
        away_recent_points_per_match: float = 1.5,
        home_recent_goals_for: float = 1.5,
        away_recent_goals_for: float = 1.5,
        home_recent_goals_against: float = 1.0,
        away_recent_goals_against: float = 1.0,
        home_recent_goal_difference: float = 0.5,
        away_recent_goal_difference: float = 0.5,
        home_rest_days: int = 7,
        away_rest_days: int = 7,
        head_to_head_matches: int = 0,
        home_h2h_points: float = 0.0,
        away_h2h_points: float = 0.0,
    ) -> dict[str, Any]:
        from worldcup_intelligence.elo import expected_scores
        home_expected, away_expected = expected_scores(
            home_rating=home_elo, away_rating=away_elo,
            neutral=neutral, home_advantage=home_advantage,
        )
        features = pd.DataFrame([{
            "home_elo": home_elo, "away_elo": away_elo, "elo_diff": home_elo - away_elo,
            "home_expected_elo": home_expected, "away_expected_elo": away_expected,
            "neutral": int(neutral),
            "home_advantage_applied": 0.0 if neutral else home_advantage,
            "tournament_k_factor": tournament_k_factor,
            "home_matches_played": home_matches_played,
            "away_matches_played": away_matches_played,
            "matches_played_diff": home_matches_played - away_matches_played,
            "home_recent_points_per_match": home_recent_points_per_match,
            "away_recent_points_per_match": away_recent_points_per_match,
            "recent_points_diff": home_recent_points_per_match - away_recent_points_per_match,
            "home_recent_goals_for": home_recent_goals_for,
            "away_recent_goals_for": away_recent_goals_for,
            "recent_goals_for_diff": home_recent_goals_for - away_recent_goals_for,
            "home_recent_goals_against": home_recent_goals_against,
            "away_recent_goals_against": away_recent_goals_against,
            "recent_goals_against_diff": home_recent_goals_against - away_recent_goals_against,
            "home_recent_goal_difference": home_recent_goal_difference,
            "away_recent_goal_difference": away_recent_goal_difference,
            "recent_goal_difference_diff": home_recent_goal_difference - away_recent_goal_difference,
            "home_rest_days": home_rest_days,
            "away_rest_days": away_rest_days,
            "rest_days_diff": home_rest_days - away_rest_days,
            "head_to_head_matches": head_to_head_matches,
            "home_head_to_head_points_per_match": home_h2h_points,
            "away_head_to_head_points_per_match": away_h2h_points,
            "head_to_head_points_diff": home_h2h_points - away_h2h_points,
        }])
        proba = self.predict_proba(features)[0]
        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_win_probability": round(float(proba[2]), 4),
            "draw_probability": round(float(proba[1]), 4),
            "away_win_probability": round(float(proba[0]), 4),
            "predicted_winner": (
                home_team if proba[2] > proba[0] and proba[2] > proba[1]
                else away_team if proba[0] > proba[2] and proba[0] > proba[1]
                else "Draw"
            ),
            "confidence": round(float(max(proba)), 4),
            "home_elo": home_elo,
            "away_elo": away_elo,
            "model": "XGBoost",
        }

    def evaluate(self, df: pd.DataFrame) -> EvaluationResult:
        if not self.is_trained:
            raise RuntimeError("Model has not been trained.")
        _, X_test, _, y_test = self.chronological_split(df)
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)
        return EvaluationResult(
            accuracy=accuracy_score(y_test, predictions),
            log_loss=log_loss(y_test, probabilities),
            confusion_matrix=confusion_matrix(y_test, predictions),
            classification_report=classification_report(y_test, predictions, output_dict=True),
        )

    def train_and_evaluate(self, dataset_path: Path | str = DATA_PATH) -> EvaluationResult:
        df = self.load_dataset(dataset_path)
        self.train(df)
        return self.evaluate(df)

    def save_model(self, output_path: Path | str = MODEL_PATH) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, output_path)

    def load_model(self, model_path: Path | str = MODEL_PATH) -> None:
        self.model = joblib.load(model_path)
        self.is_trained = True

    @property
    def feature_importance(self) -> pd.DataFrame:
        if not self.is_trained:
            raise RuntimeError("Model has not been trained.")
        return pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)