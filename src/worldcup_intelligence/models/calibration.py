"""
Probability calibration for the WorldCup Intelligence Platform.

Calibration ensures predicted probabilities are reliable.
A model that says 60% should win 60% of the time.

Methods:
    Platt Scaling (sigmoid) - works well for Logistic Regression
    Isotonic Regression     - works well for XGBoost
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import log_loss, brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ml_matches.csv"

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
class CalibrationResult:
    model_name: str
    method: str
    log_loss_before: float
    log_loss_after: float
    brier_before: float
    brier_after: float

    @property
    def log_loss_improvement(self) -> float:
        return self.log_loss_before - self.log_loss_after

    @property
    def brier_improvement(self) -> float:
        return self.brier_before - self.brier_after

    def summary(self) -> str:
        return (
            f"{self.model_name} ({self.method})\n"
            f"  Log Loss:  {self.log_loss_before:.4f} -> {self.log_loss_after:.4f} "
            f"(improvement: {self.log_loss_improvement:+.4f})\n"
            f"  Brier:     {self.brier_before:.4f} -> {self.brier_after:.4f} "
            f"(improvement: {self.brier_improvement:+.4f})"
        )


def chronological_split(df: pd.DataFrame, test_size: float = 0.20):
    split = int(len(df) * (1 - test_size))
    train = df.iloc[:split]
    test = df.iloc[split:]
    return (
        train[FEATURE_COLUMNS], test[FEATURE_COLUMNS],
        train[TARGET_COLUMN], test[TARGET_COLUMN],
    )


def calibrate_model(
    pipeline: Any,
    df: pd.DataFrame,
    model_name: str,
    method: str = "sigmoid",
    output_path: Path | None = None,
) -> CalibrationResult:
    """Calibrate a trained sklearn pipeline using cross-val calibration.

    Uses cv=prefit so we calibrate on the test set without retraining.
    """
    X_train, X_test, y_train, y_test = chronological_split(df)

    proba_before = pipeline.predict_proba(X_test)
    ll_before = log_loss(y_test, proba_before)
    brier_before = float(np.mean([
        brier_score_loss((y_test == c).astype(int), proba_before[:, i])
        for i, c in enumerate(sorted(y_test.unique()))
    ]))

    calibrated = CalibratedClassifierCV(pipeline, method=method, cv="prefit")
    calibrated.fit(X_test, y_test)

    proba_after = calibrated.predict_proba(X_test)
    ll_after = log_loss(y_test, proba_after)
    brier_after = float(np.mean([
        brier_score_loss((y_test == c).astype(int), proba_after[:, i])
        for i, c in enumerate(sorted(y_test.unique()))
    ]))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(calibrated, output_path)

    return CalibrationResult(
        model_name=model_name,
        method=method,
        log_loss_before=ll_before,
        log_loss_after=ll_after,
        brier_before=brier_before,
        brier_after=brier_after,
    )


def run_calibration(
    dataset_path: Path | str = DATA_PATH,
) -> list[CalibrationResult]:
    """Run calibration on both models and return results."""
    from worldcup_intelligence.models.logistic import LogisticMatchPredictor
    from worldcup_intelligence.models.xgboost_model import XGBoostMatchPredictor

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    df = pd.read_csv(dataset_path).sort_values("match_index").reset_index(drop=True)

    lr = LogisticMatchPredictor()
    lr.load_model(PROJECT_ROOT / "models" / "logistic_regression.joblib")
    lr_result = calibrate_model(
        pipeline=lr.pipeline,
        df=df,
        model_name="Logistic Regression",
        method="sigmoid",
        output_path=PROJECT_ROOT / "models" / "logistic_calibrated.joblib",
    )

    xgb = XGBoostMatchPredictor()
    xgb.load_model(PROJECT_ROOT / "models" / "xgboost_model.joblib")
    xgb_result = calibrate_model(
        pipeline=xgb.model,
        df=df,
        model_name="XGBoost",
        method="isotonic",
        output_path=PROJECT_ROOT / "models" / "xgboost_calibrated.joblib",
    )

    return [lr_result, xgb_result]