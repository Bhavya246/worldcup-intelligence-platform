"""
Poisson-based scoreline predictor for the WorldCup Intelligence Platform.

Models each team's goals as an independent Poisson process.
Predicts expected goals, most likely scorelines, and full
score probability matrices.

Reference: Dixon & Coles (1997) — modelling association football
scores and inefficiencies in the football betting market.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ml_matches.csv"
HOME_MODEL_PATH = PROJECT_ROOT / "models" / "poisson_home.joblib"
AWAY_MODEL_PATH = PROJECT_ROOT / "models" / "poisson_away.joblib"

FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_expected_elo",
    "away_expected_elo",
    "neutral",
    "home_advantage_applied",
    "tournament_k_factor",
    "home_recent_goals_for",
    "away_recent_goals_for",
    "home_recent_goals_against",
    "away_recent_goals_against",
    "home_recent_goal_difference",
    "away_recent_goal_difference",
    "home_recent_points_per_match",
    "away_recent_points_per_match",
    "head_to_head_matches",
    "home_head_to_head_points_per_match",
    "away_head_to_head_points_per_match",
]

MAX_GOALS = 8


@dataclass
class ScorePrediction:
    home_team: str
    away_team: str
    home_expected_goals: float
    away_expected_goals: float
    most_likely_score: str
    most_likely_score_probability: float
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    top_scorelines: list[dict[str, Any]]
    score_matrix: np.ndarray

    def __str__(self) -> str:
        top = self.top_scorelines[:5]
        lines = [
            f"Score Prediction: {self.home_team} vs {self.away_team}",
            f"  Expected Goals: {self.home_team} {self.home_expected_goals:.2f} — {self.away_team} {self.away_expected_goals:.2f}",
            f"  Most Likely Score: {self.most_likely_score} ({self.most_likely_score_probability*100:.1f}%)",
            f"  Outcome: {self.home_team} {self.home_win_probability*100:.1f}% | Draw {self.draw_probability*100:.1f}% | {self.away_team} {self.away_win_probability*100:.1f}%",
            "  Top Scorelines:",
        ]
        for s in top:
            lines.append(f"    {s['score']:>5}  {s['probability']*100:.1f}%")
        return "\n".join(lines)


class PoissonScorePredictor:
    """
    Trains two Poisson regressors — one for home goals, one for away goals.
    Uses the same feature set as the outcome models to stay consistent.
    """

    def __init__(self, test_size: float = 0.20) -> None:
        self.test_size = test_size
        self.home_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", PoissonRegressor(max_iter=1000, alpha=0.1)),
        ])
        self.away_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", PoissonRegressor(max_iter=1000, alpha=0.1)),
        ])
        self.is_trained = False

    @staticmethod
    def load_dataset(path: Path | str = DATA_PATH) -> pd.DataFrame:
        df = pd.read_csv(path)
        return df.sort_values("match_index").reset_index(drop=True)

    def chronological_split(self, df: pd.DataFrame):
        split = int(len(df) * (1 - self.test_size))
        return df.iloc[:split], df.iloc[split:]

    def train(self, df: pd.DataFrame) -> None:
        train, _ = self.chronological_split(df)
        X = train[FEATURE_COLUMNS]
        self.home_pipeline.fit(X, train["home_score"])
        self.away_pipeline.fit(X, train["away_score"])
        self.is_trained = True

    def evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        _, test = self.chronological_split(df)
        X = test[FEATURE_COLUMNS]
        home_pred = self.home_pipeline.predict(X)
        away_pred = self.away_pipeline.predict(X)
        return {
            "home_goals_mae": mean_absolute_error(test["home_score"], home_pred),
            "away_goals_mae": mean_absolute_error(test["away_score"], away_pred),
            "home_goals_rmse": float(np.sqrt(mean_squared_error(test["home_score"], home_pred))),
            "away_goals_rmse": float(np.sqrt(mean_squared_error(test["away_score"], away_pred))),
        }

    def train_and_evaluate(self, path: Path | str = DATA_PATH) -> dict[str, float]:
        df = self.load_dataset(path)
        self.train(df)
        return self.evaluate(df)

    def save(self) -> None:
        joblib.dump(self.home_pipeline, HOME_MODEL_PATH)
        joblib.dump(self.away_pipeline, AWAY_MODEL_PATH)

    def load(self) -> None:
        self.home_pipeline = joblib.load(HOME_MODEL_PATH)
        self.away_pipeline = joblib.load(AWAY_MODEL_PATH)
        self.is_trained = True

    def predict_score(
        self,
        features: pd.DataFrame,
        home_team: str,
        away_team: str,
    ) -> ScorePrediction:
        """Generate full scoreline distribution from a feature row."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        home_lambda = float(self.home_pipeline.predict(features)[0])
        away_lambda = float(self.away_pipeline.predict(features)[0])
        home_lambda = max(0.1, home_lambda)
        away_lambda = max(0.1, away_lambda)

        home_probs = np.array([poisson.pmf(g, home_lambda) for g in range(MAX_GOALS + 1)])
        away_probs = np.array([poisson.pmf(g, away_lambda) for g in range(MAX_GOALS + 1)])
        score_matrix = np.outer(home_probs, away_probs)

        home_win_prob = float(np.sum(np.tril(score_matrix, -1)))
        draw_prob = float(np.sum(np.diag(score_matrix)))
        away_win_prob = float(np.sum(np.triu(score_matrix, 1)))

        scorelines = []
        for h in range(MAX_GOALS + 1):
            for a in range(MAX_GOALS + 1):
                scorelines.append({
                    "score": f"{h}-{a}",
                    "home_goals": h,
                    "away_goals": a,
                    "probability": float(score_matrix[h, a]),
                })
        scorelines.sort(key=lambda x: x["probability"], reverse=True)

        best = scorelines[0]

        return ScorePrediction(
            home_team=home_team,
            away_team=away_team,
            home_expected_goals=round(home_lambda, 2),
            away_expected_goals=round(away_lambda, 2),
            most_likely_score=best["score"],
            most_likely_score_probability=best["probability"],
            home_win_probability=round(home_win_prob, 4),
            draw_probability=round(draw_prob, 4),
            away_win_probability=round(away_win_prob, 4),
            top_scorelines=scorelines[:10],
            score_matrix=score_matrix,
        )