"""
Unified Prediction API for the WorldCup Intelligence Platform.

This module provides a single interface to generate match predictions
from any trained model, handling all feature construction internally.

Usage:
    from worldcup_intelligence.predictor import MatchPredictor

    predictor = MatchPredictor.load()
    result = predictor.predict("Argentina", "Spain", neutral=True)
    print(result)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ModelName = Literal["logistic", "xgboost", "xgboost_calibrated"]


@dataclass(frozen=True)
class MatchPrediction:
    """Complete prediction result for a single match."""
    home_team: str
    away_team: str
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    predicted_winner: str
    confidence: float
    home_elo: float
    away_elo: float
    elo_diff: float
    model_used: str
    neutral_venue: bool
    tournament: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"Match: {self.home_team} vs {self.away_team}\n"
            f"  {self.home_team} win:  {self.home_win_probability*100:.1f}%\n"
            f"  Draw:         {self.draw_probability*100:.1f}%\n"
            f"  {self.away_team} win:  {self.away_win_probability*100:.1f}%\n"
            f"  Predicted:    {self.predicted_winner} "
            f"(confidence: {self.confidence*100:.1f}%)\n"
            f"  Elo:          {self.home_team} {self.home_elo:.0f} | "
            f"{self.away_team} {self.away_elo:.0f}\n"
            f"  Model:        {self.model_used}"
        )


class MatchPredictor:
    """
    Unified prediction interface for international football matches.

    Loads trained models and Elo ratings once, then exposes a clean
    predict() method that handles all feature construction internally.
    """

    def __init__(self, model_name: ModelName = "logistic") -> None:
        self.model_name = model_name
        self._model: Any = None
        self._elo_engine: Any = None

    @classmethod
    def load(
        cls,
        model_name: ModelName = "logistic",
        dataset_path: Path | str | None = None,
    ) -> MatchPredictor:
        """Load a trained predictor ready to make predictions."""
        instance = cls(model_name=model_name)
        instance._load_model()
        instance._build_elo_engine(dataset_path)
        return instance

    def _load_model(self) -> None:
        import joblib
        model_paths = {
            "logistic": PROJECT_ROOT / "models" / "logistic_regression.joblib",
            "xgboost": PROJECT_ROOT / "models" / "xgboost_model.joblib",
            "xgboost_calibrated": PROJECT_ROOT / "models" / "xgboost_calibrated.joblib",
        }
        path = model_paths[self.model_name]
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        self._model = joblib.load(path)

    def _build_elo_engine(self, dataset_path: Path | str | None) -> None:
        from worldcup_intelligence.elo import EloRatingEngine
        data_path = Path(dataset_path) if dataset_path else (
            PROJECT_ROOT / "data" / "processed" / "ml_matches.csv"
        )
        df = pd.read_csv(data_path).sort_values("match_index").reset_index(drop=True)
        engine = EloRatingEngine()
        for _, row in df.iterrows():
            engine.process_match(
                home_team=row["home_team"],
                away_team=row["away_team"],
                home_score=int(row["home_score"]),
                away_score=int(row["away_score"]),
                tournament=row.get("tournament"),
                neutral=row.get("neutral", False),
                date=str(row["date"]) if pd.notna(row.get("date")) else None,
            )
        self._elo_engine = engine

    def get_elo(self, team: str) -> float:
        """Return current Elo rating for a team."""
        return self._elo_engine.get_team_rating(team)

    def predict(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False,
        tournament: str = "Friendly",
        home_advantage: float = 75.0,
    ) -> MatchPrediction:
        """Predict the outcome of a match between two teams."""
        from worldcup_intelligence.elo import expected_scores
        from config.k_factors import get_k_factor

        home_elo = self.get_elo(home_team)
        away_elo = self.get_elo(away_team)
        k = get_k_factor(tournament)

        home_expected, away_expected = expected_scores(
            home_rating=home_elo,
            away_rating=away_elo,
            neutral=neutral,
            home_advantage=home_advantage,
        )

        features = pd.DataFrame([{
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "home_expected_elo": home_expected,
            "away_expected_elo": away_expected,
            "neutral": int(neutral),
            "home_advantage_applied": 0.0 if neutral else home_advantage,
            "tournament_k_factor": float(k),
            "home_matches_played": 50,
            "away_matches_played": 50,
            "matches_played_diff": 0,
            "home_recent_points_per_match": 1.5,
            "away_recent_points_per_match": 1.5,
            "recent_points_diff": 0.0,
            "home_recent_goals_for": 1.5,
            "away_recent_goals_for": 1.5,
            "recent_goals_for_diff": 0.0,
            "home_recent_goals_against": 1.0,
            "away_recent_goals_against": 1.0,
            "recent_goals_against_diff": 0.0,
            "home_recent_goal_difference": 0.5,
            "away_recent_goal_difference": 0.5,
            "recent_goal_difference_diff": 0.0,
            "home_rest_days": 7,
            "away_rest_days": 7,
            "rest_days_diff": 0,
            "head_to_head_matches": 0,
            "home_head_to_head_points_per_match": 0.0,
            "away_head_to_head_points_per_match": 0.0,
            "head_to_head_points_diff": 0.0,
        }])

        proba = self._model.predict_proba(features)[0]
        home_prob = round(float(proba[2]), 4)
        draw_prob = round(float(proba[1]), 4)
        away_prob = round(float(proba[0]), 4)

        if home_prob > draw_prob and home_prob > away_prob:
            winner = home_team
        elif away_prob > home_prob and away_prob > draw_prob:
            winner = away_team
        else:
            winner = "Draw"

        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            home_win_probability=home_prob,
            draw_probability=draw_prob,
            away_win_probability=away_prob,
            predicted_winner=winner,
            confidence=round(float(max(proba)), 4),
            home_elo=round(home_elo, 1),
            away_elo=round(away_elo, 1),
            elo_diff=round(home_elo - away_elo, 1),
            model_used=self.model_name,
            neutral_venue=neutral,
            tournament=tournament,
        )

    def predict_tournament(
        self,
        matches: list[dict[str, Any]],
    ) -> list[MatchPrediction]:
        """Predict multiple matches at once."""
        return [
            self.predict(
                home_team=m["home_team"],
                away_team=m["away_team"],
                neutral=m.get("neutral", True),
                tournament=m.get("tournament", "FIFA World Cup"),
            )
            for m in matches
        ]

    def rankings(self, limit: int = 20) -> pd.DataFrame:
        """Return current Elo rankings as a DataFrame."""
        rows = self._elo_engine.get_rankings(limit=limit)
        return pd.DataFrame(rows)