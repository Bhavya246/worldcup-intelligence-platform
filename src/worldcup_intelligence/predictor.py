"""
Unified Prediction API for the WorldCup Intelligence Platform.

Inference uses the same feature engineering pipeline as training.
No hardcoded placeholder values — all features are derived from
historical match data via build_feature_rows().

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

ModelName = Literal["logistic", "xgboost", "xgboost_tuned", "xgboost_calibrated", "mlp", "ensemble"]


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


class TeamStateIndex:
    """
    Extracts and indexes the most recent feature state for every team
    from the output of build_feature_rows().
    """

    def __init__(self, feature_rows: list[dict[str, Any]]) -> None:
        self._home_state: dict[str, dict[str, Any]] = {}
        self._away_state: dict[str, dict[str, Any]] = {}
        self._h2h_state: dict[tuple[str, str], dict[str, Any]] = {}
        self._build(feature_rows)

    def _build(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            home = row["home_team"]
            away = row["away_team"]
            self._home_state[home] = row
            self._away_state[away] = row
            key = tuple(sorted((home, away)))
            self._h2h_state[key] = row

    def _default_team_state(self) -> dict[str, float]:
        return {
            "matches_played": 0,
            "recent_points_per_match": 0.0,
            "recent_goals_for": 0.0,
            "recent_goals_against": 0.0,
            "recent_goal_difference": 0.0,
            "rest_days": 0,
        }

    def get_home_features(self, team: str) -> dict[str, Any]:
        if team in self._home_state:
            row = self._home_state[team]
            return {
                "matches_played": row["home_matches_played"],
                "recent_points_per_match": row["home_recent_points_per_match"],
                "recent_goals_for": row["home_recent_goals_for"],
                "recent_goals_against": row["home_recent_goals_against"],
                "recent_goal_difference": row["home_recent_goal_difference"],
                "rest_days": row["home_rest_days"],
            }
        if team in self._away_state:
            row = self._away_state[team]
            return {
                "matches_played": row["away_matches_played"],
                "recent_points_per_match": row["away_recent_points_per_match"],
                "recent_goals_for": row["away_recent_goals_for"],
                "recent_goals_against": row["away_recent_goals_against"],
                "recent_goal_difference": row["away_recent_goal_difference"],
                "rest_days": row["away_rest_days"],
            }
        return self._default_team_state()

    def get_away_features(self, team: str) -> dict[str, Any]:
        if team in self._away_state:
            row = self._away_state[team]
            return {
                "matches_played": row["away_matches_played"],
                "recent_points_per_match": row["away_recent_points_per_match"],
                "recent_goals_for": row["away_recent_goals_for"],
                "recent_goals_against": row["away_recent_goals_against"],
                "recent_goal_difference": row["away_recent_goal_difference"],
                "rest_days": row["away_rest_days"],
            }
        if team in self._home_state:
            row = self._home_state[team]
            return {
                "matches_played": row["home_matches_played"],
                "recent_points_per_match": row["home_recent_points_per_match"],
                "recent_goals_for": row["home_recent_goals_for"],
                "recent_goals_against": row["home_recent_goals_against"],
                "recent_goal_difference": row["home_recent_goal_difference"],
                "rest_days": row["home_rest_days"],
            }
        return self._default_team_state()

    def get_h2h_features(self, home_team: str, away_team: str) -> dict[str, Any]:
        key = tuple(sorted((home_team, away_team)))
        if key in self._h2h_state:
            row = self._h2h_state[key]
            if row["home_team"] == home_team:
                return {
                    "head_to_head_matches": row["head_to_head_matches"],
                    "home_head_to_head_points_per_match": row["home_head_to_head_points_per_match"],
                    "away_head_to_head_points_per_match": row["away_head_to_head_points_per_match"],
                    "head_to_head_points_diff": row["head_to_head_points_diff"],
                }
            else:
                return {
                    "head_to_head_matches": row["head_to_head_matches"],
                    "home_head_to_head_points_per_match": row["away_head_to_head_points_per_match"],
                    "away_head_to_head_points_per_match": row["home_head_to_head_points_per_match"],
                    "head_to_head_points_diff": -row["head_to_head_points_diff"],
                }
        return {
            "head_to_head_matches": 0,
            "home_head_to_head_points_per_match": 0.0,
            "away_head_to_head_points_per_match": 0.0,
            "head_to_head_points_diff": 0.0,
        }


class MatchPredictor:
    """
    Unified prediction interface for international football matches.

    Loads trained models and builds team state from the same feature
    engineering pipeline used during training. No hardcoded defaults.
    """

    def __init__(self, model_name: ModelName = "logistic") -> None:
        self.model_name = model_name
        self._model: Any = None
        self._elo_engine: Any = None
        self._team_state: TeamStateIndex | None = None

    @classmethod
    def load(
        cls,
        model_name: ModelName = "logistic",
        dataset_path: Path | str | None = None,
    ) -> MatchPredictor:
        """Load a trained predictor ready to make predictions."""
        instance = cls(model_name=model_name)
        instance._load_model()
        instance._build_state(dataset_path)
        return instance

    def _load_model(self) -> None:
        import joblib
        if self.model_name == "ensemble":
            self._ensemble = joblib.load(
                PROJECT_ROOT / "models" / "ensemble_final.joblib"
            )
            self._model = None
            return

        model_paths = {
            "logistic": PROJECT_ROOT / "models" / "logistic_regression.joblib",
            "xgboost": PROJECT_ROOT / "models" / "xgboost_model.joblib",
            "xgboost_tuned": PROJECT_ROOT / "models" / "xgboost_tuned.joblib",
            "xgboost_calibrated": PROJECT_ROOT / "models" / "xgboost_calibrated.joblib",
            "mlp": PROJECT_ROOT / "models" / "mlp_model.joblib",
        }
        path = model_paths[self.model_name]
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        self._model = joblib.load(path)
        self._ensemble = None

    def _build_state(self, dataset_path: Path | str | None) -> None:
        """Replay all historical matches through the shared feature pipeline."""
        from worldcup_intelligence.elo import EloRatingEngine
        from worldcup_intelligence.features.builder import build_feature_rows

        data_path = Path(dataset_path) if dataset_path else (
            PROJECT_ROOT / "data" / "processed" / "ml_matches.csv"
        )
        df = pd.read_csv(data_path).sort_values("match_index").reset_index(drop=True)
        matches = df.to_dict(orient="records")

        self._elo_engine = EloRatingEngine()
        feature_rows = build_feature_rows(matches, elo_engine=self._elo_engine)
        self._team_state = TeamStateIndex(feature_rows)

    def get_elo(self, team: str) -> float:
        return self._elo_engine.get_team_rating(team)

    def _build_features(
        self,
        home_team: str,
        away_team: str,
        neutral: bool,
        tournament: str,
        home_advantage: float,
    ) -> pd.DataFrame:
        """Build the full feature row for a prospective match."""
        from worldcup_intelligence.elo import expected_scores
        from config.k_factors import get_k_factor

        home_elo = self.get_elo(home_team)
        away_elo = self.get_elo(away_team)
        home_exp, away_exp = expected_scores(
            home_rating=home_elo,
            away_rating=away_elo,
            neutral=neutral,
            home_advantage=home_advantage,
        )
        home_form = self._team_state.get_home_features(home_team)
        away_form = self._team_state.get_away_features(away_team)
        h2h = self._team_state.get_h2h_features(home_team, away_team)

        return pd.DataFrame([{
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "home_expected_elo": home_exp,
            "away_expected_elo": away_exp,
            "neutral": int(neutral),
            "home_advantage_applied": 0.0 if neutral else home_advantage,
            "tournament_k_factor": float(get_k_factor(tournament)),
            "home_matches_played": home_form["matches_played"],
            "away_matches_played": away_form["matches_played"],
            "matches_played_diff": home_form["matches_played"] - away_form["matches_played"],
            "home_recent_points_per_match": home_form["recent_points_per_match"],
            "away_recent_points_per_match": away_form["recent_points_per_match"],
            "recent_points_diff": home_form["recent_points_per_match"] - away_form["recent_points_per_match"],
            "home_recent_goals_for": home_form["recent_goals_for"],
            "away_recent_goals_for": away_form["recent_goals_for"],
            "recent_goals_for_diff": home_form["recent_goals_for"] - away_form["recent_goals_for"],
            "home_recent_goals_against": home_form["recent_goals_against"],
            "away_recent_goals_against": away_form["recent_goals_against"],
            "recent_goals_against_diff": home_form["recent_goals_against"] - away_form["recent_goals_against"],
            "home_recent_goal_difference": home_form["recent_goal_difference"],
            "away_recent_goal_difference": away_form["recent_goal_difference"],
            "recent_goal_difference_diff": home_form["recent_goal_difference"] - away_form["recent_goal_difference"],
            "home_rest_days": home_form["rest_days"],
            "away_rest_days": away_form["rest_days"],
            "rest_days_diff": home_form["rest_days"] - away_form["rest_days"],
            **h2h,
        }])

    def predict(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False,
        tournament: str = "Friendly",
        home_advantage: float = 75.0,
    ) -> MatchPrediction:
        """Predict the outcome of a match between two teams."""
        home_elo = self.get_elo(home_team)
        away_elo = self.get_elo(away_team)
        features = self._build_features(home_team, away_team, neutral, tournament, home_advantage)

        if self.model_name == "ensemble":
            e = self._ensemble
            w = e["weights"]
            proba = (
                w[0] * e["lr"].predict_proba(features) +
                w[1] * e["xgb_tuned"].predict_proba(features) +
                w[2] * e["mlp"].predict_proba(features)
            )[0]
        else:
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

    def predict_with_score(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False,
        tournament: str = "Friendly",
        home_advantage: float = 75.0,
    ) -> dict[str, Any]:
        """Full prediction: outcome probabilities + scoreline distribution."""
        from worldcup_intelligence.models.score_predictor import PoissonScorePredictor

        outcome = self.predict(home_team, away_team, neutral, tournament, home_advantage)
        features = self._build_features(home_team, away_team, neutral, tournament, home_advantage)

        score_cols = [
            "home_elo", "away_elo", "elo_diff",
            "home_expected_elo", "away_expected_elo",
            "neutral", "home_advantage_applied", "tournament_k_factor",
            "home_recent_goals_for", "away_recent_goals_for",
            "home_recent_goals_against", "away_recent_goals_against",
            "home_recent_goal_difference", "away_recent_goal_difference",
            "home_recent_points_per_match", "away_recent_points_per_match",
            "head_to_head_matches",
            "home_head_to_head_points_per_match",
            "away_head_to_head_points_per_match",
        ]

        scorer = PoissonScorePredictor()
        scorer.load()
        score = scorer.predict_score(features[score_cols], home_team, away_team)

        return {"outcome": outcome, "score": score}

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
