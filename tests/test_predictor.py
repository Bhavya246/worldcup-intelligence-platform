"""Tests for the WorldCup Intelligence Platform."""

from __future__ import annotations

import pytest
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from worldcup_intelligence.elo import (
    EloRatingEngine,
    expected_score,
    actual_score,
    goal_difference_multiplier,
    parse_neutral_flag,
    parse_match_date,
)
from worldcup_intelligence.features.builder import (
    build_feature_rows,
    points_from_score,
    result_label,
)
from worldcup_intelligence.models.logistic import LogisticMatchPredictor
from worldcup_intelligence.models.xgboost_model import XGBoostMatchPredictor
from worldcup_intelligence.predictor import MatchPredictor, MatchPrediction


# ── Elo engine tests ─────────────────────────────────────────────

class TestEloEngine:

    def test_initial_rating(self):
        engine = EloRatingEngine()
        assert engine.get_team_rating("NewTeam") == 1500.0

    def test_winner_gains_rating(self):
        engine = EloRatingEngine()
        engine.process_match("Brazil", "Argentina", 2, 0)
        assert engine.get_team_rating("Brazil") > 1500.0
        assert engine.get_team_rating("Argentina") < 1500.0

    def test_draw_equal_teams_no_change(self):
        engine = EloRatingEngine()
        engine.process_match("France", "Spain", 1, 1)
        france = engine.get_team_rating("France")
        spain = engine.get_team_rating("Spain")
        assert abs(france - spain) < 10.0

    def test_ratings_sum_to_constant(self):
        engine = EloRatingEngine()
        engine.process_match("Germany", "Italy", 3, 1)
        total = engine.get_team_rating("Germany") + engine.get_team_rating("Italy")
        assert abs(total - 3000.0) < 0.001

    def test_home_advantage_applied(self):
        engine = EloRatingEngine(home_advantage=100.0)
        result = engine.process_match("England", "France", 1, 1, neutral=False)
        assert result["adjusted_home_rating"] == 1600.0

    def test_neutral_venue_no_advantage(self):
        engine = EloRatingEngine(home_advantage=100.0)
        result = engine.process_match("England", "France", 1, 1, neutral=True)
        assert result["adjusted_home_rating"] == 1500.0

    def test_rankings_sorted(self):
        engine = EloRatingEngine()
        engine.process_match("Brazil", "Argentina", 3, 0)
        engine.process_match("France", "Germany", 2, 0)
        rankings = engine.get_rankings()
        ratings = [r["rating"] for r in rankings]
        assert ratings == sorted(ratings, reverse=True)

    def test_history_recorded(self):
        engine = EloRatingEngine()
        engine.process_match("Brazil", "Argentina", 2, 1)
        assert len(engine.history) == 2

    def test_process_multiple_matches(self):
        engine = EloRatingEngine()
        matches = [
            {"home_team": "Brazil", "away_team": "Argentina", "home_score": 2, "away_score": 1, "date": "2024-01-01"},
            {"home_team": "France", "away_team": "Spain", "home_score": 1, "away_score": 0, "date": "2024-01-02"},
        ]
        engine.process_matches(matches)
        assert engine.match_count == 2


# ── Expected / actual score tests ────────────────────────────────

class TestExpectedScore:

    def test_equal_teams(self):
        score = expected_score(1500, 1500)
        assert abs(score - 0.5) < 0.001

    def test_stronger_team_higher_expected(self):
        assert expected_score(1600, 1500) > expected_score(1500, 1600)

    def test_actual_score_home_win(self):
        assert actual_score(2, 0) == (1.0, 0.0)

    def test_actual_score_away_win(self):
        assert actual_score(0, 1) == (0.0, 1.0)

    def test_actual_score_draw(self):
        assert actual_score(1, 1) == (0.5, 0.5)


# ── Goal difference multiplier tests ─────────────────────────────

class TestGoalDifferenceMultiplier:

    def test_draw_returns_one(self):
        assert goal_difference_multiplier(1, 1, 1500, 1500) == 1.0

    def test_one_goal_returns_one(self):
        assert goal_difference_multiplier(1, 0, 1500, 1500) == 1.0

    def test_large_margin_greater_than_one(self):
        assert goal_difference_multiplier(5, 0, 1500, 1500) > 1.0

    def test_underdog_win_higher_multiplier(self):
        underdog = goal_difference_multiplier(3, 0, 1400, 1600)
        favorite = goal_difference_multiplier(3, 0, 1600, 1400)
        assert underdog > favorite


# ── Parse utility tests ──────────────────────────────────────────

class TestParseUtilities:

    def test_parse_neutral_true(self):
        for val in [True, 1, "true", "True", "1", "yes", "y"]:
            assert parse_neutral_flag(val) is True

    def test_parse_neutral_false(self):
        for val in [False, 0, "false", "False", "0", "no"]:
            assert parse_neutral_flag(val) is False

    def test_parse_date_valid(self):
        from datetime import datetime
        result = parse_match_date("2024-07-14")
        assert isinstance(result, datetime)

    def test_parse_date_none(self):
        assert parse_match_date(None) is None


# ── Feature builder tests ────────────────────────────────────────

class TestFeatureBuilder:

    def test_points_from_score_win(self):
        assert points_from_score(2, 0) == 3

    def test_points_from_score_draw(self):
        assert points_from_score(1, 1) == 1

    def test_points_from_score_loss(self):
        assert points_from_score(0, 2) == 0

    def test_result_label_home_win(self):
        assert result_label(2, 0) == "home_win"

    def test_result_label_away_win(self):
        assert result_label(0, 1) == "away_win"

    def test_result_label_draw(self):
        assert result_label(1, 1) == "draw"

    def test_build_feature_rows_no_leakage(self):
        matches = [
            {"home_team": "A", "away_team": "B", "home_score": 2,
             "away_score": 1, "date": "2024-01-01", "tournament": "Friendly"},
            {"home_team": "B", "away_team": "A", "home_score": 0,
             "away_score": 0, "date": "2024-01-08", "tournament": "Friendly"},
        ]
        rows = build_feature_rows(matches)
        assert rows[0]["home_elo"] == 1500.0
        assert rows[0]["away_elo"] == 1500.0
        assert rows[1]["home_elo"] != 1500.0

    def test_build_feature_rows_count(self):
        matches = [
            {"home_team": "A", "away_team": "B", "home_score": 1,
             "away_score": 0, "date": "2024-01-01", "tournament": "Friendly"},
        ]
        rows = build_feature_rows(matches)
        assert len(rows) == 1


# ── Prediction API tests ─────────────────────────────────────────

class TestMatchPrediction:

    def test_probabilities_sum_to_one(self):
        pred = MatchPrediction(
            home_team="Argentina", away_team="Spain",
            home_win_probability=0.411, draw_probability=0.212,
            away_win_probability=0.377, predicted_winner="Argentina",
            confidence=0.411, home_elo=2092.0, away_elo=2068.0,
            elo_diff=24.0, model_used="logistic",
            neutral_venue=True, tournament="FIFA World Cup",
        )
        total = pred.home_win_probability + pred.draw_probability + pred.away_win_probability
        assert abs(total - 1.0) < 0.01

    def test_prediction_to_dict(self):
        pred = MatchPrediction(
            home_team="Argentina", away_team="Spain",
            home_win_probability=0.411, draw_probability=0.212,
            away_win_probability=0.377, predicted_winner="Argentina",
            confidence=0.411, home_elo=2092.0, away_elo=2068.0,
            elo_diff=24.0, model_used="logistic",
            neutral_venue=True, tournament="FIFA World Cup",
        )
        d = pred.to_dict()
        assert d["home_team"] == "Argentina"
        assert d["predicted_winner"] == "Argentina"


# ── Model tests ──────────────────────────────────────────────────

class TestLogisticPredictor:

    def test_predict_match_returns_probabilities(self):
        predictor = LogisticMatchPredictor()
        predictor.load_model(ROOT / "models" / "logistic_regression.joblib")
        result = predictor.predict_match(
            home_team="Argentina", away_team="Spain",
            home_elo=2092.0, away_elo=2068.0, neutral=True,
        )
        total = result["home_win_probability"] + result["draw_probability"] + result["away_win_probability"]
        assert abs(total - 1.0) < 0.001

    def test_predict_match_has_winner(self):
        predictor = LogisticMatchPredictor()
        predictor.load_model(ROOT / "models" / "logistic_regression.joblib")
        result = predictor.predict_match(
            home_team="Brazil", away_team="Fiji",
            home_elo=1985.0, away_elo=1400.0, neutral=False,
        )
        assert result["predicted_winner"] == "Brazil"


class TestXGBoostPredictor:

    def test_predict_match_returns_probabilities(self):
        predictor = XGBoostMatchPredictor()
        predictor.load_model(ROOT / "models" / "xgboost_model.joblib")
        result = predictor.predict_match(
            home_team="Argentina", away_team="Spain",
            home_elo=2092.0, away_elo=2068.0, neutral=True,
        )
        total = result["home_win_probability"] + result["draw_probability"] + result["away_win_probability"]
        assert abs(total - 1.0) < 0.001

    def test_feature_importance_shape(self):
        predictor = XGBoostMatchPredictor()
        predictor.load_model(ROOT / "models" / "xgboost_model.joblib")
        fi = predictor.feature_importance
        assert len(fi) == 30
        assert "feature" in fi.columns
        assert "importance" in fi.columns