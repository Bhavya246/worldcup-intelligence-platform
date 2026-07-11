from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from worldcup_intelligence.elo import (  # noqa: E402
    DEFAULT_HOME_ADVANTAGE,
    DEFAULT_RATING,
    EloRatingEngine,
    actual_score,
    apply_home_advantage,
    expected_score,
    expected_scores,
    goal_difference_multiplier,
    initialize_team,
    parse_neutral_flag,
    update_elo,
)


class EloTests(unittest.TestCase):
    def test_initialize_team_adds_unknown_team_without_overwriting_existing_rating(self):
        ratings = {"Brazil": 1600.0}

        self.assertEqual(initialize_team(ratings, "Argentina"), DEFAULT_RATING)
        self.assertEqual(initialize_team(ratings, "Brazil"), 1600.0)
        self.assertEqual(ratings, {"Brazil": 1600.0, "Argentina": DEFAULT_RATING})

    def test_expected_score_is_balanced_for_equal_ratings(self):
        self.assertAlmostEqual(expected_score(1500, 1500), 0.5)

    def test_expected_score_favors_higher_rated_team(self):
        self.assertGreater(expected_score(1700, 1500), 0.5)
        self.assertLess(expected_score(1500, 1700), 0.5)

    def test_expected_scores_apply_home_advantage_unless_neutral(self):
        home_expected, away_expected = expected_scores(1500, 1500)
        neutral_home_expected, neutral_away_expected = expected_scores(
            1500,
            1500,
            neutral=True,
        )

        self.assertGreater(home_expected, 0.5)
        self.assertLess(away_expected, 0.5)
        self.assertAlmostEqual(neutral_home_expected, 0.5)
        self.assertAlmostEqual(neutral_away_expected, 0.5)

    def test_actual_score_handles_home_win_away_win_and_draw(self):
        self.assertEqual(actual_score(3, 0), (1.0, 0.0))
        self.assertEqual(actual_score(0, 2), (0.0, 1.0))
        self.assertEqual(actual_score(1, 1), (0.5, 0.5))

    def test_apply_home_advantage_respects_neutral_flag(self):
        self.assertEqual(
            apply_home_advantage(1500, neutral=False),
            1500 + DEFAULT_HOME_ADVANTAGE,
        )
        self.assertEqual(apply_home_advantage(1500, neutral=True), 1500)

    def test_goal_difference_multiplier_rewards_decisive_wins(self):
        self.assertEqual(goal_difference_multiplier(1, 0, 1500, 1500), 1.0)
        self.assertGreater(goal_difference_multiplier(4, 0, 1500, 1500), 1.0)

    def test_update_elo_applies_rating_delta(self):
        self.assertAlmostEqual(update_elo(1500, expected=0.5, actual=1.0, k=20), 1510)

    def test_engine_process_match_updates_both_teams_with_tournament_k_factor(self):
        engine = EloRatingEngine()

        result = engine.process_match(
            home_team="Brazil",
            away_team="Argentina",
            home_score=2,
            away_score=1,
            tournament="Friendly",
            neutral=True,
        )

        self.assertEqual(result["k_factor"], 20)
        self.assertAlmostEqual(engine.ratings["Brazil"], 1510)
        self.assertAlmostEqual(engine.ratings["Argentina"], 1490)
        self.assertEqual(len(engine.history), 2)

    def test_engine_home_advantage_reduces_home_win_rating_gain(self):
        neutral_engine = EloRatingEngine()
        home_advantage_engine = EloRatingEngine()

        neutral_result = neutral_engine.process_match(
            "Brazil",
            "Argentina",
            2,
            1,
            tournament="Friendly",
            neutral=True,
        )
        home_advantage_result = home_advantage_engine.process_match(
            "Brazil",
            "Argentina",
            2,
            1,
            tournament="Friendly",
            neutral=False,
        )

        self.assertGreater(home_advantage_result["home_expected"], 0.5)
        self.assertLess(
            home_advantage_result["home_rating_after"],
            neutral_result["home_rating_after"],
        )

    def test_engine_process_matches_accepts_iterable_rows(self):
        engine = EloRatingEngine()

        ratings = engine.process_matches(
            [
                {
                    "home_team": "Brazil",
                    "away_team": "Argentina",
                    "home_score": 2,
                    "away_score": 1,
                    "tournament": "Friendly",
                    "neutral": "TRUE",
                    "date": "2020-01-01",
                },
                {
                    "home_team": "Argentina",
                    "away_team": "Brazil",
                    "home_score": 0,
                    "away_score": 0,
                    "tournament": "Friendly",
                    "neutral": "TRUE",
                    "date": "2020-01-02",
                },
            ]
        )

        self.assertEqual(set(ratings), {"Brazil", "Argentina"})
        self.assertGreater(ratings["Brazil"], ratings["Argentina"])

    def test_engine_sorts_matches_chronologically(self):
        engine = EloRatingEngine()

        engine.process_matches(
            [
                {
                    "date": "2020-01-02",
                    "home_team": "Brazil",
                    "away_team": "Argentina",
                    "home_score": 0,
                    "away_score": 1,
                    "tournament": "Friendly",
                    "neutral": True,
                },
                {
                    "date": "2020-01-01",
                    "home_team": "Brazil",
                    "away_team": "Argentina",
                    "home_score": 1,
                    "away_score": 0,
                    "tournament": "Friendly",
                    "neutral": True,
                },
            ]
        )

        self.assertEqual(engine.history[0].date, "2020-01-01")
        self.assertEqual(engine.history[2].date, "2020-01-02")

    def test_engine_exposes_rankings_and_team_history(self):
        engine = EloRatingEngine()
        engine.process_match("Brazil", "Argentina", 3, 0, tournament="Friendly", neutral=True)

        rankings = engine.get_rankings()
        brazil_history = engine.get_rating_history("Brazil")

        self.assertEqual(rankings[0]["team"], "Brazil")
        self.assertEqual(rankings[0]["rank"], 1)
        self.assertEqual(len(brazil_history), 1)
        self.assertGreater(brazil_history[0]["rating_change"], 0)

    def test_parse_neutral_flag_handles_common_input_forms(self):
        self.assertTrue(parse_neutral_flag(True))
        self.assertTrue(parse_neutral_flag("TRUE"))
        self.assertTrue(parse_neutral_flag("yes"))
        self.assertFalse(parse_neutral_flag(False))
        self.assertFalse(parse_neutral_flag("FALSE"))


if __name__ == "__main__":
    unittest.main()
