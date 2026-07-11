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

from worldcup_intelligence.features import (  # noqa: E402
    FEATURE_COLUMNS,
    build_feature_rows,
    result_label,
)


class FeatureBuilderTests(unittest.TestCase):
    def test_result_label_handles_all_match_outcomes(self):
        self.assertEqual(result_label(2, 1), "home_win")
        self.assertEqual(result_label(1, 2), "away_win")
        self.assertEqual(result_label(1, 1), "draw")

    def test_build_feature_rows_uses_pre_match_state(self):
        rows = build_feature_rows(
            [
                {
                    "date": "2020-01-01",
                    "home_team": "Brazil",
                    "away_team": "Argentina",
                    "home_score": 2,
                    "away_score": 1,
                    "tournament": "Friendly",
                    "neutral": "TRUE",
                },
                {
                    "date": "2020-01-02",
                    "home_team": "Argentina",
                    "away_team": "Brazil",
                    "home_score": 0,
                    "away_score": 0,
                    "tournament": "Friendly",
                    "neutral": "TRUE",
                },
            ]
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["home_elo"], 1500)
        self.assertEqual(rows[0]["away_elo"], 1500)
        self.assertEqual(rows[0]["home_matches_played"], 0)
        self.assertEqual(rows[0]["away_matches_played"], 0)
        self.assertLess(rows[1]["home_elo"], 1500)
        self.assertGreater(rows[1]["away_elo"], 1500)
        self.assertEqual(rows[1]["home_matches_played"], 1)
        self.assertEqual(rows[1]["away_matches_played"], 1)
        self.assertEqual(rows[1]["home_recent_points_per_match"], 0)
        self.assertEqual(rows[1]["away_recent_points_per_match"], 3)

    def test_build_feature_rows_sorts_chronologically(self):
        rows = build_feature_rows(
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

        self.assertEqual(rows[0]["date"], "2020-01-01")
        self.assertEqual(rows[1]["date"], "2020-01-02")

    def test_build_feature_rows_outputs_declared_feature_columns(self):
        rows = build_feature_rows(
            [
                {
                    "date": "2020-01-01",
                    "home_team": "Brazil",
                    "away_team": "Argentina",
                    "home_score": 1,
                    "away_score": 1,
                    "tournament": "Friendly",
                    "neutral": False,
                }
            ]
        )

        for column in FEATURE_COLUMNS:
            self.assertIn(column, rows[0])

        self.assertEqual(rows[0]["target"], "draw")
        self.assertEqual(rows[0]["target_code"], 1)

    def test_head_to_head_features_are_computed_before_current_match(self):
        rows = build_feature_rows(
            [
                {
                    "date": "2020-01-01",
                    "home_team": "Brazil",
                    "away_team": "Argentina",
                    "home_score": 2,
                    "away_score": 1,
                    "tournament": "Friendly",
                    "neutral": True,
                },
                {
                    "date": "2020-02-01",
                    "home_team": "Argentina",
                    "away_team": "Brazil",
                    "home_score": 1,
                    "away_score": 1,
                    "tournament": "Friendly",
                    "neutral": True,
                },
            ]
        )

        self.assertEqual(rows[0]["head_to_head_matches"], 0)
        self.assertEqual(rows[1]["head_to_head_matches"], 1)
        self.assertEqual(rows[1]["home_head_to_head_points_per_match"], 0)
        self.assertEqual(rows[1]["away_head_to_head_points_per_match"], 3)


if __name__ == "__main__":
    unittest.main()
