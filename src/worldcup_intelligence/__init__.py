"""Core package for the WorldCup Intelligence Platform."""

from worldcup_intelligence.elo import (
    DEFAULT_HOME_ADVANTAGE,
    DEFAULT_RATING,
    EloRatingEngine,
    RatingHistoryEntry,
    actual_score,
    apply_home_advantage,
    expected_score,
    expected_scores,
    goal_difference_multiplier,
    initialize_team,
    parse_neutral_flag,
    update_elo,
)

__all__ = [
    "DEFAULT_HOME_ADVANTAGE",
    "DEFAULT_RATING",
    "EloRatingEngine",
    "RatingHistoryEntry",
    "actual_score",
    "apply_home_advantage",
    "expected_score",
    "expected_scores",
    "goal_difference_multiplier",
    "initialize_team",
    "parse_neutral_flag",
    "update_elo",
]
