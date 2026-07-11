"""Reusable Elo rating utilities for international football matches."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import math
from typing import Any, Iterable, Mapping, MutableMapping

from config.k_factors import get_k_factor

DEFAULT_RATING = 1500.0
DEFAULT_HOME_ADVANTAGE = 75.0
MIN_GOAL_DIFFERENCE_MULTIPLIER = 1.0


@dataclass(frozen=True)
class RatingHistoryEntry:
    """Rating state for one team after one processed match."""

    match_index: int
    date: str | None
    team: str
    opponent: str
    side: str
    tournament: str | None
    neutral: bool
    rating_before: float
    rating_after: float
    opponent_rating_before: float
    expected_result: float
    actual_result: float
    k_factor: float
    goal_difference_multiplier: float

    @property
    def rating_change(self) -> float:
        return self.rating_after - self.rating_before

    def to_dict(self) -> dict[str, float | int | str | bool | None]:
        data = asdict(self)
        data["rating_change"] = self.rating_change
        return data


def initialize_team(
    ratings: MutableMapping[str, float],
    team: str,
    initial_rating: float = DEFAULT_RATING,
) -> float:
    """Ensure a team has a rating and return the current rating."""
    if team not in ratings:
        ratings[team] = float(initial_rating)

    return ratings[team]


def expected_score(team_rating: float, opponent_rating: float) -> float:
    """Calculate the expected score for one team against an opponent."""
    return 1 / (1 + 10 ** ((opponent_rating - team_rating) / 400))


def expected_scores(
    home_rating: float,
    away_rating: float,
    neutral: bool = False,
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
) -> tuple[float, float]:
    """Calculate home and away expected scores with venue adjustment."""
    adjusted_home_rating = apply_home_advantage(
        home_rating=home_rating,
        neutral=neutral,
        home_advantage=home_advantage,
    )
    home_expected = expected_score(adjusted_home_rating, away_rating)
    return home_expected, 1 - home_expected


def actual_score(home_score: int, away_score: int) -> tuple[float, float]:
    """Return home and away actual Elo scores from a final scoreline."""
    if home_score > away_score:
        return 1.0, 0.0

    if away_score > home_score:
        return 0.0, 1.0

    return 0.5, 0.5


def apply_home_advantage(
    home_rating: float,
    neutral: bool = False,
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
) -> float:
    """Return the home team's matchup rating after venue adjustment."""
    if neutral:
        return home_rating

    return home_rating + home_advantage


def goal_difference_multiplier(
    home_score: int,
    away_score: int,
    home_rating: float,
    away_rating: float,
) -> float:
    """Return a conservative multiplier for decisive wins.

    The multiplier stays at 1.0 for draws and one-goal wins, then grows
    logarithmically for larger margins. The rating-difference term rewards
    decisive underdog wins more than decisive favorite wins.
    """
    goal_difference = abs(home_score - away_score)
    if goal_difference <= 1:
        return MIN_GOAL_DIFFERENCE_MULTIPLIER

    winner_rating_advantage = (
        home_rating - away_rating if home_score > away_score else away_rating - home_rating
    )
    denominator = max(0.1, 2.2 + winner_rating_advantage * 0.001)
    multiplier = math.log(goal_difference + 1) * (2.2 / denominator)

    return max(MIN_GOAL_DIFFERENCE_MULTIPLIER, multiplier)


def update_elo(
    team_rating: float,
    expected: float,
    actual: float,
    k: int | float,
) -> float:
    """Apply one Elo update."""
    return team_rating + float(k) * (actual - expected)


def parse_neutral_flag(value: Any) -> bool:
    """Normalize neutral-venue values from CSV, notebooks, or APIs."""
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    if isinstance(value, (int, float)):
        return bool(value)

    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "t", "yes", "y"}


def parse_match_date(value: Any) -> datetime | None:
    """Parse supported match date formats for chronological processing."""
    if value in (None, ""):
        return None

    if isinstance(value, datetime):
        return value

    text = str(value).strip()
    for date_format in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, date_format)
        except ValueError:
            continue

    return None


@dataclass
class EloRatingEngine:
    """Stateful Elo engine that can process matches chronologically."""

    initial_rating: float = DEFAULT_RATING
    home_advantage: float = DEFAULT_HOME_ADVANTAGE
    use_goal_difference: bool = True
    ratings: dict[str, float] = field(default_factory=dict)
    history: list[RatingHistoryEntry] = field(default_factory=list)
    match_count: int = 0

    def initialize_team(self, team: str) -> float:
        return initialize_team(self.ratings, team, self.initial_rating)

    def process_match(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        tournament: str | None = None,
        neutral: bool | str | int | None = False,
        date: str | None = None,
    ) -> dict[str, float | bool]:
        """Update ratings for one completed match and return match details."""
        neutral_flag = parse_neutral_flag(neutral)
        home_rating = self.initialize_team(home_team)
        away_rating = self.initialize_team(away_team)
        adjusted_home_rating = apply_home_advantage(
            home_rating=home_rating,
            neutral=neutral_flag,
            home_advantage=self.home_advantage,
        )

        home_expected = expected_score(adjusted_home_rating, away_rating)
        away_expected = 1 - home_expected
        home_actual, away_actual = actual_score(home_score, away_score)
        k_factor = get_k_factor(tournament)
        margin_multiplier = (
            goal_difference_multiplier(
                home_score=home_score,
                away_score=away_score,
                home_rating=adjusted_home_rating,
                away_rating=away_rating,
            )
            if self.use_goal_difference
            else 1.0
        )
        effective_k = k_factor * margin_multiplier

        home_new_rating = update_elo(
            home_rating,
            home_expected,
            home_actual,
            effective_k,
        )
        away_new_rating = update_elo(
            away_rating,
            away_expected,
            away_actual,
            effective_k,
        )

        self.ratings[home_team] = home_new_rating
        self.ratings[away_team] = away_new_rating
        self.match_count += 1

        self.history.extend(
            [
                RatingHistoryEntry(
                    match_index=self.match_count,
                    date=date,
                    team=home_team,
                    opponent=away_team,
                    side="home",
                    tournament=tournament,
                    neutral=neutral_flag,
                    rating_before=home_rating,
                    rating_after=home_new_rating,
                    opponent_rating_before=away_rating,
                    expected_result=home_expected,
                    actual_result=home_actual,
                    k_factor=float(k_factor),
                    goal_difference_multiplier=margin_multiplier,
                ),
                RatingHistoryEntry(
                    match_index=self.match_count,
                    date=date,
                    team=away_team,
                    opponent=home_team,
                    side="away",
                    tournament=tournament,
                    neutral=neutral_flag,
                    rating_before=away_rating,
                    rating_after=away_new_rating,
                    opponent_rating_before=home_rating,
                    expected_result=away_expected,
                    actual_result=away_actual,
                    k_factor=float(k_factor),
                    goal_difference_multiplier=margin_multiplier,
                ),
            ]
        )

        return {
            "home_rating_before": home_rating,
            "away_rating_before": away_rating,
            "adjusted_home_rating": adjusted_home_rating,
            "home_expected": home_expected,
            "away_expected": away_expected,
            "home_actual": home_actual,
            "away_actual": away_actual,
            "k_factor": float(k_factor),
            "effective_k": float(effective_k),
            "goal_difference_multiplier": margin_multiplier,
            "neutral": neutral_flag,
            "home_rating_after": home_new_rating,
            "away_rating_after": away_new_rating,
        }

    def process_matches(
        self,
        matches: Iterable[Mapping[str, Any]],
        sort_by_date: bool = True,
    ) -> dict[str, float]:
        """Process iterable match rows with home/away teams, scores, tournament."""
        match_rows = list(matches)
        if sort_by_date:
            match_rows = [
                row
                for _, row in sorted(
                    enumerate(match_rows),
                    key=lambda item: (
                        parse_match_date(item[1].get("date")) is None,
                        parse_match_date(item[1].get("date")) or datetime.min,
                        item[0],
                    ),
                )
            ]

        for match in match_rows:
            self.process_match(
                home_team=str(match["home_team"]),
                away_team=str(match["away_team"]),
                home_score=int(match["home_score"]),
                away_score=int(match["away_score"]),
                tournament=match.get("tournament"),
                neutral=match.get("neutral", False),
                date=str(match["date"]) if match.get("date") is not None else None,
            )

        return self.ratings

    def get_team_rating(self, team: str) -> float:
        """Return a team's current rating, or the initial rating if unseen."""
        return self.ratings.get(team, self.initial_rating)

    def get_rankings(self, limit: int | None = None) -> list[dict[str, float | int | str]]:
        """Return current Elo rankings sorted from strongest to weakest."""
        ranked_teams = sorted(self.ratings.items(), key=lambda item: item[1], reverse=True)
        if limit is not None:
            ranked_teams = ranked_teams[:limit]

        return [
            {"rank": rank, "team": team, "rating": rating}
            for rank, (team, rating) in enumerate(ranked_teams, start=1)
        ]

    def get_rating_history(
        self,
        team: str | None = None,
    ) -> list[dict[str, float | int | str | bool | None]]:
        """Return rating history for all teams or one selected team."""
        entries = self.history
        if team is not None:
            entries = [entry for entry in entries if entry.team == team]

        return [entry.to_dict() for entry in entries]
