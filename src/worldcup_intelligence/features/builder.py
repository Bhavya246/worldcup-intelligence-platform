"""Leak-free feature engineering for international match prediction."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from config.k_factors import get_k_factor
from worldcup_intelligence.elo import (
    EloRatingEngine,
    expected_scores,
    parse_match_date,
    parse_neutral_flag,
)

RESULT_TO_CODE = {
    "away_win": 0,
    "draw": 1,
    "home_win": 2,
}

FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_expected_elo",
    "away_expected_elo",
    "neutral",
    "home_advantage_applied",
    "tournament_k_factor",
    "home_matches_played",
    "away_matches_played",
    "matches_played_diff",
    "home_recent_points_per_match",
    "away_recent_points_per_match",
    "recent_points_diff",
    "home_recent_goals_for",
    "away_recent_goals_for",
    "recent_goals_for_diff",
    "home_recent_goals_against",
    "away_recent_goals_against",
    "recent_goals_against_diff",
    "home_recent_goal_difference",
    "away_recent_goal_difference",
    "recent_goal_difference_diff",
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
    "head_to_head_matches",
    "home_head_to_head_points_per_match",
    "away_head_to_head_points_per_match",
    "head_to_head_points_diff",
]

TARGET_COLUMNS = [
    "target",
    "target_code",
]


@dataclass(frozen=True)
class TeamMatchSummary:
    date: datetime | None
    goals_for: int
    goals_against: int
    points: int

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


@dataclass
class TeamFormState:
    window: int = 5
    records: dict[str, deque[TeamMatchSummary]] = field(
        default_factory=lambda: defaultdict(deque)
    )
    matches_played: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_match_date: dict[str, datetime] = field(default_factory=dict)

    def summarize(self, team: str, match_date: datetime | None) -> dict[str, float | int]:
        history = self.records[team]
        match_count = len(history)

        goals_for = sum(record.goals_for for record in history)
        goals_against = sum(record.goals_against for record in history)
        goal_difference = sum(record.goal_difference for record in history)
        points = sum(record.points for record in history)

        rest_days = 0
        previous_date = self.last_match_date.get(team)
        if previous_date is not None and match_date is not None:
            rest_days = max(0, (match_date - previous_date).days)

        denominator = match_count or 1

        return {
            "matches_played": self.matches_played[team],
            "recent_matches": match_count,
            "recent_points": points,
            "recent_points_per_match": points / denominator,
            "recent_goals_for": goals_for / denominator,
            "recent_goals_against": goals_against / denominator,
            "recent_goal_difference": goal_difference / denominator,
            "rest_days": rest_days,
        }

    def update(
        self,
        team: str,
        match_date: datetime | None,
        goals_for: int,
        goals_against: int,
    ) -> None:
        points = points_from_score(goals_for, goals_against)
        history = self.records[team]
        history.append(
            TeamMatchSummary(
                date=match_date,
                goals_for=goals_for,
                goals_against=goals_against,
                points=points,
            )
        )
        while len(history) > self.window:
            history.popleft()

        self.matches_played[team] += 1
        if match_date is not None:
            self.last_match_date[team] = match_date


@dataclass
class HeadToHeadState:
    window: int = 5
    records: dict[tuple[str, str], deque[dict[str, int]]] = field(
        default_factory=lambda: defaultdict(deque)
    )

    def summarize(self, home_team: str, away_team: str) -> dict[str, float | int]:
        key = self._key(home_team, away_team)
        history = self.records[key]
        match_count = len(history)
        denominator = match_count or 1
        home_points = sum(record.get(home_team, 0) for record in history)
        away_points = sum(record.get(away_team, 0) for record in history)

        return {
            "head_to_head_matches": match_count,
            "home_head_to_head_points_per_match": home_points / denominator,
            "away_head_to_head_points_per_match": away_points / denominator,
        }

    def update(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
    ) -> None:
        key = self._key(home_team, away_team)
        history = self.records[key]
        history.append(
            {
                home_team: points_from_score(home_score, away_score),
                away_team: points_from_score(away_score, home_score),
            }
        )
        while len(history) > self.window:
            history.popleft()

    @staticmethod
    def _key(team_a: str, team_b: str) -> tuple[str, str]:
        return tuple(sorted((team_a, team_b)))


def points_from_score(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3

    if goals_for == goals_against:
        return 1

    return 0


def result_label(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "home_win"

    if away_score > home_score:
        return "away_win"

    return "draw"


def sort_matches_chronologically(
    matches: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    rows = list(matches)
    return [
        row
        for _, row in sorted(
            enumerate(rows),
            key=lambda item: (
                parse_match_date(item[1].get("date")) is None,
                parse_match_date(item[1].get("date")) or datetime.min,
                item[0],
            ),
        )
    ]


def build_feature_rows(
    matches: Iterable[Mapping[str, Any]],
    rolling_window: int = 5,
    elo_engine: EloRatingEngine | None = None,
) -> list[dict[str, Any]]:
    """Build one leak-free feature row per match.

    Features are computed before the current match updates Elo, rolling form,
    rest-day, and head-to-head state.
    """
    engine = elo_engine or EloRatingEngine()
    form_state = TeamFormState(window=rolling_window)
    head_to_head_state = HeadToHeadState(window=rolling_window)
    feature_rows: list[dict[str, Any]] = []

    for match_index, match in enumerate(sort_matches_chronologically(matches), start=1):
        home_team = str(match["home_team"])
        away_team = str(match["away_team"])
        home_score = int(match["home_score"])
        away_score = int(match["away_score"])
        tournament = str(match.get("tournament") or "")
        date_text = str(match["date"]) if match.get("date") is not None else None
        match_date = parse_match_date(date_text)
        neutral = parse_neutral_flag(match.get("neutral", False))

        home_elo = engine.get_team_rating(home_team)
        away_elo = engine.get_team_rating(away_team)
        home_expected, away_expected = expected_scores(
            home_rating=home_elo,
            away_rating=away_elo,
            neutral=neutral,
            home_advantage=engine.home_advantage,
        )

        home_form = form_state.summarize(home_team, match_date)
        away_form = form_state.summarize(away_team, match_date)
        head_to_head = head_to_head_state.summarize(home_team, away_team)
        label = result_label(home_score, away_score)

        row = {
            "match_index": match_index,
            "date": date_text,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "tournament": tournament,
            "country": match.get("country"),
            "city": match.get("city"),
            "target": label,
            "target_code": RESULT_TO_CODE[label],
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "home_expected_elo": home_expected,
            "away_expected_elo": away_expected,
            "neutral": int(neutral),
            "home_advantage_applied": 0 if neutral else engine.home_advantage,
            "tournament_k_factor": get_k_factor(tournament),
            "home_matches_played": home_form["matches_played"],
            "away_matches_played": away_form["matches_played"],
            "matches_played_diff": home_form["matches_played"]
            - away_form["matches_played"],
            "home_recent_points_per_match": home_form["recent_points_per_match"],
            "away_recent_points_per_match": away_form["recent_points_per_match"],
            "recent_points_diff": home_form["recent_points_per_match"]
            - away_form["recent_points_per_match"],
            "home_recent_goals_for": home_form["recent_goals_for"],
            "away_recent_goals_for": away_form["recent_goals_for"],
            "recent_goals_for_diff": home_form["recent_goals_for"]
            - away_form["recent_goals_for"],
            "home_recent_goals_against": home_form["recent_goals_against"],
            "away_recent_goals_against": away_form["recent_goals_against"],
            "recent_goals_against_diff": home_form["recent_goals_against"]
            - away_form["recent_goals_against"],
            "home_recent_goal_difference": home_form["recent_goal_difference"],
            "away_recent_goal_difference": away_form["recent_goal_difference"],
            "recent_goal_difference_diff": home_form["recent_goal_difference"]
            - away_form["recent_goal_difference"],
            "home_rest_days": home_form["rest_days"],
            "away_rest_days": away_form["rest_days"],
            "rest_days_diff": home_form["rest_days"] - away_form["rest_days"],
            **head_to_head,
        }
        row["head_to_head_points_diff"] = (
            row["home_head_to_head_points_per_match"]
            - row["away_head_to_head_points_per_match"]
        )
        feature_rows.append(row)

        engine.process_match(
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            tournament=tournament,
            neutral=neutral,
            date=date_text,
        )
        form_state.update(home_team, match_date, home_score, away_score)
        form_state.update(away_team, match_date, away_score, home_score)
        head_to_head_state.update(home_team, away_team, home_score, away_score)

    return feature_rows


def write_feature_dataset(
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output.write_text("", encoding="utf-8")
        return output

    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output


def build_feature_dataset(
    input_path: str | Path,
    output_path: str | Path,
    rolling_window: int = 5,
) -> Path:
    with Path(input_path).open(encoding="utf-8", newline="") as file:
        rows = build_feature_rows(csv.DictReader(file), rolling_window=rolling_window)

    return write_feature_dataset(rows, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ML-ready match features.")
    parser.add_argument("--input", required=True, help="Path to cleaned match CSV.")
    parser.add_argument("--output", required=True, help="Path for feature CSV output.")
    parser.add_argument("--rolling-window", type=int, default=5)
    args = parser.parse_args()

    output_path = build_feature_dataset(
        input_path=args.input,
        output_path=args.output,
        rolling_window=args.rolling_window,
    )
    print(f"Wrote feature dataset to {output_path}")


if __name__ == "__main__":
    main()
