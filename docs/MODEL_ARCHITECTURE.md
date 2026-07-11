# Model Architecture

## Version 1 Foundation

The first production model layer is an advanced football Elo engine. It is the
strength model that later machine learning pipelines use as a feature source.

## Elo Engine

Implementation:

- `src/worldcup_intelligence/elo.py`

Core API:

- `initialize_team()`
- `expected_score()`
- `expected_scores()`
- `actual_score()`
- `update_elo()`
- `goal_difference_multiplier()`
- `EloRatingEngine`

## Rating Logic

Each team starts from `DEFAULT_RATING = 1500`.

For every completed match, the engine:

1. Initializes unseen teams.
2. Applies home advantage when the match is not neutral.
3. Computes home and away expected results.
4. Converts the final score into actual Elo results.
5. Looks up tournament importance through `config/k_factors.py`.
6. Applies a conservative goal-difference multiplier.
7. Updates both teams.
8. Stores one rating-history entry per team.

## Home Advantage

Home advantage is represented as a rating-point adjustment applied only to the
home team for the expectation calculation.

Neutral matches do not receive this adjustment.

Default:

- `DEFAULT_HOME_ADVANTAGE = 75`

## Tournament Importance

Tournament K-factors live in:

- `config/k_factors.py`

Unknown tournaments fall back to `DEFAULT_K_FACTOR = 20`, equivalent to a
friendly match.

## Goal Difference

One-goal wins and draws use a multiplier of `1.0`.

Larger wins receive a logarithmic multiplier. The multiplier is dampened when a
strong favorite wins heavily and amplified when an underdog wins heavily.

This keeps rating movement realistic for football while still rewarding
decisive results.

## Rating History

`EloRatingEngine.history` stores one `RatingHistoryEntry` per team per match.

Each entry includes:

- match index
- date
- team
- opponent
- side
- tournament
- neutral venue flag
- rating before
- rating after
- expected result
- actual result
- K-factor
- goal-difference multiplier
- rating change

The public method `get_rating_history(team=None)` returns history as dictionaries
for downstream feature engineering, analytics, and dashboard charts.

## Rankings

`EloRatingEngine.get_rankings(limit=None)` returns sorted team rankings:

- rank
- team
- rating

This ranking API is used by the prediction service and dashboard.

## Data Leakage Policy

The Elo engine processes matches chronologically. When rows include a `date`
field, `process_matches()` sorts by date before updating ratings.

Future feature pipelines should use pre-match ratings and historical features
only. Post-match rating values must never be used as features for that same
match.
