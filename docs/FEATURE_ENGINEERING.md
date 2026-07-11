# Feature Engineering

## Goal

Build reusable, leak-free features for international football match prediction.

The feature pipeline converts cleaned historical matches into an ML-ready CSV
where each row represents one match and every model feature is known before the
match is played.

## Implementation

Main module:

- `src/worldcup_intelligence/features/builder.py`

Public API:

- `build_feature_rows()`
- `build_feature_dataset()`
- `write_feature_dataset()`
- `result_label()`

## Generated Dataset

Input:

- `data/processed/clean_matches.csv`

Output:

- `data/processed/ml_matches.csv`

Generation command:

```bash
PYTHONPATH=src python -m worldcup_intelligence.features.builder \
  --input data/processed/clean_matches.csv \
  --output data/processed/ml_matches.csv \
  --rolling-window 5
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m worldcup_intelligence.features.builder `
  --input data\processed\clean_matches.csv `
  --output data\processed\ml_matches.csv `
  --rolling-window 5
```

The dataset is generated from code and may be ignored by Git depending on local
data tracking policy.

## Feature Groups

### Elo Features

- `home_elo`
- `away_elo`
- `elo_diff`
- `home_expected_elo`
- `away_expected_elo`

These are pre-match ratings and probabilities from the Elo engine.

### Venue Features

- `neutral`
- `home_advantage_applied`

Neutral venues receive no home-advantage adjustment.

### Tournament Feature

- `tournament_k_factor`

This comes from `config/k_factors.py` and represents match importance.

### Experience Features

- `home_matches_played`
- `away_matches_played`
- `matches_played_diff`

These count prior matches seen by the pipeline, not including the current match.

### Recent Form Features

Computed over the previous `N` matches for each team:

- `home_recent_points_per_match`
- `away_recent_points_per_match`
- `recent_points_diff`
- `home_recent_goals_for`
- `away_recent_goals_for`
- `recent_goals_for_diff`
- `home_recent_goals_against`
- `away_recent_goals_against`
- `recent_goals_against_diff`
- `home_recent_goal_difference`
- `away_recent_goal_difference`
- `recent_goal_difference_diff`

Default rolling window:

- `5` matches

### Rest Features

- `home_rest_days`
- `away_rest_days`
- `rest_days_diff`

Rest days are calculated from each team's previous match date.

### Head-To-Head Features

- `head_to_head_matches`
- `home_head_to_head_points_per_match`
- `away_head_to_head_points_per_match`
- `head_to_head_points_diff`

These are calculated before the current match updates the head-to-head state.

## Targets

Target columns:

- `target`
- `target_code`

Labels:

- `away_win` -> `0`
- `draw` -> `1`
- `home_win` -> `2`

## Leakage Controls

The feature builder sorts matches chronologically before processing.

For each match, it:

1. Reads current pre-match Elo ratings.
2. Reads rolling form from previous matches only.
3. Reads head-to-head history from previous meetings only.
4. Writes the feature row and target.
5. Updates Elo, team form, rest-day state, and head-to-head state.

This ensures current-match scores and post-match ratings are never used as input
features for the same match.

## Model Feature Columns

`FEATURE_COLUMNS` in `builder.py` is the canonical list of columns that model
training should use.

Metadata columns such as team names, dates, scores, and targets should be used
for auditing and splitting, not as model inputs.
