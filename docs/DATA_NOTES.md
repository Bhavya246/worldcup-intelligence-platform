# Data Notes

## Questions

1. What is a competition?
2. What is a match?
3. What is a team?
4. How is the winner determined?
5. Which fields may become features?

## Observations

(To be filled during exploration)

## Discoveries

- FIFA World Cup competition_id = 43
- Women's World Cup competition_id = 72
- season_id identifies tournament editions
- StatsBomb contains selected World Cup editions, not all editions

## Matches

2022 FIFA World Cup

- 64 matches
- 55 columns
- Each row represents one match

Potential target variables:

- home_score
- away_score
- winner (derived)

## Feature Engineering Thoughts

Why Elo Ratings?

- Machine learning models work better with numerical features than team names.
- Elo ratings summarize team strength.
- Elo incorporates historical match performance.
- Elo difference may become one of the most important features in the model.

## Candidate Features

### Team Strength
- Elo Rating
- Elo Difference
- FIFA Ranking

### Recent Form
- Last 5 Matches
- Last 10 Matches
- Win Percentage

### Tournament Context
- Competition Stage
- Group Position

### Team Information
- Manager
- Starting XI
- Substitutes

### Environment
- Venue
- Weather

### Team Exploration

- Unique Home Teams: 32
- Unique Away Teams: 32
- All World Cup teams are represented

## First Engineered Feature

Feature Name:
winner

Logic:

- Home score > Away score → Home Team
- Away score > Home score → Away Team
- Equal scores → Draw

Created using:
determine_winner(row)

Purpose:
Target variable for future prediction models.