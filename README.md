# ⚽ WorldCup Intelligence Platform

A professional football analytics platform for predicting international match outcomes,
built with Python, Scikit-learn, XGBoost, and Streamlit.

Trained on **25,403 international matches (2000–2026)** and validated on the
**FIFA World Cup 2026** — correctly predicting both semi-finals.

---

## 🏆 FIFA World Cup 2026 Results

| Match | Model Prediction | Actual Result |
|-------|-----------------|---------------|
| SF1: France vs Spain | **Spain** (46.3%) | ✅ Spain won |
| SF2: England vs Argentina | **Argentina** (56.8%) | ✅ Argentina won |
| Final: Argentina vs Spain | **Argentina** (41.1%) | TBD |

---

## 🏗️ Architecture

The platform is built in three layers:

```
Layer 1: Data Pipeline
  Raw CSV → Validation → Feature Engineering → ML-ready Dataset

Layer 2: Elo Rating Engine
  Chronological processing → Tournament weighting → Home advantage → Goal difference

Layer 3: Machine Learning
  Logistic Regression + XGBoost → Calibration → Unified Prediction API
```

---

## 📁 Project Structure

```
worldcup-intelligence-platform/
├── config/
│   └── k_factors.py          # Tournament importance weights
├── dashboard/
│   └── app.py                # Streamlit dashboard
├── data/
│   ├── raw/                  # Raw match data
│   └── processed/
│       └── ml_matches.csv    # ML-ready feature dataset (25,403 rows)
├── models/
│   ├── logistic_regression.joblib
│   ├── xgboost_model.joblib
│   └── xgboost_calibrated.joblib
├── notebooks/
│   ├── 01_data_discovery.ipynb
│   ├── 02_elo_ratings.ipynb
│   ├── 03_data_validation.ipynb
│   ├── 04_building_worldcup_elo.ipynb
│   └── 05_model_training.ipynb
├── src/worldcup_intelligence/
│   ├── elo.py                # Elo rating engine
│   ├── predictor.py          # Unified prediction API
│   ├── features/
│   │   └── builder.py        # Leak-free feature engineering
│   └── models/
│       ├── logistic.py       # Logistic Regression model
│       ├── xgboost_model.py  # XGBoost model
│       ├── calibration.py    # Probability calibration
│       └── dataset.py        # Dataset utilities
└── tests/
    └── test_predictor.py     # 36 unit tests
```

---

## 🚀 Quick Start

### 1. Clone and install
```bash
git clone https://github.com/Bhavya246/worldcup-intelligence-platform
cd worldcup-intelligence-platform
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Run the dashboard
```bash
streamlit run dashboard/app.py
```

### 3. Use the prediction API
```python
from worldcup_intelligence.predictor import MatchPredictor

predictor = MatchPredictor.load(model_name="logistic")
result = predictor.predict("Argentina", "Spain", neutral=True, tournament="FIFA World Cup")
print(result)
# Match: Argentina vs Spain
#   Argentina win:  41.1%
#   Draw:           21.2%
#   Spain win:      37.6%
#   Predicted:      Argentina (confidence: 41.1%)
```

### 4. Run tests
```bash
pytest tests/ -v
```

---

## 📊 Model Performance

| Model | Accuracy | Log Loss | Notes |
|-------|----------|----------|-------|
| Logistic Regression | 60.37% | 0.872 | Strong baseline |
| XGBoost | 60.02% | 0.885 | Original |
| XGBoost (Tuned) | **60.43%** | 0.874 | Best accuracy |
| MLP Neural Network | 59.88% | 0.879 | Deep learning |
| Ensemble LR+XGBt+MLP | 60.43% | **0.868** | Best log loss |

> Baseline (random): ~33.3% accuracy
> Ensemble wins on Log Loss — best probability calibration
> XGBoost Tuned wins on Accuracy
> For a prediction platform, Log Loss is the more important metric

---

## 🔧 Feature Engineering

31 features engineered per match, all computed **before** the match
to prevent data leakage:

| Category | Features |
|----------|----------|
| Elo Ratings | home_elo, away_elo, elo_diff, expected scores |
| Venue | neutral, home_advantage_applied |
| Tournament | tournament_k_factor |
| Form (rolling 5) | points/match, goals for/against, goal difference |
| Rest | home_rest_days, away_rest_days |
| Head-to-Head | h2h matches, h2h points per match |

---

## ⚙️ Elo Rating Engine

Custom Elo engine with:
- **Tournament weighting** — World Cup (K=60) vs Friendly (K=20)
- **Home advantage** — 75 rating points by default
- **Goal difference multiplier** — logarithmic, rewards underdog wins more
- **Neutral venue handling** — removes home advantage at neutral grounds
- **Chronological processing** — no future data leakage

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| ML | Scikit-learn 1.9, XGBoost 3.3, MLP Neural Network |
| Data | Pandas, NumPy |
| Dashboard | Streamlit 1.59 |
| Serialization | Joblib |
| Testing | Pytest (36 tests) |
| Score Prediction | Poisson Regression (Dixon-Coles) |

---

## 👤 Author

**Bhavya Arora**
Associate Software Engineer · Accenture India
B.Tech Information Technology · Manipal University Jaipur (2024)

---

*Built during the FIFA World Cup 2026 — predictions posted live before each match.*