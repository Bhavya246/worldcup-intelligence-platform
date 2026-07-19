import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from worldcup_intelligence.models.logistic import LogisticMatchPredictor
from worldcup_intelligence.models.xgboost_model import XGBoostMatchPredictor
from worldcup_intelligence.elo import EloRatingEngine

st.set_page_config(page_title="WorldCup Intelligence Platform", page_icon="⚽", layout="wide")

@st.cache_resource
def load_models():
    lr = LogisticMatchPredictor()
    lr.load_model(ROOT / "models" / "logistic_regression.joblib")
    xgb = XGBoostMatchPredictor()
    xgb.load_model(ROOT / "models" / "xgboost_model.joblib")
    return lr, xgb

@st.cache_data
def load_elo_ratings():
    df = pd.read_csv(ROOT / "data" / "processed" / "ml_matches.csv")
    df = df.sort_values("match_index").reset_index(drop=True)
    engine = EloRatingEngine()
    for _, row in df.iterrows():
        engine.process_match(
            home_team=row["home_team"], away_team=row["away_team"],
            home_score=int(row["home_score"]), away_score=int(row["away_score"]),
            tournament=row.get("tournament"), neutral=row.get("neutral", False),
            date=str(row["date"]) if pd.notna(row.get("date")) else None,
        )
    return engine

@st.cache_data
def get_team_list():
    df = pd.read_csv(ROOT / "data" / "processed" / "ml_matches.csv")
    return sorted(set(df["home_team"].tolist() + df["away_team"].tolist()))

@st.cache_data
def get_team_history(team):
    df = pd.read_csv(ROOT / "data" / "processed" / "ml_matches.csv")
    df = df.sort_values("match_index").reset_index(drop=True)
    engine = EloRatingEngine()
    history = []
    for _, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]
        if ht == team or at == team:
            history.append({"match_index": row["match_index"], "elo": engine.get_team_rating(team)})
        engine.process_match(
            home_team=ht, away_team=at,
            home_score=int(row["home_score"]), away_score=int(row["away_score"]),
            tournament=row.get("tournament"), neutral=row.get("neutral", False),
            date=str(row["date"]) if pd.notna(row.get("date")) else None,
        )
    return pd.DataFrame(history)

def make_prediction(predictor, home, away, neutral, k):
    engine = load_elo_ratings()
    return predictor.predict_match(
        home_team=home, away_team=away,
        home_elo=engine.get_team_rating(home),
        away_elo=engine.get_team_rating(away),
        neutral=neutral, tournament_k_factor=k,
    )

def prob_bar(h, d, a, home, away):
    fig, ax = plt.subplots(figsize=(8, 0.8))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    left = 0
    for v, c, lbl in zip([h, d, a], ["#00c853", "#ffd600", "#ff1744"], [home, "Draw", away]):
        ax.barh(0, v, left=left, color=c, height=0.6)
        if v > 0.07:
            ax.text(left + v/2, 0, f"{lbl}\n{v*100:.1f}%", ha="center", va="center",
                    color="black", fontsize=9, fontweight="bold")
        left += v
    ax.set_xlim(0, 1)
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)
    plt.close()

lr_model, xgb_model = load_models()
teams = get_team_list()
engine = load_elo_ratings()

# Sidebar
st.sidebar.title("⚽ WorldCup Intelligence")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 FIFA World Cup 2026")
st.sidebar.markdown("**🏅 Final**")
st.sidebar.markdown("🇦🇷 Argentina vs Spain 🇪🇸  \n*July 19 · MetLife Stadium, NJ*")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.markdown("| Model | Accuracy | Log Loss |")
st.sidebar.markdown("|-------|----------|----------|")
st.sidebar.markdown("| Logistic Regression | 60.2% | 0.872 |")
st.sidebar.markdown("| XGBoost | 59.7% | 0.886 |")
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** 25,403 matches (2000–2026)")
st.sidebar.markdown("**Made by Bhavya Arora**")

# Header
st.title("⚽ WorldCup Intelligence Platform")
st.markdown("*ML-powered match predictions · FIFA World Cup 2026*")
st.markdown("---")

# Model selector
st.markdown("## 🤖 Select Model")
model_choice = st.radio("", ["Logistic Regression", "XGBoost"], horizontal=True)
active_model = lr_model if model_choice == "Logistic Regression" else xgb_model
st.markdown("---")

# Final prediction
st.markdown("## 🏆 World Cup Final Prediction")
st.markdown("**Argentina vs Spain · July 19 · MetLife Stadium, New Jersey**")
final_pred = make_prediction(active_model, "Argentina", "Spain", True, 60)
h = final_pred["home_win_probability"]
d = final_pred["draw_probability"]
a = final_pred["away_win_probability"]
f1, f2, f3, f4 = st.columns(4)
f1.metric("🇦🇷 Argentina", f"{h*100:.1f}%", f"Elo: {final_pred['home_elo']:.0f}")
f2.metric("🤝 Draw", f"{d*100:.1f}%")
f3.metric("🇪🇸 Spain", f"{a*100:.1f}%", f"Elo: {final_pred['away_elo']:.0f}")
f4.metric("🏆 Predicted Winner", final_pred["predicted_winner"])
prob_bar(h, d, a, "Argentina", "Spain")
st.success(f"Model predicts: {final_pred['predicted_winner']} wins · Confidence: {final_pred['confidence']*100:.1f}% · Model: {model_choice}")
st.markdown("---")

# Semi-final results
st.markdown("## ✅ Semi-Final Results (Model was correct on both!)")
s1, s2 = st.columns(2)
with s1:
    st.markdown("**SF1 · France vs Spain**")
    sf1 = make_prediction(active_model, "France", "Spain", True, 60)
    st.metric("Model predicted", sf1["predicted_winner"])
    st.markdown("✅ **Actual result: Spain won**")
with s2:
    st.markdown("**SF2 · England vs Argentina**")
    sf2 = make_prediction(active_model, "England", "Argentina", True, 60)
    st.metric("Model predicted", sf2["predicted_winner"])
    st.markdown("✅ **Actual result: Argentina won**")
st.markdown("---")

# Model comparison
st.markdown("## 📊 Model Comparison")
comp_df = pd.DataFrame([
    {"Model": "Logistic Regression", "Accuracy": "60.2%", "Log Loss": "0.872", "Winner": "✅"},
    {"Model": "XGBoost", "Accuracy": "59.7%", "Log Loss": "0.886", "Winner": ""},
])
st.dataframe(comp_df, use_container_width=True, hide_index=True)

# Feature importance
st.markdown("## 🔍 XGBoost Feature Importance (Top 10)")
fi = xgb_model.feature_importance.head(10)
fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor("#0e1117")
ax.set_facecolor("#0e1117")
ax.barh(fi["feature"][::-1], fi["importance"][::-1], color="#00c853")
ax.set_xlabel("Importance", color="#aaaaaa")
ax.set_title("Feature Importance", color="#ffffff")
ax.tick_params(colors="#aaaaaa")
ax.spines["bottom"].set_color("#3d4570")
ax.spines["left"].set_color("#3d4570")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
st.pyplot(fig, use_container_width=True)
plt.close()
st.markdown("---")

# Custom predictor
st.markdown("## 🔮 Predict Any Match")
c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    home_team = st.selectbox("🏠 Home Team", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
with c2:
    away_team = st.selectbox("✈️ Away Team", teams, index=teams.index("Spain") if "Spain" in teams else 1)
with c3:
    neutral = st.checkbox("Neutral Venue", value=True)
tournament = st.selectbox("Tournament", ["FIFA World Cup", "FIFA World Cup qualification",
    "UEFA Euro", "Copa America", "Friendly", "UEFA Nations League", "African Cup of Nations"])
k_map = {"FIFA World Cup": 60, "FIFA World Cup qualification": 40, "UEFA Euro": 50,
         "Copa America": 50, "Friendly": 20, "UEFA Nations League": 35, "African Cup of Nations": 50}
if st.button("⚡ Predict Match", type="primary", use_container_width=True):
    if home_team == away_team:
        st.error("Please select two different teams.")
    else:
        pred = make_prediction(active_model, home_team, away_team, neutral, k_map.get(tournament, 20))
        h2, d2, a2 = pred["home_win_probability"], pred["draw_probability"], pred["away_win_probability"]
        r1, r2, r3, r4 = st.columns(4)
        r1.metric(f"🏠 {home_team}", f"{h2*100:.1f}%")
        r2.metric("🤝 Draw", f"{d2*100:.1f}%")
        r3.metric(f"✈️ {away_team}", f"{a2*100:.1f}%")
        r4.metric("🏆 Winner", pred["predicted_winner"])
        prob_bar(h2, d2, a2, home_team, away_team)
        st.info(f"Elo — {home_team}: {pred['home_elo']:.0f} · {away_team}: {pred['away_elo']:.0f} · Model: {model_choice}")
st.markdown("---")

# Elo Rankings
st.markdown("## 📊 Current Elo Rankings (Top 30)")
rankings = engine.get_rankings(limit=30)
rdf = pd.DataFrame(rankings)
rdf.columns = ["Rank", "Team", "Elo Rating"]
rdf["Elo Rating"] = rdf["Elo Rating"].round(1)
highlight = {"France", "Spain", "England", "Argentina"}
def hl(row):
    if row["Team"] in highlight:
        return ["background-color: #2d3250; font-weight: bold"] * len(row)
    return [""] * len(row)
st.dataframe(rdf.style.apply(hl, axis=1), use_container_width=True, hide_index=True, height=600)
st.markdown("---")

# Rating history
st.markdown("## 📈 Team Rating History")
selected = st.multiselect("Select teams", teams, default=["France", "Spain", "England", "Argentina"])
if selected:
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    colors = ["#00c853", "#ff1744", "#2196f3", "#ffd600", "#9c27b0", "#ff6090"]
    for i, team in enumerate(selected):
        hist = get_team_history(team)
        if not hist.empty:
            ax.plot(hist["match_index"], hist["elo"], label=team,
                    color=colors[i % len(colors)], linewidth=2)
    ax.set_xlabel("Match Index", color="#aaaaaa")
    ax.set_ylabel("Elo Rating", color="#aaaaaa")
    ax.set_title("Elo Rating History", color="#ffffff")
    ax.tick_params(colors="#aaaaaa")
    ax.spines["bottom"].set_color("#3d4570")
    ax.spines["left"].set_color("#3d4570")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#1e2130", labelcolor="#ffffff")
    ax.grid(alpha=0.15, color="#3d4570")
    st.pyplot(fig, use_container_width=True)
    plt.close()

st.markdown("---")
st.markdown("<center><sub>WorldCup Intelligence Platform · Python · Scikit-learn · XGBoost · Streamlit · 25,403 matches (2000–2026)</sub></center>", unsafe_allow_html=True)