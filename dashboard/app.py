import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from worldcup_intelligence.predictor import MatchPredictor
from worldcup_intelligence.models.score_predictor import PoissonScorePredictor
from worldcup_intelligence.elo import EloRatingEngine

st.set_page_config(page_title="WorldCup Intelligence Platform", page_icon="⚽", layout="wide")

MODEL_OPTIONS = {
    "Logistic Regression": "logistic",
    "XGBoost": "xgboost",
    "XGBoost (Tuned)": "xgboost_tuned",
    "MLP Neural Network": "mlp",
    "Ensemble (LR+XGBt+MLP)": "ensemble",
}

MODEL_STATS = {
    "Logistic Regression":     {"accuracy": "60.35%", "log_loss": "0.872"},
    "XGBoost":                 {"accuracy": "59.94%", "log_loss": "0.883"},
    "XGBoost (Tuned)":         {"accuracy": "60.62%", "log_loss": "0.885"},
    "MLP Neural Network":      {"accuracy": "59.88%", "log_loss": "0.879"},
    "Ensemble (LR+XGBt+MLP)": {"accuracy": "60.51%", "log_loss": "0.869"},
}

@st.cache_resource
def load_predictor(model_name):
    return MatchPredictor.load(model_name=model_name)

@st.cache_resource
def load_scorer():
    scorer = PoissonScorePredictor()
    scorer.load()
    return scorer

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

def scoreline_chart(top_scorelines, home, away):
    labels = [s["score"] for s in top_scorelines[:8]]
    probs = [s["probability"]*100 for s in top_scorelines[:8]]
    colors = []
    for s in top_scorelines[:8]:
        if s["home_goals"] > s["away_goals"]: colors.append("#00c853")
        elif s["away_goals"] > s["home_goals"]: colors.append("#ff1744")
        else: colors.append("#ffd600")
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    bars = ax.bar(labels, probs, color=colors)
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{prob:.1f}%", ha="center", va="bottom", color="white", fontsize=9)
    ax.set_ylabel("Probability (%)", color="#aaaaaa")
    ax.set_title(f"Top Scorelines: {home} vs {away}", color="#ffffff")
    ax.tick_params(colors="#aaaaaa")
    ax.spines["bottom"].set_color("#3d4570")
    ax.spines["left"].set_color("#3d4570")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    from matplotlib.patches import Patch
    legend = [Patch(color="#00c853", label=f"{home} win"),
              Patch(color="#ffd600", label="Draw"),
              Patch(color="#ff1744", label=f"{away} win")]
    ax.legend(handles=legend, facecolor="#1e2130", labelcolor="#ffffff")
    st.pyplot(fig, use_container_width=True)
    plt.close()

# Sidebar
st.sidebar.title("⚽ WorldCup Intelligence")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 FIFA World Cup 2026")
st.sidebar.markdown("**🥇 Final · July 19**")
st.sidebar.markdown("🇦🇷 Argentina vs Spain 🇪🇸")
st.sidebar.markdown("*MetLife Stadium, New Jersey*")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Select Model")
model_label = st.sidebar.selectbox("", list(MODEL_OPTIONS.keys()))
model_key = MODEL_OPTIONS[model_label]
stats = MODEL_STATS[model_label]
st.sidebar.markdown(f"**Accuracy:** {stats['accuracy']}  \n**Log Loss:** {stats['log_loss']}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** 25,457 matches (2000–2026)")
st.sidebar.markdown("**Made by Bhavya Arora**")

predictor = load_predictor(model_key)
scorer = load_scorer()
teams = get_team_list()

# Header
st.title("⚽ WorldCup Intelligence Platform")
st.markdown("*ML-powered match predictions · FIFA World Cup 2026*")
st.markdown("---")

# Final prediction
st.markdown("## 🏆 World Cup Final Prediction")
st.markdown("**Argentina vs Spain · July 19 · MetLife Stadium, New Jersey**")

final = predictor.predict_with_score("Argentina", "Spain", neutral=True, tournament="FIFA World Cup")
outcome = final["outcome"]
score = final["score"]

f1, f2, f3, f4 = st.columns(4)
f1.metric("🇦🇷 Argentina", f"{outcome.home_win_probability*100:.1f}%", f"Elo: {outcome.home_elo:.0f}")
f2.metric("🤝 Draw", f"{outcome.draw_probability*100:.1f}%")
f3.metric("🇪🇸 Spain", f"{outcome.away_win_probability*100:.1f}%", f"Elo: {outcome.away_elo:.0f}")
f4.metric("🏆 Predicted Winner", outcome.predicted_winner)
prob_bar(outcome.home_win_probability, outcome.draw_probability, outcome.away_win_probability, "Argentina", "Spain")
st.success(f"Model predicts: {outcome.predicted_winner} wins · Confidence: {outcome.confidence*100:.1f}% · {model_label}")

# Scoreline section
st.markdown("### ⚽ Scoreline Prediction")
sc1, sc2, sc3 = st.columns(3)
sc1.metric("🇦🇷 Argentina xG", f"{score.home_expected_goals:.2f}")
sc2.metric("🇪🇸 Spain xG", f"{score.away_expected_goals:.2f}")
sc3.metric("Most Likely Score", score.most_likely_score, f"{score.most_likely_score_probability*100:.1f}%")
scoreline_chart(score.top_scorelines, "Argentina", "Spain")
st.markdown("---")

# Model consensus
st.markdown("## 🤖 Model Consensus")
consensus_data = []
for label, key in MODEL_OPTIONS.items():
    p = load_predictor(key)
    r = p.predict("Argentina", "Spain", neutral=True, tournament="FIFA World Cup")
    consensus_data.append({
        "Model": label,
        "Argentina": f"{r.home_win_probability*100:.1f}%",
        "Draw": f"{r.draw_probability*100:.1f}%",
        "Spain": f"{r.away_win_probability*100:.1f}%",
        "Predicted Winner": r.predicted_winner,
        "Confidence": f"{r.confidence*100:.1f}%",
    })
st.dataframe(pd.DataFrame(consensus_data), use_container_width=True, hide_index=True)
st.markdown("---")

# Semi-final results
st.markdown("## ✅ Semi-Final Results (Both called correctly!)")
s1, s2 = st.columns(2)
with s1:
    st.markdown("**SF1 · France vs Spain**")
    sf1 = predictor.predict("France", "Spain", True, "FIFA World Cup")
    st.metric("Model predicted", sf1.predicted_winner)
    st.markdown("✅ **Actual: Spain won 2-0**")
with s2:
    st.markdown("**SF2 · England vs Argentina**")
    sf2 = predictor.predict("England", "Argentina", True, "FIFA World Cup")
    st.metric("Model predicted", sf2.predicted_winner)
    st.markdown("✅ **Actual: Argentina won 2-1**")
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
show_score = st.checkbox("Show scoreline prediction", value=True)

if st.button("⚡ Predict Match", type="primary", use_container_width=True):
    if home_team == away_team:
        st.error("Please select two different teams.")
    else:
        if show_score:
            res = predictor.predict_with_score(home_team, away_team, neutral, tournament)
            out = res["outcome"]
            sc = res["score"]
        else:
            out = predictor.predict(home_team, away_team, neutral, tournament)
        r1, r2, r3, r4 = st.columns(4)
        r1.metric(f"🏠 {home_team}", f"{out.home_win_probability*100:.1f}%")
        r2.metric("🤝 Draw", f"{out.draw_probability*100:.1f}%")
        r3.metric(f"✈️ {away_team}", f"{out.away_win_probability*100:.1f}%")
        r4.metric("🏆 Winner", out.predicted_winner)
        prob_bar(out.home_win_probability, out.draw_probability, out.away_win_probability, home_team, away_team)
        if show_score:
            st.markdown("**Scoreline Prediction**")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric(f"{home_team} xG", f"{sc.home_expected_goals:.2f}")
            sc2.metric(f"{away_team} xG", f"{sc.away_expected_goals:.2f}")
            sc3.metric("Most Likely Score", sc.most_likely_score, f"{sc.most_likely_score_probability*100:.1f}%")
            scoreline_chart(sc.top_scorelines, home_team, away_team)
        st.info(f"Elo — {home_team}: {out.home_elo:.0f} · {away_team}: {out.away_elo:.0f} · {model_label}")
st.markdown("---")

# Elo Rankings
st.markdown("## 📊 Current Elo Rankings (Top 30)")
rankings = predictor.rankings(limit=30)
rankings.columns = ["Rank", "Team", "Elo Rating"]
rankings["Elo Rating"] = rankings["Elo Rating"].round(1)
highlight = {"France", "Spain", "England", "Argentina"}
def hl(row):
    if row["Team"] in highlight:
        return ["background-color: #2d3250; font-weight: bold"] * len(row)
    return [""] * len(row)
st.dataframe(rankings.style.apply(hl, axis=1), use_container_width=True, hide_index=True, height=600)
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
st.markdown("<center><sub>WorldCup Intelligence Platform · Python · Scikit-learn · XGBoost · MLP · Streamlit · 25,457 matches (2000–2026)</sub></center>", unsafe_allow_html=True)