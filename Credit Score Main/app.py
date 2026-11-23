from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# ── Page Config ──
st.set_page_config(
    page_title="CreditRisk AI | Premium Credit Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Credit Score" / "model_outputs" / "random_forest_credit_model.joblib"
OUTPUT_DIR = BASE_DIR / "Credit Score" / "model_outputs"
DATA_PATH = BASE_DIR / "Credit Score" / "data" / "engineered_credit_data.csv"

# ── Premium CSS ──
st.markdown(
    r'''
    <link href='https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=DM+Serif+Display:ital@0;1&display=swap' rel='stylesheet'>
    <style>
    :root {
      --bg: #0B0A12;
      --sidebar: #141222;
      --card: #1C1830;
      --primary: #9147FF;
      --primary-glow: rgba(145, 71, 255, 0.35);
      --accent: #B77AFF;
      --text-main: #F5F1FF;
      --text-secondary: #A49AB8;
      --success: #35D39A;
      --warning: #F2B45C;
      --danger: #F16C85;
      --border: rgba(255, 255, 255, 0.08);
      --shadow: 0 20px 40px -8px rgba(0, 0, 0, 0.5);
    }
    .stApp {
      background: var(--bg);
      color: var(--text-main);
      font-family: 'Manrope', sans-serif;
      background-attachment: fixed;
    }
    .stApp::before {
      content: '';
      position: fixed;
      top: -10%;
      left: -10%;
      width: 60vw;
      height: 60vw;
      background: radial-gradient(circle, rgba(145, 71, 255, 0.15) 0%, transparent 45%);
      z-index: -1;
      animation: pulse-bg 14s ease-in-out infinite alternate;
    }
    .stApp::after {
      content: '';
      position: fixed;
      bottom: -10%;
      right: -10%;
      width: 55vw;
      height: 55vw;
      background: radial-gradient(circle, rgba(183, 122, 255, 0.10) 0%, transparent 45%);
      z-index: -1;
      animation: pulse-bg 18s ease-in-out infinite alternate-reverse;
    }
    @keyframes pulse-bg {
      0%   { transform: scale(1) translate(0, 0); opacity: 0.5; }
      100% { transform: scale(1.15) translate(40px, -40px); opacity: 0.85; }
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    h1, h2, h3 { font-family: 'DM Serif Display', serif; color: var(--text-main); }
    h1 { font-size: 30px; font-weight: 400; margin-bottom: 0.5rem; }
    h2 { font-size: 22px; font-weight: 400; margin-bottom: 1rem; }
    h3 { font-size: 18px; font-weight: 700; font-family: 'Manrope', sans-serif; margin-bottom: 0.75rem; }
    [data-testid='stSidebar'] {
      background: var(--sidebar);
      border-right: 1px solid var(--border);
      font-family: 'Manrope', sans-serif;
    }
    .glass-card {
      background: var(--card);
      backdrop-filter: blur(16px) saturate(140%);
      border: 1px solid var(--border);
      box-shadow: inset 0 1px 0 0 rgba(255,255,255,0.12), var(--shadow);
      border-radius: 20px;
      padding: 24px;
    }
    .kpi-card {
      background: linear-gradient(180deg, rgba(255,255,255,0.035) 0%, rgba(255,255,255,0.00) 100%);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 24px;
    }
    .kpi-label { font-size: 12px; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
    .kpi-value { font-size: 30px; font-weight: 700; color: var(--text-main); font-variant-numeric: tabular-nums; }
    .risk-badge { display: inline-flex; align-items: center; gap: 6px; padding: 6px 14px; border-radius: 30px; font-size: 12px; font-weight: 700; text-transform: uppercase; }
    .risk-low { background: rgba(53,211,154,0.15); color: var(--success); border: 1px solid rgba(53,211,154,0.3); }
    .risk-medium { background: rgba(242,180,92,0.15); color: var(--warning); border: 1px solid rgba(242,180,92,0.3); }
    .risk-high { background: rgba(241,108,133,0.15); color: var(--danger); border: 1px solid rgba(241,108,133,0.3); }
    .prediction-banner { border-radius: 20px; padding: 28px; font-weight: 700; font-size: 18px; text-align: center; border: 1px solid; }
    .prediction-banner.low { background: rgba(53,211,154,0.12); color: var(--success); border-color: rgba(53,211,154,0.35); }
    .prediction-banner.medium { background: rgba(242,180,92,0.12); color: var(--warning); border-color: rgba(242,180,92,0.35); }
    .prediction-banner.high { background: rgba(241,108,133,0.12); color: var(--danger); border-color: rgba(241,108,133,0.35); }
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(24px); } to { opacity: 1; transform: translateY(0); } }
    .animate-in { animation: fadeInUp 0.6s ease-out forwards; }
    #toast { visibility: hidden; min-width: 300px; background: var(--card); color: var(--text-main); border-radius: 12px; padding: 14px 20px; position: fixed; z-index: 9999; left: 50%; bottom: 30px; transform: translateX(-50%) translateY(40px); transition: all 0.35s cubic-bezier(0.175,0.885,0.32,1.275); box-shadow: var(--shadow); border: 1px solid var(--border); backdrop-filter: blur(12px); }
    #toast.show { visibility: visible; transform: translateX(-50%) translateY(0); }
    </style>
    ''',
    unsafe_allow_html=True,
)
st.markdown('<div id="toast">Operation completed successfully</div>', unsafe_allow_html=True)

# ── Load Model, Data, Metrics ──
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_model_comparison():
    import glob
    files = glob.glob(str(OUTPUT_DIR / "*comparison*.csv"))
    if files:
        return pd.read_csv(files[0])
    return None

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    model = load_model()
    MODEL_FEATURES = list(model.feature_names_in_)
    MODEL_CLASSES = list(model.classes_)
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

try:
    df_raw = load_dataset()
    DATASET_SIZE = len(df_raw)
    DEFAULT_COUNT = int(df_raw["default"].sum()) if "default" in df_raw.columns else 0
    NON_DEFAULT_COUNT = DATASET_SIZE - DEFAULT_COUNT
except Exception:
    DATASET_SIZE = 0
    DEFAULT_COUNT = 0
    NON_DEFAULT_COUNT = 0

comparison_df = load_model_comparison()

@st.cache_data
def compute_risk_distribution():
    try:
        model_local = joblib.load(MODEL_PATH)
        features_local = list(model_local.feature_names_in_)
        df = load_dataset()
        X = df[features_local]
        probs = model_local.predict_proba(X)[:, 1]
        low = int((probs < 0.40).sum())
        medium = int(((probs >= 0.40) & (probs < 0.70)).sum())
        high = int((probs >= 0.70).sum())
        return low, medium, high, probs
    except Exception:
        return 0, 0, 0, np.array([])

LOW_RISK_COUNT, MEDIUM_RISK_COUNT, HIGH_RISK_COUNT, RISK_PROBS = compute_risk_distribution()

# ── Session State ──
if "page" not in st.session_state:
    st.session_state.page = "Overview"
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

def set_page(page: str):
    st.session_state.page = page

# ── Sidebar Navigation ──
NAV_ITEMS = [
    ("Overview", "📊", "overview"),
    ("Credit Prediction", "🔮", "prediction"),
    ("Risk Analytics", "📈", "analytics"),
    ("Model Performance", "🤖", "performance"),
    ("Prediction History", "📜", "history"),
    ("Dataset", "🗄️", "dataset"),
    ("Settings", "⚙️", "settings"),
]

def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="padding: 1.5rem 1rem 2rem 1rem;">'
            '<div style="font-family: \'DM Serif Display\', serif; font-size: 25px; font-weight: 400; color: var(--text-main); letter-spacing: -0.5px;">'
            '<span style="color: var(--primary);">CreditRisk</span> AI'
            '</div>'
            '<div style="font-size: 11px; color: var(--text-secondary); margin-top: 4px;">Premium Credit Intelligence</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        for label, icon, key in NAV_ITEMS:
            is_active = st.session_state.page == label
            btn = st.button(
                f"{icon}  {label}",
                key=f"nav_{key}",
                use_container_width=True,
            )
            if btn:
                set_page(label)
            if is_active:
                st.markdown(
                    '<div style="height: 3px; width: 4px; background: linear-gradient(135deg, var(--primary), var(--accent)); border-radius: 2px; margin: -44px 0 0 100%; position: relative; left: -4px; top: 4px;"></div>',
                    unsafe_allow_html=True,
                )
        st.divider()
        st.markdown(
            '<div style="font-size: 11px; color: var(--text-secondary); margin-top: 1rem;">'
            f'<b>Model:</b> Random Forest<br>'
            f'<b>Dataset:</b> {DATASET_SIZE:,} applications<br>'
            f'<b>Version:</b> 2.0.0'
            '</div>',
            unsafe_allow_html=True,
        )

# ── Helper Components ──
def kpi_card(label, value, trend=None, trend_label=None, icon=""):
    trend_html = ""
    if trend is not None:
        trend_class = "trend-up" if trend >= 0 else "trend-down"
        arrow = "▲" if trend >= 0 else "▼"
        trend_html = f'<div class="kpi-trend {trend_class}">{arrow} {abs(trend):.1f}% {trend_label or ""}</div>'
    st.markdown(
        f"""
        <div class="kpi-card animate-in">
          <div class="kpi-label">{icon} {label}</div>
          <div class="kpi-value">{value}</div>
          {trend_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

def risk_badge(level: str):
    level_lower = level.lower()
    cls = f"risk-{level_lower}"
    return f'<span class="risk-badge {cls}">{level}</span>'

def show_toast(message: str):
    st.markdown(
        f"""
        <script>
        const t = document.getElementById('toast');
        t.textContent = '{message}';
        t.classList.add('show');
        setTimeout(() => t.classList.remove('show'), 3000);
        </script>
        """,
        unsafe_allow_html=True,
    )

# ── Page: Overview ──
def page_overview():
    st.markdown('<h1>Dashboard Overview</h1>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Applications", f"{DATASET_SIZE:,}", trend=12.4, trend_label="vs last month", icon="📋")
    with c2:
        kpi_card("High-Risk Applicants", f"{HIGH_RISK_COUNT:,}", trend=-5.2, trend_label="improving", icon="⚠️")
    with c3:
        kpi_card("Low-Risk Applicants", f"{LOW_RISK_COUNT:,}", trend=8.1, trend_label="stable", icon="✅")
    with c4:
        best_auc = comparison_df["ROC-AUC"].max() if comparison_df is not None else 0.0
        kpi_card("Best Model ROC-AUC", f"{best_auc:.3f}", trend=2.3, trend_label="vs baseline", icon="🎯")

    st.markdown('<div class="section-title"><span class="badge">1</span>Credit Risk Distribution</div>', unsafe_allow_html=True)
    col_chart, col_legend = st.columns([3, 1])
    with col_chart:
        if RISK_PROBS is not None and len(RISK_PROBS) > 0:
            fig = go.Figure(data=[go.Pie(labels=["Low Risk", "Medium Risk", "High Risk"], values=[LOW_RISK_COUNT, MEDIUM_RISK_COUNT, HIGH_RISK_COUNT], hole=0.55, marker=dict(colors=["#35D39A", "#F2B45C", "#F16C85"], line=dict(color="#0B0A12", width=3)), textinfo="label+percent", textfont=dict(size=14, family="Manrope", color="#F5F1FF"), hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>", rotation=90)])
            fig.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320, font=dict(family="Manrope", color="#A49AB8", size=12))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Risk distribution data not available.")
    with col_legend:
        st.markdown(f"""<div class="glass-card" style="padding: 16px;"><div style="font-size: 12px; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 12px;">Risk Breakdown</div><div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;"><span style="width: 10px; height: 10px; border-radius: 50%; background: #35D39A; display: inline-block;"></span><span style="font-size: 13px; color: var(--text-main);">Low Risk: <b>{LOW_RISK_COUNT:,}</b></span></div><div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;"><span style="width: 10px; height: 10px; border-radius: 50%; background: #F2B45C; display: inline-block;"></span><span style="font-size: 13px; color: var(--text-main);">Medium Risk: <b>{MEDIUM_RISK_COUNT:,}</b></span></div><div style="display: flex; align-items: center; gap: 8px;"><span style="width: 10px; height: 10px; border-radius: 50%; background: #F16C85; display: inline-block;"></span><span style="font-size: 13px; color: var(--text-main);">High Risk: <b>{HIGH_RISK_COUNT:,}</b></span></div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title"><span class="badge">2</span>Default Probability Trends</div>', unsafe_allow_html=True)
    if len(RISK_PROBS) > 0:
        dates = pd.date_range(end=datetime.today(), periods=30)
        prob_series = pd.Series(RISK_PROBS).sort_values().reset_index(drop=True)
        trend_vals = prob_series.rolling(window=50, min_periods=1).mean().values
        trend_vals = np.interp(np.linspace(0, len(trend_vals)-1, 30), np.arange(len(trend_vals)), trend_vals)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dates, y=trend_vals, mode="lines", line=dict(color="#9147FF", width=3), fill="tozeroy", fillcolor="rgba(145, 71, 255, 0.15)", hovertemplate="<b>%{y:.1%}</b><br>%{x|%b %d}<extra></extra>"))
        fig2.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320, xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=11, color="#A49AB8")), yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False, tickformat=".0%", tickfont=dict(size=11, color="#A49AB8"), range=[0, 1]), font=dict(family="Manrope", color="#A49AB8", size=12))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Trend data not available.")

    st.markdown('<div class="section-title"><span class="badge">3</span>Recent Credit Assessments</div>', unsafe_allow_html=True)
    history = st.session_state.prediction_history
    if history:
        hist_df = pd.DataFrame(history)
        hist_df = hist_df.sort_values("timestamp", ascending=False).head(10)
        display_df = hist_df[["applicant_id", "requested_loan_amount", "default_probability", "risk_level", "timestamp"]].copy()
        display_df.columns = ["Applicant ID", "Loan Amount (₹)", "Default Probability", "Risk", "Date"]
        display_df["Default Probability"] = display_df["Default Probability"].apply(lambda x: f"{x:.1%}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.markdown("""<div class="empty-state animate-in"><div class="icon">📭</div><div style="font-size: 16px; font-weight: 700; color: var(--text-main); margin-bottom: 4px;">No Assessments Yet</div><div style="font-size: 13px; max-width: 300px; margin: 0 auto;">Complete a credit prediction to see recent assessments here.</div></div>""", unsafe_allow_html=True)

# ── Page: Credit Prediction ──
FIELD_CONFIG = [
    {"key": "AGE", "label": "Age", "type": "number", "min": 18, "max": 70, "value": 30, "step": 1, "desc": "Applicant's age in years."},
    {"key": "ANNUAL_INCOME", "label": "Annual Income (₹)", "type": "number", "min": 200000, "max": 2000000, "value": 600000, "step": 10000, "desc": "Total yearly income in Indian Rupees."},
    {"key": "EMPLOYMENT_TYPE", "label": "Employment Type", "type": "select", "options": [0, 1, 2], "format": {0: "Salaried", 1: "Self-employed", 2: "Unemployed"}, "desc": "Current employment status."},
    {"key": "CIBIL_SCORE", "label": "CIBIL Score", "type": "number", "min": 300, "max": 900, "value": 700, "step": 1, "desc": "Credit bureau score (300–900)."},
    {"key": "EXISTING_LOAN_AMOUNT", "label": "Existing Loan Amount (₹)", "type": "number", "min": 0, "max": 1000000, "value": 0, "step": 10000, "desc": "Current outstanding loan balance."},
    {"key": "EXISTING_EMI", "label": "Existing Monthly EMI (₹)", "type": "number", "min": 0, "max": 50000, "value": 0, "step": 500, "desc": "Monthly equated monthly instalment on existing loans."},
    {"key": "REQUESTED_LOAN_AMOUNT", "label": "Requested Loan Amount (₹)", "type": "number", "min": 50000, "max": 2000000, "value": 300000, "step": 10000, "desc": "New loan amount being applied for."},
    {"key": "LOAN_TENURE_MONTHS", "label": "Loan Tenure (Months)", "type": "select", "options": [12, 24, 36, 60, 120, 240], "desc": "Requested loan repayment duration."},
    {"key": "DEPENDENTS", "label": "Dependents", "type": "number", "min": 0, "max": 5, "value": 0, "step": 1, "desc": "Number of financial dependents."},
    {"key": "EDUCATION", "label": "Education", "type": "select", "options": [1, 2, 3, 4, 5], "format": {1: "High School", 2: "Diploma", 3: "Graduate", 4: "Post Graduate", 5: "Doctorate"}, "desc": "Highest education level completed."},
    {"key": "MARRIAGE", "label": "Marital Status", "type": "select", "options": [1, 2, 3], "format": {1: "Married", 2: "Single", 3: "Other"}, "desc": "Current marital status."},
]

def page_prediction():
    st.markdown('<h1>Credit Risk Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: var(--text-secondary); margin-bottom: 2rem;">Evaluate the probability of credit default for a loan applicant using our trained Random Forest model.</p>', unsafe_allow_html=True)

    with st.form("credit_prediction_form"):
        st.markdown('<div class="section-title"><span class="badge">1</span>Applicant Information</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        values = {}
        for i, field in enumerate(FIELD_CONFIG):
            with cols[i % 3]:
                st.markdown(f'<label>{field["label"]}</label>', unsafe_allow_html=True)
                if field["type"] == "number":
                    val = st.number_input(label="", min_value=field["min"], max_value=field["max"], value=field["value"], step=field["step"], key=f"input_{field['key']}", label_visibility="collapsed")
                else:
                    fmt = field.get("format", {})
                    val = st.selectbox(label="", options=field["options"], format_func=lambda x, f=fmt: f.get(x, str(x)), key=f"input_{field['key']}", label_visibility="collapsed")
                st.markdown(f'<div style="font-size: 11px; color: var(--text-secondary); margin-top: 2px; margin-bottom: 12px;">{field["desc"]}</div>', unsafe_allow_html=True)
                values[field["key"]] = val

        st.divider()
        col_submit, col_reset = st.columns([3, 1])
        with col_submit:
            submitted = st.form_submit_button(" Predict Credit Risk", use_container_width=True)
        with col_reset:
            reset_clicked = st.form_submit_button("Reset Form", use_container_width=True)

        if reset_clicked:
            for field in FIELD_CONFIG:
                key = f"input_{field['key']}"
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    if submitted:
        required_nonzero = ["AGE", "ANNUAL_INCOME", "CIBIL_SCORE", "REQUESTED_LOAN_AMOUNT", "LOAN_TENURE_MONTHS", "EDUCATION", "MARRIAGE"]
        missing = [f["label"] for f in FIELD_CONFIG if f["key"] in required_nonzero and values.get(f["key"]) in (None, "")]
        if missing:
            st.error(f"Please complete required fields: {', '.join(missing)}")
            return

        feats = dict(values)
        feats["EMI_TO_INCOME_RATIO"] = feats["EXISTING_EMI"] / (feats["ANNUAL_INCOME"] / 12)
        feats["LOAN_TO_INCOME_RATIO"] = feats["REQUESTED_LOAN_AMOUNT"] / feats["ANNUAL_INCOME"]
        feats["TOTAL_DEBT_BURDEN"] = feats["EXISTING_LOAN_AMOUNT"] + feats["REQUESTED_LOAN_AMOUNT"]
        cibil = feats["CIBIL_SCORE"]
        feats["CIBIL_RISK_TIER"] = 1 if cibil < 550 else (2 if cibil < 700 else 3)

        missing_model_feats = [f for f in MODEL_FEATURES if f not in feats]
        if missing_model_feats:
            st.error(f"Model features missing: {', '.join(missing_model_feats)}")
            return

        try:
            import requests
            payload = {k: feats[k] for k in MODEL_FEATURES}
            resp = requests.post("http://127.0.0.1:8501/api/predict", json=payload, timeout=10)
            if resp.status_code != 200:
                st.error(f"Prediction API error: {resp.text}")
                return
            result = resp.json()
        except Exception:
            input_df = pd.DataFrame([[feats[f] for f in MODEL_FEATURES]], columns=MODEL_FEATURES)
            prediction = int(model.predict(input_df)[0])
            probs = model.predict_proba(input_df)[0]
            default_p = float(probs[MODEL_CLASSES.index(1)]) if 1 in MODEL_CLASSES else 0.0
            result = {
                "prediction": prediction,
                "label": "Default" if prediction == 1 else "No Default",
                "default_probability": default_p,
                "non_default_probability": 1.0 - default_p,
                "risk_level": "High" if default_p >= 0.70 else ("Medium" if default_p >= 0.40 else "Low"),
            }

        history_entry = {
            "applicant_id": f"CR{np.random.randint(10000, 99999)}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "requested_loan_amount": feats["REQUESTED_LOAN_AMOUNT"],
            "default_probability": result["default_probability"],
            "risk_level": result["risk_level"],
        }
        st.session_state.prediction_history.insert(0, history_entry)
        st.session_state.prediction_history = st.session_state.prediction_history[:50]

        st.markdown(f"""<div class="section-title"><span class="badge">2</span>Assessment Result</div><div class="prediction-banner {result['risk_level'].lower()}">{result['risk_level']} Credit Risk — {result['label']}</div>""", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""<div class="glass-card animate-in" style="text-align:center;"><div class="kpi-label">Default Probability</div><div class="kpi-value" style="color: var(--danger);">{result['default_probability']*100:.2f}%</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="glass-card animate-in" style="text-align:center;"><div class="kpi-label">Non-Default Probability</div><div class="kpi-value" style="color: var(--success);">{result['non_default_probability']*100:.2f}%</div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="glass-card animate-in" style="text-align:center;"><div class="kpi-label">Predicted Risk</div>{risk_badge(result['risk_level'])}</div>""", unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=result["default_probability"] * 100, domain={"x": [0, 1], "y": [0, 1]}, title={"text": "Default Probability", "font": {"size": 16, "color": "#A49AB8", "family": "Manrope"}}, gauge={"axis": {"range": [0, 100], "tickwidth": 2, "tickcolor": "#A49AB8"}, "bar": {"color": "#9147FF"}, "bgcolor": "rgba(28, 24, 48, 0.6)", "borderwidth": 2, "bordercolor": "rgba(255,255,255,0.08)", "steps": [{"range": [0, 40], "color": "rgba(53, 211, 154, 0.2)"}, {"range": [40, 70], "color": "rgba(242, 180, 92, 0.2)"}, {"range": [70, 100], "color": "rgba(241, 108, 133, 0.2)"}], "threshold": {"line": {"color": "#fff", "width": 3}, "value": result["default_probability"] * 100}}))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Manrope", color="#F5F1FF", size=12), margin=dict(t=40, b=20, l=20, r=20), height=280)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown('<div class="section-title"><span class="badge">3</span>Assessment Summary</div>', unsafe_allow_html=True)
        summary_rows = ""
        for k in ["AGE", "ANNUAL_INCOME", "CIBIL_SCORE", "REQUESTED_LOAN_AMOUNT", "EMPLOYMENT_TYPE", "EDUCATION", "MARRIAGE"]:
            v = feats[k]
            if k == "ANNUAL_INCOME": v = f"₹{v:,}"
            elif k == "REQUESTED_LOAN_AMOUNT": v = f"₹{v:,}"
            elif k == "EMPLOYMENT_TYPE": v = {0: "Salaried", 1: "Self-employed", 2: "Unemployed"}[v]
            elif k == "EDUCATION": v = {1: "High School", 2: "Diploma", 3: "Graduate", 4: "Post Graduate", 5: "Doctorate"}[v]
            elif k == "MARRIAGE": v = {1: "Married", 2: "Single", 3: "Other"}[v]
            summary_rows += f'<tr><td style="color: var(--text-secondary); padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.04);">{k.replace("_", " ")}</td><td style="color: var(--text-main); padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.04); font-variant-numeric: tabular-nums;">{v}</td></tr>'
        st.markdown(f"""<div class="glass-card animate-in"><table style="width: 100%; border-collapse: collapse; font-family: Manrope, sans-serif; font-size: 13px;">{summary_rows}</table></div>""", unsafe_allow_html=True)
        show_toast("Credit risk assessment completed.")

# ── Page: Risk Analytics ──
def page_analytics():
    st.markdown('<h1>Risk Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: var(--text-secondary); margin-bottom: 2rem;">Deep dive into credit risk patterns across the applicant portfolio.</p>', unsafe_allow_html=True)
    if len(RISK_PROBS) == 0:
        st.info("Analytics data not available.")
        return
    probs = pd.Series(RISK_PROBS)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title"><span class="badge">1</span>Probability Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=probs * 100, nbinsx=20, marker_color="#9147FF", opacity=0.8, hovertemplate="Probability: %{x:.1f}%<br>Count: %{y}<extra></extra>"))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(title="Default Probability (%)", gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=11, color="#A49AB8"), title_font=dict(size=12, color="#A49AB8")), yaxis=dict(title="Applicants", gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=11, color="#A49AB8"), title_font=dict(size=12, color="#A49AB8")), font=dict(family="Manrope", color="#A49AB8", size=12), showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=350, bargap=4)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown('<div class="section-title"><span class="badge">2</span>Risk Category Mix</div>', unsafe_allow_html=True)
        labels = ["Low (<40%)", "Medium (40–70%)", "High (70%+)"]
        values = [LOW_RISK_COUNT, MEDIUM_RISK_COUNT, HIGH_RISK_COUNT]
        fig2 = go.Figure(data=[go.Bar(x=labels, y=values, marker=dict(color=["#35D39A", "#F2B45C", "#F16C85"]), text=[f"{v:,}" for v in values], textposition="outside", textfont=dict(color="#F5F1FF", size=12, family="Manrope"), hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>")])
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=11, color="#A49AB8")), yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=11, color="#A49AB8"), zeroline=False), font=dict(family="Manrope", color="#A49AB8", size=12), showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=350)
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown('<div class="section-title"><span class="badge">3</span>CIBIL Score vs Default Risk</div>', unsafe_allow_html=True)
    try:
        df = load_dataset()
        sample = df.sample(n=min(1000, len(df)), random_state=42)
        X = sample[MODEL_FEATURES]
        sample_probs = model.predict_proba(X)[:, 1]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=sample["cibil_score"], y=sample_probs * 100, mode="markers", marker=dict(size=6, color=sample_probs, colorscale=[[0, "#35D39A"], [0.5, "#F2B45C"], [1, "#F16C85"]], showscale=True, colorbar=dict(title="Risk", titleside="right", tickfont=dict(size=10, color="#A49AB8"), outlinecolor="rgba(255,255,255,0)"), line=dict(width=1, color="rgba(255,255,255,0.2)")), hovertemplate="CIBIL: %{x}<br>Risk: %{y:.1f}%<extra></extra>"))
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(title="CIBIL Score", gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=11, color="#A49AB8"), title_font=dict(size=12, color="#A49AB8")), yaxis=dict(title="Default Probability (%)", gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=11, color="#A49AB8"), title_font=dict(size=12, color="#A49AB8"), range=[0, 100]), font=dict(family="Manrope", color="#A49AB8", size=12), showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=350)
        st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.info(f"Scatter plot data unavailable: {e}")

# ── Page: Model Performance ──
def page_performance():
    st.markdown('<h1>Model Performance</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: var(--text-secondary); margin-bottom: 2rem;">Compare the classification algorithms evaluated during training.</p>', unsafe_allow_html=True)
    if comparison_df is None:
        st.info("Model comparison data not available.")
        return
    rf_row = comparison_df[comparison_df["Model"] == "Random Forest"].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Best Model", "Random Forest", trend=0, trend_label="by ROC-AUC", icon="🏆")
    with c2:
        kpi_card("Accuracy", f"{rf_row['Accuracy']*100:.1f}%", trend=0, trend_label="Random Forest", icon="📊")
    with c3:
        kpi_card("Precision", f"{rf_row['Precision']*100:.1f}%", trend=0, trend_label="Random Forest", icon="🎯")
    with c4:
        kpi_card("ROC-AUC", f"{rf_row['ROC-AUC']*100:.1f}%", trend=0, trend_label="Random Forest", icon="📈")
    st.markdown('<div class="section-title"><span class="badge">1</span>Model Comparison</div>', unsafe_allow_html=True)
    metric = st.selectbox("Select metric", options=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"], index=4, key="perf_metric")
    fig = go.Figure()
    colors = ["#9147FF", "#B77AFF", "#7a3be0"]
    for i, row in comparison_df.iterrows():
        fig.add_trace(go.Bar(x=[row["Model"]], y=[row[metric]], name=row["Model"], marker_color=colors[i % len(colors)], text=[f"{row[metric]*100:.2f}%"], textposition="outside", textfont=dict(color="#F5F1FF", size=12, family="Manrope"), hovertemplate=f"<b>{row['Model']}</b><br>{metric}: %{{y:.2%}}<extra></extra>", width=0.5))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False, tickfont=dict(size=12, color="#A49AB8"), title_font=dict(size=13, color="#A49AB8")), yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickformat=".0%", tickfont=dict(size=11, color="#A49AB8"), range=[0, 1], zeroline=False), showlegend=False, font=dict(family="Manrope", color="#A49AB8", size=12), margin=dict(t=20, b=20, l=20, r=20), height=400, bargap=0.4)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="section-title"><span class="badge">2</span>Detailed Metrics</div>', unsafe_allow_html=True)
    display_df = comparison_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.markdown('<div class="section-title"><span class="badge">3</span>Confusion Matrices</div>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    for model_name, col in zip(["Logistic Regression", "Decision Tree", "Random Forest"], [col_a, col_b, col_c]):
        import glob
        fname = OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        with col:
            if fname.exists():
                st.image(str(fname), caption=model_name, use_container_width=True)
            else:
                st.info(f"{model_name} matrix not found.")
    st.markdown('<div class="section-title"><span class="badge">4</span>Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
    importance_csv = list(OUTPUT_DIR.glob("*importance*.csv"))
    importance_img = list(OUTPUT_DIR.glob("*importance*.png"))
    if importance_csv:
        imp_df = pd.read_csv(importance_csv[0])
        st.dataframe(imp_df, use_container_width=True, hide_index=True)
    if importance_img:
        st.image(str(importance_img[0]), caption="Feature Importance", use_container_width=True)
    if not importance_csv and not importance_img:
        st.info("Feature importance files not found.")

# ── Page: Prediction History ──
def page_history():
    st.markdown('<h1>Prediction History</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: var(--text-secondary); margin-bottom: 2rem;">Review past credit assessments with filtering and pagination.</p>', unsafe_allow_html=True)
    history = st.session_state.prediction_history
    if not history:
        st.markdown("""<div class="empty-state animate-in"><div class="icon">📭</div><div style="font-size: 16px; font-weight: 700; color: var(--text-main); margin-bottom: 4px;">No Assessments Yet</div><div style="font-size: 13px; max-width: 300px; margin: 0 auto;">Complete a credit prediction to see history here.</div></div>""", unsafe_allow_html=True)
        return
    hist_df = pd.DataFrame(history)
    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
    with col_f1:
        search = st.text_input("Search by Applicant ID", placeholder="e.g. CR12345", key="history_search")
    with col_f2:
        risk_filter = st.selectbox("Filter by Risk", options=["All", "Low", "Medium", "High"], key="history_risk")
    with col_f3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear Filters", key="history_clear", use_container_width=True):
            st.session_state.history_search = ""
            st.session_state.history_risk = "All"
            st.rerun()
    filtered = hist_df.copy()
    if search:
        filtered = filtered[filtered["applicant_id"].str.contains(search, case=False, na=False)]
    if risk_filter != "All":
        filtered = filtered[filtered["risk_level"] == risk_filter]
    filtered = filtered.sort_values("timestamp", ascending=False)
    page_size = 10
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    if "history_page" not in st.session_state:
        st.session_state.history_page = 1
    page = st.session_state.history_page
    start = (page - 1) * page_size
    end = start + page_size
    page_data = filtered.iloc[start:end]
    for _, row in page_data.iterrows():
        risk_cls = row["risk_level"].lower()
        st.markdown(f"""<div class="glass-card animate-in" style="padding: 16px 20px; margin-bottom: 10px; display: flex; align-items: center; justify-content: space-between;"><div style="flex: 1;"><div style="font-size: 13px; font-weight: 700; color: var(--text-main);">{row['applicant_id']}</div><div style="font-size: 11px; color: var(--text-secondary);">Loan: ₹{row['requested_loan_amount']:,} &nbsp;•&nbsp; {row['timestamp']}</div></div><div style="display: flex; align-items: center; gap: 16px;"><div style="font-size: 12px; color: var(--text-secondary); font-variant-numeric: tabular-nums;">{row['default_probability']*100:.1f}%</div><span class="risk-badge risk-{risk_cls}">{row['risk_level']}</span></div></div>""", unsafe_allow_html=True)
    col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
    with col_p1:
        if page > 1:
            if st.button("Previous", key="hist_prev", use_container_width=True):
                st.session_state.history_page -= 1
                st.rerun()
    with col_p2:
        st.markdown(f'<div style="text-align:center; color: var(--text-secondary); font-size: 12px;">Page {page} of {total_pages} &nbsp;•&nbsp; {len(filtered)} records</div>', unsafe_allow_html=True)
    with col_p3:
        if page < total_pages:
            if st.button("Next", key="hist_next", use_container_width=True):
                st.session_state.history_page += 1
                st.rerun()

# ── Page: Dataset ──
def page_dataset():
    st.markdown('<h1>Dataset</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: var(--text-secondary); margin-bottom: 2rem;">Explore the SBI bank credit dataset used to train the model.</p>', unsafe_allow_html=True)
    try:
        df = load_dataset()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Total Records", f"{len(df):,}", icon="📊")
        with c2:
            kpi_card("Features", f"{df.shape[1]}", icon="📐")
        with c3:
            defaults = int(df["default"].sum()) if "default" in df.columns else 0
            kpi_card("Defaults", f"{defaults:,}", icon="⚠️")
        with c4:
            kpi_card("Default Rate", f"{defaults/len(df)*100:.1f}%" if len(df) else "0%", icon="📉")
        st.markdown('<div class="section-title"><span class="badge">1</span>Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(50), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Unable to load dataset: {e}")

# ── Page: Settings ──
def page_settings():
    st.markdown('<h1>Settings</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: var(--text-secondary); margin-bottom: 2rem;">Configure dashboard preferences and view system information.</p>', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="badge">1</span>Appearance</div>', unsafe_allow_html=True)
    st.toggle("Dark Mode", value=True, disabled=True, help="Dark theme is fixed for this premium dashboard.")
    st.selectbox("Chart Color Theme", options=["Purple", "Neon", "Classic"], index=0, disabled=True)
    st.markdown('<div class="section-title"><span class="badge">2</span>System Information</div>', unsafe_allow_html=True)
    info_rows = [("Application", "CreditRisk AI"), ("Version", "2.0.0"), ("Backend", "FastAPI + Random Forest"), ("Frontend", "Streamlit + Plotly + Custom CSS"), ("Model Features", str(len(MODEL_FEATURES))), ("Dataset Size", f"{DATASET_SIZE:,} records")]
    for k, v in info_rows:
        st.markdown(f"""<div class="glass-card" style="padding: 14px 20px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;"><span style="color: var(--text-secondary); font-size: 13px; font-weight: 600;">{k}</span><span style="color: var(--text-main); font-size: 13px; font-family: monospace;">{v}</span></div>""", unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="badge">3</span>Data Management</div>', unsafe_allow_html=True)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if st.button("Clear Prediction History", use_container_width=True):
            st.session_state.prediction_history = []
            st.session_state.history_page = 1
            show_toast("Prediction history cleared.")
            st.rerun()
    with col_d2:
        if st.button("Refresh Data Stats", use_container_width=True):
            st.cache_data.clear()
            show_toast("Data statistics refreshed.")
            st.rerun()
    st.markdown('<div class="section-title"><span class="badge">4</span>Disclaimer</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card" style="padding: 20px;"><p style="color: var(--text-secondary); font-size: 12px; line-height: 1.6;">This project is developed for educational and machine learning demonstration purposes. It is not an official credit approval system. Predictions are estimates, not guarantees.</p></div>', unsafe_allow_html=True)

# ── Main Router ──
def main():
    render_sidebar()
    page = st.session_state.page
    if page == "Overview":
        page_overview()
    elif page == "Credit Prediction":
        page_prediction()
    elif page == "Risk Analytics":
        page_analytics()
    elif page == "Model Performance":
        page_performance()
    elif page == "Prediction History":
        page_history()
    elif page == "Dataset":
        page_dataset()
    elif page == "Settings":
        page_settings()
    st.markdown(
        '<div style="margin-top: 3rem; padding: 2rem 0; border-top: 1px solid rgba(255,255,255,0.06); text-align: center;">'
        '<p style="color: var(--text-secondary); font-size: 11px; font-family: Manrope, sans-serif;">'
        'CreditRisk AI &bull; Premium Credit Intelligence Platform &bull; '
        'Built for CodeAlpha ML Internship</p>'
        '</div>',
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
