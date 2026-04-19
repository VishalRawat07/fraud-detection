import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudGuard",
    page_icon="🛡️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* App background */
.stApp { background: #0f1117; color: #e8eaf0; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #2a2f3d;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #8b92a5;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: .08em;
}

/* Number inputs & selects */
.stNumberInput input, .stSelectbox select {
    background: #1e2333 !important;
    border: 1px solid #2a2f3d !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Mono', monospace !important;
}
.stNumberInput input:focus, .stSelectbox select:focus {
    border-color: #4f7fff !important;
    box-shadow: 0 0 0 2px rgba(79,127,255,.25) !important;
}

/* Labels */
.stNumberInput label, .stSelectbox label {
    color: #8b92a5 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: .06em !important;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, #4f7fff, #7b5ea7) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 0.65rem 2.5rem !important;
    letter-spacing: .04em;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: .88; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #1e2333;
    border: 1px solid #2a2f3d;
    border-radius: 12px;
    padding: 1rem 1.25rem !important;
}
[data-testid="stMetric"] label { color: #8b92a5 !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: .06em; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #e8eaf0 !important; font-family: 'DM Mono', monospace; }

/* Result banners */
.fraud-banner {
    background: #2d1a1a;
    border: 1px solid #c0392b;
    border-left: 4px solid #e74c3c;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    color: #f1948a;
    font-size: 15px;
    font-weight: 500;
}
.safe-banner {
    background: #1a2d1e;
    border: 1px solid #27ae60;
    border-left: 4px solid #2ecc71;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    color: #82e0aa;
    font-size: 15px;
    font-weight: 500;
}
.banner-title { font-size: 18px; font-weight: 700; margin-bottom: 4px; }
.banner-sub   { font-size: 13px; opacity: .75; }

/* Section headers */
.section-header {
    color: #8b92a5;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .12em;
    margin: 1.5rem 0 .5rem;
    border-bottom: 1px solid #2a2f3d;
    padding-bottom: 6px;
}

/* Feature table */
.feat-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.feat-table th { color: #8b92a5; font-weight: 500; text-align: left; padding: 6px 10px; border-bottom: 1px solid #2a2f3d; }
.feat-table td { color: #e8eaf0; padding: 7px 10px; border-bottom: 1px solid #1e2333; }
.feat-table tr:last-child td { border-bottom: none; }
.tag-high { background:#2d1a1a; color:#e74c3c; border-radius:6px; padding:2px 8px; font-size:11px; font-weight:600; }
.tag-low  { background:#1a2d1e; color:#2ecc71; border-radius:6px; padding:2px 8px; font-size:11px; font-weight:600; }
.tag-med  { background:#2d2a1a; color:#f39c12; border-radius:6px; padding:2px 8px; font-size:11px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────


@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_pipeline.pkl")


try:
    model = load_model()
    model_loaded = True
except Exception:
    model_loaded = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FraudGuard")
    st.markdown("---")
    st.markdown("**Transaction input**")
    st.markdown(
        "Fill in the details on the right panel and click **Predict** to analyse the transaction.")
    st.markdown("---")
    st.markdown("**Risk thresholds**")
    high_thresh = st.slider("High risk above (%)", 50, 90, 70)
    st.markdown("---")
    if model_loaded:
        st.success("Model loaded ✓")
    else:
        st.warning("Model not found — using rule-based fallback")

# ── Feature engineering helpers ───────────────────────────────────────────────


def compute_features(tx_type, amount, old_orig, new_orig, old_dest, new_dest):
    """Return both the model input DataFrame and a dict of risk signals."""
    # Balance error: how much the books don't balance
    balance_error_orig = abs((old_orig - amount) - new_orig)
    balance_error_dest = abs((old_dest + amount) - new_dest)

    # Surge ratio: amount vs sender's old balance
    surge_ratio = amount / (old_orig + 1)

    # Drain flags
    zero_drain_orig = int(new_orig == 0 and old_orig > 0)
    zero_dest = int(old_dest == 0 and new_dest == 0)

    df = pd.DataFrame([{
        "type":             tx_type,
        "amount":           amount,
        "oldbalanceOrg":    old_orig,
        "newbalanceOrig":   new_orig,
        "oldbalanceDest":   old_dest,
        "newbalanceDest":   new_dest,
        "balanceDiffOrig":  old_orig - new_orig - amount,
        "balanceDiffDest":  new_dest - old_dest - amount,
    }])

    signals = {
        "balance_error_orig": balance_error_orig,
        "balance_error_dest": balance_error_dest,
        "surge_ratio":        surge_ratio,
        "zero_drain_orig":    zero_drain_orig,
        "zero_dest":          zero_dest,
    }
    return df, signals


def rule_based_score(tx_type, amount, old_orig, new_orig, old_dest, new_dest, signals):
    """Heuristic fraud probability (0–1) used when model is unavailable."""
    score = 0.0
    if tx_type in ("CASH_OUT", "TRANSFER"):
        score += 0.20
    if signals["zero_drain_orig"]:
        score += 0.35
    if signals["surge_ratio"] > 0.9:
        score += 0.20
    if signals["balance_error_orig"] > 1:
        score += 0.15
    if signals["zero_dest"]:
        score += 0.10
    return min(score, 0.99)


# ── Gauge chart ───────────────────────────────────────────────────────────────
def fraud_gauge(prob, high_thresh):
    color = "#e74c3c" if prob * \
        100 >= high_thresh else ("#f39c12" if prob > 0.35 else "#2ecc71")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 36,
                                        "color": "#e8eaf0", "family": "DM Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#2a2f3d", "tickfont": {"color": "#8b92a5", "size": 11}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#1e2333",
            "bordercolor": "#2a2f3d",
            "steps": [
                {"range": [0, 35],         "color": "#1a2d1e"},
                {"range": [35, high_thresh], "color": "#2d2a1a"},
                {"range": [high_thresh, 100], "color": "#2d1a1a"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": prob * 100},
        },
        title={"text": "Fraud risk score", "font": {
            "color": "#8b92a5", "size": 13}},
    ))
    fig.update_layout(
        height=240, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e8eaf0",
    )
    return fig


# ── Signal bar chart ──────────────────────────────────────────────────────────
def signal_chart(signals, prob):
    labels = [
        "Account drain",
        "Surge ratio",
        "Balance error (sender)",
        "Balance error (receiver)",
        "Zero-balance dest",
    ]
    # Normalise each signal to 0-1 for visual display
    raw = [
        signals["zero_drain_orig"],
        min(signals["surge_ratio"], 1.0),
        min(signals["balance_error_orig"] /
            (signals["balance_error_orig"] + 1000), 1.0),
        min(signals["balance_error_dest"] /
            (signals["balance_error_dest"] + 1000), 1.0),
        signals["zero_dest"],
    ]
    colors = ["#e74c3c" if v > 0.5 else "#f39c12" if v >
              0.2 else "#2ecc71" for v in raw]

    fig = go.Figure(go.Bar(
        x=raw, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.0%}" for v in raw],
        textposition="outside",
        textfont={"color": "#8b92a5", "size": 11, "family": "DM Mono"},
    ))
    fig.update_layout(
        height=230, margin=dict(l=10, r=40, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 1.25], showgrid=False,
                   zeroline=False, showticklabels=False),
        yaxis=dict(tickfont={"color": "#8b92a5", "size": 12}),
        bargap=0.35,
    )
    return fig


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("<h2 style='margin:0 0 4px;color:#e8eaf0'>Transaction analyser</h2>",
            unsafe_allow_html=True)
st.markdown("<p style='color:#8b92a5;margin:0 0 1.5rem'>Enter the transaction details below and click Predict.</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    transaction_type = st.selectbox(
        "Transaction type", ["PAYMENT", "TRANSFER", "CASH_OUT"])
with col2:
    amount = st.number_input("Amount (₹)", min_value=0.0,
                             value=1000.0, step=100.0, format="%.2f")
with col3:
    st.write("")  # spacer

col4, col5, col6, col7 = st.columns(4)
with col4:
    old_orig = st.number_input(
        "Old balance — sender", min_value=0.0, value=10000.0, format="%.2f")
with col5:
    new_orig = st.number_input(
        "New balance — sender", min_value=0.0, value=9000.0, format="%.2f")
with col6:
    old_dest = st.number_input(
        "Old balance — receiver", min_value=0.0, value=0.0, format="%.2f")
with col7:
    new_dest = st.number_input(
        "New balance — receiver", min_value=0.0, value=0.0, format="%.2f")

st.markdown("")
predict_clicked = st.button("🔍  Predict transaction")

# ── Prediction & results ──────────────────────────────────────────────────────
if predict_clicked:
    input_df, signals = compute_features(
        transaction_type, amount, old_orig, new_orig, old_dest, new_dest
    )

    if model_loaded:
        prediction = int(model.predict(input_df)[0])
        if hasattr(model, "predict_proba"):
            fraud_prob = float(model.predict_proba(input_df)[0][1])
        else:
            fraud_prob = rule_based_score(
                transaction_type, amount, old_orig, new_orig, old_dest, new_dest, signals
            )
    else:
        fraud_prob = rule_based_score(
            transaction_type, amount, old_orig, new_orig, old_dest, new_dest, signals
        )
        prediction = int(fraud_prob * 100 >= high_thresh)

    is_fraud = prediction == 1
    risk_pct = round(fraud_prob * 100, 1)

    st.markdown("---")

    # Result banner
    if is_fraud:
        st.markdown(f"""
        <div class="fraud-banner">
            <div class="banner-title">⚠️ Fraud risk detected</div>
            <div>This transaction shows patterns consistent with fraudulent activity.</div>
            <div class="banner-sub">Risk score: {risk_pct}% — exceeds your threshold of {high_thresh}%</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="safe-banner">
            <div class="banner-title">✅ Transaction looks legitimate</div>
            <div>No significant fraud signals detected for this transaction.</div>
            <div class="banner-sub">Risk score: {risk_pct}% — below your threshold of {high_thresh}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Metric row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk score",    f"{risk_pct}%")
    m2.metric("Amount",        f"₹{amount:,.0f}")
    m3.metric("Surge ratio",   f"{min(signals['surge_ratio'], 99.9):.1f}×")
    m4.metric("Balance error", f"₹{signals['balance_error_orig']:,.0f}")

    st.markdown("")

    # Charts
    g_col, b_col = st.columns([1, 1])
    with g_col:
        st.markdown("<div class='section-header'>Risk gauge</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(fraud_gauge(fraud_prob, high_thresh),
                        use_container_width=True)
    with b_col:
        st.markdown(
            "<div class='section-header'>Signal breakdown</div>", unsafe_allow_html=True)
        st.plotly_chart(signal_chart(signals, fraud_prob),
                        use_container_width=True)

    # Feature table
    st.markdown("<div class='section-header'>Engineered features</div>",
                unsafe_allow_html=True)

    def risk_tag(val, low=0.1, high=0.5):
        if val > high:
            return "<span class='tag-high'>High</span>"
        elif val > low:
            return "<span class='tag-med'>Medium</span>"
        else:
            return "<span class='tag-low'>Low</span>"

    surge_norm = min(signals["surge_ratio"], 1.0)
    be_orig_n = min(signals["balance_error_orig"] /
                    (signals["balance_error_orig"] + 1000), 1.0)
    be_dest_n = min(signals["balance_error_dest"] /
                    (signals["balance_error_dest"] + 1000), 1.0)

    rows = [
        ("Account drain (sender)",     f"{signals['zero_drain_orig']}",              risk_tag(
            signals["zero_drain_orig"], .05, .5)),
        ("Surge ratio",
         f"{signals['surge_ratio']:.3f}×",             risk_tag(surge_norm, .3, .7)),
        ("Balance error — sender",
         f"₹{signals['balance_error_orig']:,.2f}",     risk_tag(be_orig_n, .1, .5)),
        ("Balance error — receiver",
         f"₹{signals['balance_error_dest']:,.2f}",     risk_tag(be_dest_n, .1, .5)),
        ("Zero-balance destination",   f"{signals['zero_dest']}",
         risk_tag(signals["zero_dest"], .05, .5)),
    ]

    table_html = "<table class='feat-table'><thead><tr><th>Feature</th><th>Value</th><th>Risk level</th></tr></thead><tbody>"
    for name, val, tag in rows:
        table_html += f"<tr><td>{name}</td><td style='font-family:DM Mono,monospace'>{val}</td><td>{tag}</td></tr>"
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)
