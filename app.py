"""
VeriClaim — AI-Powered Health Insurance Fraud Detection
Streamlit MVP | Team 5 — Eduinx AIPM Capstone | June 2026
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go
import plotly.express as px
import os
import io

# ── PAGE CONFIG ───────────────────────────────────────────────────
st.set_page_config(
    page_title="VeriClaim — Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── STYLES ────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .main { background-color: #F8FAFB; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E8EDF2;
    border-radius: 10px;
    padding: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }

  /* Score pills */
  .pill-red    { background:#FADBD8; color:#922B21; padding:4px 12px; border-radius:20px; font-weight:600; font-size:13px; display:inline-block; }
  .pill-amber  { background:#FEF9E7; color:#B7770D; padding:4px 12px; border-radius:20px; font-weight:600; font-size:13px; display:inline-block; }
  .pill-green  { background:#D5F5E3; color:#1E8449; padding:4px 12px; border-radius:20px; font-weight:600; font-size:13px; display:inline-block; }

  /* Section headers */
  .section-header {
    font-size: 13px; font-weight: 600; color: #717D7E;
    letter-spacing: 0.06em; text-transform: uppercase;
    margin-bottom: 8px; margin-top: 4px;
  }

  /* Card box */
  .info-card {
    background: #FFFFFF; border: 1px solid #E8EDF2;
    border-radius: 10px; padding: 16px 20px;
    margin-bottom: 12px;
  }

  /* SHAP factor row */
  .shap-row {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0; border-bottom: 1px solid #F0F3F4;
  }
  .shap-label { font-size: 13px; color: #1A252F; flex: 1; }
  .shap-bar-pos { background: #E74C3C; height: 12px; border-radius: 4px; }
  .shap-bar-neg { background: #2ECC71; height: 12px; border-radius: 4px; }

  /* Alert boxes */
  .alert-red    { background:#FADBD8; border-left:4px solid #E74C3C; padding:12px 16px; border-radius:6px; margin:8px 0; }
  .alert-amber  { background:#FEF9E7; border-left:4px solid #F39C12; padding:12px 16px; border-radius:6px; margin:8px 0; }
  .alert-green  { background:#D5F5E3; border-left:4px solid #27AE60; padding:12px 16px; border-radius:6px; margin:8px 0; }

  /* Hide streamlit branding */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model       = joblib.load("model/vericlaim_model.pkl")
    feature_cols = joblib.load("model/feature_cols.pkl")
    metrics     = joblib.load("model/metrics.pkl")
    return model, feature_cols, metrics

@st.cache_data
def load_scored_providers():
    df = pd.read_csv("model/scored_providers.csv")
    return df

MODEL_LOADED = os.path.exists("model/vericlaim_model.pkl")

# ── HELPER FUNCTIONS ──────────────────────────────────────────────
def score_to_routing(score):
    if score < 30:
        return "Auto-Clear", "green"
    elif score < 70:
        return "Soft Review", "amber"
    else:
        return "Full Investigation", "red"

def routing_pill(label, color):
    return f'<span class="pill-{color}">{label}</span>'

def score_gauge(score, title="Risk Score"):
    color = "#27AE60" if score < 30 else ("#F39C12" if score < 70 else "#E74C3C")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x":[0,1], "y":[0,1]},
        title={"text": title, "font":{"size":18, "color":"#1A252F"}},
        number={"font":{"size":40, "color":color}},
        gauge={
            "axis":{"range":[0,100], "tickwidth":1, "tickcolor":"#BDC3C7"},
            "bar":{"color":color},
            "bgcolor":"white",
            "borderwidth":2,
            "bordercolor":"#E8EDF2",
            "steps":[
                {"range":[0,30],   "color":"#D5F5E3"},
                {"range":[30,70],  "color":"#FEF9E7"},
                {"range":[70,100], "color":"#FADBD8"},
            ],
            "threshold":{
                "line":{"color":color,"width":4},
                "thickness":0.9,
                "value":score
            }
        }
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=20,r=20,t=40,b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def shap_explanation(model, feature_cols, feature_values):
    """Generate SHAP explanation for a single provider's features."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(np.array(feature_values).reshape(1,-1))

    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0] if shap_values.ndim == 1 else shap_values[0]

    pairs = sorted(zip(feature_cols, sv, feature_values),
                   key=lambda x: abs(x[1]), reverse=True)
    return pairs[:5]

SHAP_LABELS = {
    "IP_TotalClaims":            "Total inpatient claims filed",
    "IP_AvgClaimAmt":            "Average inpatient claim amount (₹)",
    "IP_TotalClaimAmt":          "Total inpatient claim value (₹)",
    "IP_AvgClaimDuration":       "Average patient stay duration (days)",
    "IP_MaxClaimDuration":       "Longest inpatient stay (days)",
    "IP_UniquePatients":         "Number of unique patients treated",
    "IP_UniqueAdmittingDoctors": "Number of unique admitting physicians",
    "IP_AvgDeductible":          "Average deductible paid per claim",
    "OP_TotalClaims":            "Total outpatient claims filed",
    "OP_AvgClaimAmt":            "Average outpatient claim amount (₹)",
    "OP_TotalClaimAmt":          "Total outpatient claim value (₹)",
    "OP_UniquePatients":         "Unique outpatient visits",
    "OP_AvgDeductible":          "Average outpatient deductible",
    "Bene_AvgAge":               "Average patient age",
    "Bene_AvgChronicCond":       "Average chronic conditions per patient",
    "Bene_AvgIPReimbursement":   "Average annual IP reimbursement per patient",
    "Bene_AvgOPReimbursement":   "Average annual OP reimbursement per patient",
    "IP_ClaimPerPatient":        "Inpatient claims per unique patient (ratio)",
    "OP_ClaimPerPatient":        "Outpatient claims per unique patient (ratio)",
    "IP_OP_ClaimRatio":          "Ratio of inpatient to outpatient claims",
    "AvgClaimAmtRatio":          "Inpatient vs outpatient average claim ratio",
}

def shap_reason(feature, value, shap_val):
    label = SHAP_LABELS.get(feature, feature)
    direction = "increases" if shap_val > 0 else "reduces"
    if "Amt" in feature or "Reimbursement" in feature:
        val_str = f"₹{value:,.0f}"
    elif "Duration" in feature:
        val_str = f"{value:.1f} days"
    elif "Ratio" in feature:
        val_str = f"{value:.2f}x"
    elif "Age" in feature:
        val_str = f"{value:.1f} years"
    else:
        val_str = f"{value:.1f}"
    return f"{label}: {val_str} — {direction} fraud risk"


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 8px 0 16px;'>
      <span style='font-size:32px;'>🛡️</span><br>
      <span style='font-size:20px; font-weight:700; color:#1B4F72;'>VeriClaim</span><br>
      <span style='font-size:12px; color:#717D7E;'>AI Fraud Detection</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Investigator Queue",
         "🔍 Claim Deep Dive",
         "📁 Batch Upload",
         "📊 Management Dashboard"],
        label_visibility="collapsed"
    )

    st.divider()

    if MODEL_LOADED:
        _, _, metrics = load_model()
        st.markdown('<p class="section-header">Model Performance</p>', unsafe_allow_html=True)
        st.metric("F1 Score",  f"{metrics['f1']:.4f}")
        st.metric("ROC-AUC",   f"{metrics['auc']:.4f}")
        st.caption("XGBoost + SMOTE | Kaggle Medicare Dataset")
    else:
        st.warning("Model not found.\nRun: `python train_model.py`")

    st.divider()
    st.caption("Team 5 — Eduinx AIPM Capstone\nJune 2026 | v1.0")


# ══════════════════════════════════════════════════════════════════
# PAGE 1 — INVESTIGATOR QUEUE
# ══════════════════════════════════════════════════════════════════
if page == "🏠 Investigator Queue":
    st.markdown("## 🏠 Investigator Queue")
    st.caption("Claims ranked by AI fraud risk score — highest risk first")

    if not MODEL_LOADED:
        st.error("Please run `python train_model.py` first to generate the model.")
        st.stop()

    df = load_scored_providers()
    model, feature_cols, _ = load_model()

    # ── KPI CARDS ─────────────────────────────────────────────────
    total       = len(df)
    auto_clear  = (df["RiskScore"] < 30).sum()
    soft_review = ((df["RiskScore"] >= 30) & (df["RiskScore"] < 70)).sum()
    full_inv    = (df["RiskScore"] >= 70).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Providers", f"{total:,}")
    c2.metric("🟢 Auto-Clear",   f"{auto_clear:,}",  f"{auto_clear/total*100:.1f}%")
    c3.metric("🟡 Soft Review",  f"{soft_review:,}", f"{soft_review/total*100:.1f}%")
    c4.metric("🔴 Full Investigation", f"{full_inv:,}", f"{full_inv/total*100:.1f}%")

    st.divider()

    # ── FILTERS ───────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns([2,2,1])
    with col_f1:
        search = st.text_input("🔍 Search Provider ID", placeholder="e.g. PRV51459")
    with col_f2:
        routing_filter = st.multiselect(
            "Filter by routing",
            ["Auto-Clear","Soft Review","Full Investigation"],
            default=["Soft Review","Full Investigation"]
        )
    with col_f3:
        show_n = st.selectbox("Show top", [20, 50, 100, "All"], index=0)

    # ── BUILD DISPLAY TABLE ───────────────────────────────────────
    display = df.copy().sort_values("RiskScore", ascending=False)

    display["Routing"] = display["RiskScore"].apply(lambda s: score_to_routing(s)[0])

    if search:
        display = display[display["Provider"].str.contains(search, case=False, na=False)]

    if routing_filter:
        display = display[display["Routing"].isin(routing_filter)]

    if show_n != "All":
        display = display.head(int(show_n))

    # Build human-readable columns
    queue_table = pd.DataFrame({
        "Provider ID":       display["Provider"],
        "Risk Score":        display["RiskScore"].round(1),
        "Routing":           display["Routing"],
        "IP Claims":         display.get("IP_TotalClaims", pd.Series([0]*len(display))).fillna(0).astype(int),
        "Avg Claim (₹)":     display.get("IP_AvgClaimAmt", pd.Series([0]*len(display))).fillna(0).round(0),
        "Unique Patients":   display.get("IP_UniquePatients", pd.Series([0]*len(display))).fillna(0).astype(int),
        "Fraud Label":       display["PotentialFraud"].map({1:"⚠️ Fraud", 0:"✅ Genuine"}),
    })

    # Colour-code the score column
    def colour_score(val):
        if val < 30:   return "background-color: #D5F5E3; color: #1E8449; font-weight:600"
        elif val < 70: return "background-color: #FEF9E7; color: #B7770D; font-weight:600"
        else:          return "background-color: #FADBD8; color: #922B21; font-weight:600"

    styled = queue_table.style.applymap(colour_score, subset=["Risk Score"])
    st.dataframe(styled, use_container_width=True, height=480)

    st.caption(f"Showing {len(queue_table)} providers · Click 'Claim Deep Dive' in the sidebar to analyse any provider")

    # ── SCORE DISTRIBUTION ────────────────────────────────────────
    st.divider()
    st.markdown("**Risk Score Distribution**")
    fig_dist = px.histogram(
        df, x="RiskScore", nbins=40,
        color_discrete_sequence=["#2E86C1"],
        labels={"RiskScore":"Fraud Risk Score", "count":"Number of Providers"}
    )
    fig_dist.add_vline(x=30, line_dash="dash", line_color="#27AE60",  annotation_text="Auto-Clear boundary")
    fig_dist.add_vline(x=70, line_dash="dash", line_color="#E74C3C",  annotation_text="Full Investigation boundary")
    fig_dist.update_layout(height=280, margin=dict(l=0,r=0,t=20,b=0),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_dist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 2 — CLAIM DEEP DIVE
# ══════════════════════════════════════════════════════════════════
elif page == "🔍 Claim Deep Dive":
    st.markdown("## 🔍 Claim Deep Dive")
    st.caption("Enter a Provider ID to get a detailed AI risk analysis with SHAP explanation")

    if not MODEL_LOADED:
        st.error("Please run `python train_model.py` first.")
        st.stop()

    model, feature_cols, _ = load_model()
    df = load_scored_providers()

    provider_list = df["Provider"].tolist()

    col_input, col_btn = st.columns([3,1])
    with col_input:
        provider_id = st.selectbox(
            "Select or type Provider ID",
            options=provider_list,
            index=0,
        )

    if provider_id:
        row = df[df["Provider"] == provider_id]
        if row.empty:
            st.error(f"Provider '{provider_id}' not found in dataset.")
            st.stop()

        row = row.iloc[0]
        score    = round(row["RiskScore"], 1)
        fraud_prob = round(row["FraudProbability"] * 100, 1)
        routing, color = score_to_routing(score)
        true_label = row.get("PotentialFraud", None)

        # ── TOP SECTION ───────────────────────────────────────────
        col_gauge, col_detail = st.columns([1, 2])

        with col_gauge:
            st.plotly_chart(score_gauge(score), use_container_width=True)
            st.markdown(
                f'<div style="text-align:center; margin-top:-8px;">'
                f'{routing_pill(routing, color)}</div>',
                unsafe_allow_html=True
            )
            if true_label is not None:
                label_str = "⚠️ Confirmed Fraud" if true_label == 1 else "✅ Genuine Provider"
                st.markdown(f'<div style="text-align:center; margin-top:8px; font-size:13px; color:#717D7E;">Ground truth: {label_str}</div>', unsafe_allow_html=True)

        with col_detail:
            st.markdown('<p class="section-header">Provider Summary</p>', unsafe_allow_html=True)

            ip_claims  = int(row.get("IP_TotalClaims", 0))
            op_claims  = int(row.get("OP_TotalClaims", 0))
            avg_claim  = float(row.get("IP_AvgClaimAmt", 0))
            patients   = int(row.get("IP_UniquePatients", 0))
            avg_dur    = float(row.get("IP_AvgClaimDuration", 0))

            kv_data = {
                "Provider ID":            provider_id,
                "Fraud Risk Score":       f"{score}/100  ({fraud_prob}% probability)",
                "Routing Decision":       routing,
                "Inpatient Claims":       f"{ip_claims:,}",
                "Outpatient Claims":      f"{op_claims:,}",
                "Average Claim Amount":   f"₹{avg_claim:,.0f}",
                "Unique Patients":        f"{patients:,}",
                "Avg Stay Duration":      f"{avg_dur:.1f} days",
            }
            for k, v in kv_data.items():
                c1, c2 = st.columns([1.4, 2])
                c1.markdown(f"**{k}**")
                c2.markdown(v)

        st.divider()

        # ── ROUTING ALERT ─────────────────────────────────────────
        if color == "red":
            st.markdown(f'<div class="alert-red"><b>🔴 Full Investigation Required</b><br>This provider has a high fraud probability ({fraud_prob}%). A senior investigator must review all claims, request supporting documents, and consider initiating a provider audit.</div>', unsafe_allow_html=True)
        elif color == "amber":
            st.markdown(f'<div class="alert-amber"><b>🟡 Soft Review Recommended</b><br>Risk score {score}/100. An investigator should review the flagged factors below and decide whether to approve or escalate. Target: 15 minutes per case.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-green"><b>🟢 Auto-Clear</b><br>Risk score {score}/100. This provider meets the auto-clear threshold. Claim can be approved for fast-track settlement without investigator review.</div>', unsafe_allow_html=True)

        st.divider()

        # ── SHAP EXPLANATION ──────────────────────────────────────
        st.markdown('<p class="section-header">AI Explanation — Why This Score?</p>', unsafe_allow_html=True)
        st.caption("Top 5 factors driving the fraud risk score for this provider")

        try:
            feat_values = [row.get(f, 0) for f in feature_cols]
            top_factors = shap_explanation(model, feature_cols, feat_values)

            for feat, sv, val in top_factors:
                reason = shap_reason(feat, val, sv)
                direction_icon = "🔴" if sv > 0 else "🟢"
                bar_pct = min(abs(sv) / max(abs(s) for _,s,_ in top_factors) * 100, 100)

                with st.container():
                    cols = st.columns([0.05, 0.6, 0.35])
                    cols[0].markdown(direction_icon)
                    cols[1].markdown(f"<span style='font-size:13px;'>{reason}</span>", unsafe_allow_html=True)
                    bar_color = "#E74C3C" if sv > 0 else "#27AE60"
                    cols[2].markdown(
                        f'<div style="background:{bar_color};height:10px;border-radius:5px;width:{bar_pct:.0f}%;margin-top:8px;"></div>',
                        unsafe_allow_html=True
                    )
                st.markdown('<hr style="margin:4px 0; border-color:#F0F3F4;">', unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"SHAP explanation not available: {e}")

        st.divider()

        # ── INVESTIGATOR ACTIONS ──────────────────────────────────
        st.markdown('<p class="section-header">Investigator Actions</p>', unsafe_allow_html=True)
        col_a1, col_a2, col_a3, col_a4 = st.columns(4)

        with col_a1:
            if st.button("✅ Approve Claim", use_container_width=True):
                st.success("Claim approved. Action logged.")

        with col_a2:
            if st.button("⬆️ Escalate to Senior", use_container_width=True):
                st.warning("Escalated to senior investigator. Case ID logged.")

        with col_a3:
            if st.button("📄 Request Documents", use_container_width=True):
                st.info("Document request triggered. Provider notified.")

        with col_a4:
            if st.button("🚨 Flag for Audit", use_container_width=True):
                st.error("Provider flagged for full audit. Compliance team notified.")

        # Override reason
        with st.expander("Override AI recommendation (with reason)"):
            override_reason = st.text_area("Reason for override (required for audit trail)", height=80)
            override_action = st.selectbox("Override decision", ["Approve", "Reject", "Escalate"])
            if st.button("Submit Override"):
                if override_reason.strip():
                    st.success(f"Override logged: '{override_action}' — Reason: '{override_reason}'")
                else:
                    st.error("Please enter a reason before submitting override.")


# ══════════════════════════════════════════════════════════════════
# PAGE 3 — BATCH UPLOAD
# ══════════════════════════════════════════════════════════════════
elif page == "📁 Batch Upload":
    st.markdown("## 📁 Batch CSV Upload")
    st.caption("Upload a CSV of providers to score all claims at once")

    if not MODEL_LOADED:
        st.error("Please run `python train_model.py` first.")
        st.stop()

    model, feature_cols, _ = load_model()

    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("**Expected CSV format:** Your file should have a `Provider` column plus any of the feature columns. Missing columns will be filled with 0.")

    # Show expected columns
    with st.expander("View expected column names"):
        st.code("\n".join(["Provider"] + feature_cols))
    st.markdown("</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Upload a CSV with provider data to get fraud risk scores"
    )

    # Demo with the scored providers
    if st.button("📥 Load Demo Data (use training dataset)"):
        df_demo = load_scored_providers()[["Provider"] + [c for c in feature_cols if c in load_scored_providers().columns]].head(50)
        st.session_state["batch_df"] = df_demo
        st.success("Demo data loaded — 50 providers from training dataset")

    if uploaded_file:
        try:
            raw = pd.read_csv(uploaded_file)
            st.session_state["batch_df"] = raw
            st.success(f"✓ Uploaded {len(raw):,} rows, {len(raw.columns)} columns")
        except Exception as e:
            st.error(f"Could not read file: {e}")

    if "batch_df" in st.session_state:
        raw = st.session_state["batch_df"].copy()

        # Fill missing feature columns
        for col in feature_cols:
            if col not in raw.columns:
                raw[col] = 0
        raw[feature_cols] = raw[feature_cols].fillna(0)

        # Score
        with st.spinner("Scoring all providers..."):
            X_batch = raw[feature_cols].values
            proba   = model.predict_proba(X_batch)[:,1]
            scores  = (proba * 100).round(1)
            routings = [score_to_routing(s)[0] for s in scores]

        results = pd.DataFrame({
            "Provider":      raw["Provider"] if "Provider" in raw.columns else [f"P{i}" for i in range(len(raw))],
            "Risk Score":    scores,
            "Fraud Prob %":  (proba * 100).round(2),
            "Routing":       routings,
        })

        # Sort by score desc
        results = results.sort_values("Risk Score", ascending=False).reset_index(drop=True)

        st.divider()
        st.markdown(f"### Scored {len(results):,} Providers")

        # Summary
        col1, col2, col3 = st.columns(3)
        col1.metric("🟢 Auto-Clear",        (results["Routing"]=="Auto-Clear").sum())
        col2.metric("🟡 Soft Review",        (results["Routing"]=="Soft Review").sum())
        col3.metric("🔴 Full Investigation", (results["Routing"]=="Full Investigation").sum())

        # Table
        def colour_routing(val):
            if val == "Auto-Clear":        return "background-color:#D5F5E3; color:#1E8449"
            elif val == "Soft Review":     return "background-color:#FEF9E7; color:#B7770D"
            else:                          return "background-color:#FADBD8; color:#922B21"

        styled_res = results.style.applymap(colour_routing, subset=["Routing"])
        st.dataframe(styled_res, use_container_width=True, height=420)

        # Download
        csv_out = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Scored Results CSV",
            data=csv_out,
            file_name="vericlaim_scored_results.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════
# PAGE 4 — MANAGEMENT DASHBOARD
# ══════════════════════════════════════════════════════════════════
elif page == "📊 Management Dashboard":
    st.markdown("## 📊 Management Dashboard")
    st.caption("Claims operations overview — for Rajesh Khanna, Head of Claims Operations")

    if not MODEL_LOADED:
        st.error("Please run `python train_model.py` first.")
        st.stop()

    df = load_scored_providers()
    _, _, metrics = load_model()

    total     = len(df)
    ac        = (df["RiskScore"] < 30).sum()
    sr        = ((df["RiskScore"] >= 30) & (df["RiskScore"] < 70)).sum()
    fi        = (df["RiskScore"] >= 70).sum()

    # Estimated fraud value (illustrative calculation)
    df["IP_AvgClaimAmt_filled"] = df.get("IP_AvgClaimAmt", pd.Series([25000]*len(df))).fillna(25000)
    df["IP_TotalClaims_filled"] = df.get("IP_TotalClaims", pd.Series([10]*len(df))).fillna(10)
    df["EstClaimValue"] = df["IP_AvgClaimAmt_filled"] * df["IP_TotalClaims_filled"]
    flagged_value = df[df["RiskScore"] >= 70]["EstClaimValue"].sum()
    fraud_savings_est = flagged_value * 0.25  # assume 25% of flagged value is actual fraud caught

    # ── KPI CARDS ─────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Providers Scored",     f"{total:,}")
    k2.metric("🟢 Auto-Cleared",            f"{ac:,}",  f"{ac/total*100:.1f}%")
    k3.metric("🟡 Soft Review",             f"{sr:,}",  f"{sr/total*100:.1f}%")
    k4.metric("🔴 Full Investigation",      f"{fi:,}",  f"{fi/total*100:.1f}%")
    k5.metric("💰 Est. Fraud Under Review", f"₹{fraud_savings_est/1e7:.1f} Cr")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Risk Score Distribution**")
        fig_dist = px.histogram(
            df, x="RiskScore", nbins=30,
            color_discrete_sequence=["#2E86C1"],
            labels={"RiskScore":"Risk Score","count":"Providers"}
        )
        fig_dist.add_vline(x=30, line_dash="dash", line_color="#27AE60")
        fig_dist.add_vline(x=70, line_dash="dash", line_color="#E74C3C")
        fig_dist.update_layout(height=280,margin=dict(l=0,r=0,t=8,b=0),
                               paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_right:
        st.markdown("**Routing Breakdown**")
        fig_pie = go.Figure(go.Pie(
            labels=["Auto-Clear","Soft Review","Full Investigation"],
            values=[ac, sr, fi],
            hole=0.55,
            marker_colors=["#27AE60","#F39C12","#E74C3C"],
        ))
        fig_pie.update_traces(textposition="outside", textinfo="percent+label")
        fig_pie.update_layout(height=280,margin=dict(l=0,r=0,t=8,b=0),
                              showlegend=False,
                              paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ── HIGH RISK PROVIDER TABLE ───────────────────────────────────
    st.markdown("**🔴 High-Risk Providers — Requiring Full Investigation**")
    high_risk = df[df["RiskScore"] >= 70].sort_values("RiskScore", ascending=False).head(20)
    hr_display = pd.DataFrame({
        "Provider ID":      high_risk["Provider"],
        "Risk Score":       high_risk["RiskScore"].round(1),
        "IP Claims":        high_risk.get("IP_TotalClaims", pd.Series([0]*len(high_risk))).fillna(0).astype(int),
        "Avg Claim (₹)":   high_risk.get("IP_AvgClaimAmt", pd.Series([0]*len(high_risk))).fillna(0).round(0),
        "Fraud Label":      high_risk["PotentialFraud"].map({1:"⚠️ Fraud", 0:"✅ Genuine"}),
    })
    st.dataframe(
        hr_display.style.applymap(
            lambda v: "background-color:#FADBD8;color:#922B21;font-weight:600",
            subset=["Risk Score"]
        ),
        use_container_width=True, height=320
    )

    st.divider()

    # ── MODEL PERFORMANCE ─────────────────────────────────────────
    st.markdown("**🤖 Model Performance (XGBoost + SHAP)**")
    mp1, mp2, mp3 = st.columns(3)
    mp1.metric("F1 Score",  metrics["f1"],  help="Harmonic mean of precision and recall on fraud class")
    mp2.metric("ROC-AUC",   metrics["auc"], help="Area under ROC curve — model discrimination ability")
    mp3.metric("Never-Auto-Reject", "✅ Active", help="AI never rejects a claim — all adverse decisions are human-made")

    st.caption("Source: Kaggle Healthcare Provider Fraud Detection dataset (Medicare). Model trained with SMOTE for class imbalance. Retraining cadence: monthly.")

    st.divider()

    # ── EXPORT ────────────────────────────────────────────────────
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        export_csv = df[["Provider","RiskScore","PotentialFraud"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export All Scores (CSV)",
            data=export_csv,
            file_name="vericlaim_all_scores.csv",
            mime="text/csv",
        )
    with col_exp2:
        hr_csv = hr_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export High-Risk Providers (CSV)",
            data=hr_csv,
            file_name="vericlaim_high_risk.csv",
            mime="text/csv",
        )
