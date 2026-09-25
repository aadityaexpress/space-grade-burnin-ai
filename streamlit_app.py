"""
app.py
Interactive QA Mission-Assurance Streamlit Dashboard for Space-Grade IC Burn-In Screening.
Demonstrates:
 - Module A: Dynamic Part Average Testing (DPAT) vs Static Limits
 - Module B: Time-Series Drift Predictor (24h -> 168h) & Early Chamber Rejection
 - Module C: QA Inspector Audit Station with Plain-Language Justifications
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from data_generator import generate_burnin_dataset, split_and_save_data, STATIC_LIMITS, SAFETY_SLOPES
from module_a_outlier_detector import DynamicOutlierDetector
from module_b_drift_predictor import DriftPredictor
from module_c_explainability import QAInspectorExplainer

st.set_page_config(
    page_title="Space-Grade Burn-In Defect Detector",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }
    .subtitle { font-size: 1.1rem; color: #4B5563; margin-bottom: 20px; }
    .metric-card { background-color: #F3F4F6; border-radius: 8px; padding: 15px; border-left: 5px solid #2563EB; }
    .verdict-pass { background-color: #DCFCE7; border: 1px solid #16A34A; color: #166534; padding: 12px; border-radius: 6px; font-weight: 600; }
    .verdict-reject { background-color: #FEE2E2; border: 1px solid #DC2626; color: #991B1B; padding: 12px; border-radius: 6px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_or_generate_data():
    if os.path.exists("burnin_train.csv") and os.path.exists("burnin_test.csv"):
        train_df = pd.read_csv("burnin_train.csv")
        test_df = pd.read_csv("burnin_test.csv")
    else:
        train_df, test_df = split_and_save_data(".")
    return train_df, test_df


@st.cache_resource
def train_models(train_df, k_sigma, safety_slope):
    # Train Module A
    detector = DynamicOutlierDetector(k_sigma=k_sigma, contamination=0.08, cost_ratio_fn_fp=10.0)
    detector.fit(train_df, y_true=train_df["is_defective"])

    # Train Module B
    predictor = DriftPredictor(safety_slope=safety_slope, datasheet_limit=STATIC_LIMITS["I_leak_uA"])
    predictor.fit(train_df)

    # Init Module C
    explainer = QAInspectorExplainer(drift_predictor=predictor)

    return detector, predictor, explainer


# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/color/96/satellite.png", width=70)
st.sidebar.title("Screening Controls")
st.sidebar.markdown("**AEC-Q001 Burn-In Parameters**")

k_sigma = st.sidebar.slider("DPAT Dynamic K-Sigma Limit", min_value=2.0, max_value=5.0, value=3.5, step=0.1)
safety_slope = st.sidebar.slider("Safety Drift Slope (µA/h)", min_value=0.05, max_value=0.30, value=0.15, step=0.01)
static_limit = STATIC_LIMITS["I_leak_uA"]

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Test Standards:**
- MIL-STD-883 Method 1015
- AEC-Q001 DPAT Spec
- Chamber Temp: 125°C
""")

# Load Data and Models
train_df, test_df = load_or_generate_data()
detector, predictor, explainer = train_models(train_df, k_sigma, safety_slope)

# Run Inference on Test Set
_, mod_a_flags, annotated_a = detector.predict_scores(test_df)
pred_168h, early_reject_flags, scored_df = predictor.predict(annotated_a)

# Combined screening decision
combined_flags = np.maximum(scored_df["module_a_flag"].values, scored_df["early_rejection_flag"].values)
scored_df["final_reject"] = combined_flags

# Header
st.markdown('<div class="main-title">🛰️ Space-Grade IC Burn-In Defect & Drift Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predictive Screening & Dynamic Outlier Detection for Environmental Stress Screening (125°C)</div>', unsafe_allow_html=True)

# High Level KPI Bar
col1, col2, col3, col4, col5 = st.columns(5)
total_screened = len(scored_df)
total_defective = int(scored_df["is_defective"].sum())
caught_defects = int(np.sum((scored_df["is_defective"] == 1) & (scored_df["final_reject"] == 1)))
escaped_defects = total_defective - caught_defects
recall_pct = (caught_defects / total_defective) * 100.0 if total_defective > 0 else 100.0
hours_saved = int(scored_df["early_rejection_flag"].sum()) * 144

# Traditional Static Baseline comparison
traditional_escapes = int(np.sum((scored_df["is_defective"] == 1) & (scored_df["val_0h"] <= static_limit) & (scored_df["val_24h"] <= static_limit)))

with col1:
    st.metric("Total ICs Screened", f"{total_screened:,}")
with col2:
    st.metric("Detection Recall", f"{recall_pct:.1f}%", delta=f"{traditional_escapes - escaped_defects} fewer escapes vs static")
with col3:
    st.metric("Escaped Defects", f"{escaped_defects}", delta="-0 is goal", delta_color="inverse")
with col4:
    st.metric("Early Rejections @ 24h", f"{int(scored_df['early_rejection_flag'].sum())}")
with col5:
    st.metric("Chamber Hours Saved", f"{hours_saved:,} hrs")

# Main Content Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Module A: Dynamic Screening (DPAT)",
    "📈 Module B: Time-Series Drift Predictor",
    "🔍 Module C: QA Inspector Audit Station",
    "📋 Lot Telemetry Data Explorer"
])

# ----------------- TAB 1: MODULE A -----------------
with tab1:
    st.subheader("Module A: Dynamic Part Average Testing (DPAT) vs Static Limits")
    st.markdown("""
    **The Problem:** Traditional static screening uses an absolute datasheet limit (e.g., **50 µA**). 
    A part measuring **45 µA** passes static screening. However, if the wafer lot average is **10 µA**, 
    that part is an extreme **17σ anomaly** containing a fatal latent defect!
    """)

    selected_lot = st.selectbox("Select Wafer Lot to Inspect:", sorted(scored_df["lot_id"].unique()))
    lot_data = scored_df[scored_df["lot_id"] == selected_lot]
    lot_stats = detector.lot_stats.get(selected_lot, list(detector.lot_stats.values())[0])

    col_l1, col_l2 = st.columns([1, 1])

    with col_l1:
        # Lot Distribution Histogram with DPAT vs Static Limit
        fig_dist = px.histogram(
            lot_data,
            x="val_0h",
            color="defect_type",
            nbins=40,
            title=f"Lot {selected_lot} Parameter Distribution at t=0h",
            labels={"val_0h": "Leakage Current I_leak (µA)", "defect_type": "Classification"},
            color_discrete_map={
                "normal": "#2563EB",
                "latent_distribution_outlier": "#DC2626",
                "early_drifter": "#F59E0B",
                "gross_failure": "#7C3AED"
            }
        )
        # Add DPAT Limit Line
        fig_dist.add_vline(x=lot_stats["dpat_upper"], line_dash="dash", line_color="orange",
                           annotation_text=f"DPAT Dynamic Limit ({lot_stats['dpat_upper']:.2f} µA)")
        # Add Static Limit Line
        fig_dist.add_vline(x=static_limit, line_dash="solid", line_color="red",
                           annotation_text=f"Static Limit ({static_limit} µA)")
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_l2:
        # Multivariate correlation plot: val_0h vs iddq_0h
        fig_scatter = px.scatter(
            lot_data,
            x="val_0h",
            y="iddq_0h",
            color="module_a_flag",
            title=f"Multivariate Space: Leakage vs Iddq (Lot {selected_lot})",
            labels={"val_0h": "Leakage Current (µA)", "iddq_0h": "Iddq Current (mA)", "module_a_flag": "Flagged Outlier"},
            color_continuous_scale=[[0, "#2563EB"], [1, "#DC2626"]],
            hover_data=["component_id", "defect_type"]
        )
        fig_scatter.add_vline(x=static_limit, line_dash="solid", line_color="red")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Explanation Callout
    latent_in_lot = lot_data[lot_data["defect_type"] == "latent_distribution_outlier"]
    if len(latent_in_lot) > 0:
        example_p = latent_in_lot.iloc[0]
        st.warning(f"""
        ⚠️ **Live In-Spec Latent Anomaly Detected**: 
        Component **{example_p['component_id']}** measured **{example_p['val_0h']:.2f} µA**. 
        - Static Datasheet Limit: **50.0 µA** -> 🟢 **STATIC PASS** (Would have flown on a satellite!)
        - Dynamic DPAT Limit: **{lot_stats['dpat_upper']:.2f} µA** -> 🔴 **DYNAMIC REJECT** (Saved the mission)
        """)

# ----------------- TAB 2: MODULE B -----------------
with tab2:
    st.subheader("Module B: Time-Series Drift Predictor & Early 24h Rejection")
    st.markdown("""
    **The Problem:** Running thermal burn-in chambers for 168 hours is expensive. 
    By forecasting $V_{168h}$ using only $0h$ and $24h$ data, we calculate the **Safety Drift Slope** 
    and eject failing components at **24 hours**, saving **144 hours** per defective part!
    """)

    col_b1, col_b2 = st.columns([1, 1])

    with col_b1:
        # Trajectory forecast curves for sample components
        st.markdown("##### Degradation Trajectories (0h → 24h → Predicted 168h)")
        
        # Pick 3 normal and 3 drifters
        sample_normal = scored_df[scored_df["defect_type"] == "normal"].head(3)
        sample_drifters = scored_df[scored_df["defect_type"] == "early_drifter"].head(3)
        trajectory_samples = pd.concat([sample_normal, sample_drifters])

        fig_traj = go.Figure()
        time_points = [0, 24, 168]

        for _, row in trajectory_samples.iterrows():
            cid = row["component_id"]
            is_drifter = row["defect_type"] == "early_drifter"
            color = "red" if is_drifter else "blue"
            name = f"{cid} (Drifter - Reject)" if is_drifter else f"{cid} (Normal - Pass)"
            
            # Trajectory
            vals = [row["val_0h"], row["val_24h"], row["pred_val_168h"]]
            fig_traj.add_trace(go.Scatter(
                x=time_points,
                y=vals,
                mode="lines+markers",
                name=name,
                line=dict(color=color, dash="dash" if is_drifter else "solid", width=2)
            ))

        # Add Static Limit Horizontal Line
        fig_traj.add_hline(y=static_limit, line_color="darkred", annotation_text="Datasheet Static Limit (50 µA)")
        fig_traj.update_layout(
            xaxis_title="Burn-In Hours (125°C Chamber)",
            yaxis_title="Leakage Current (µA)",
            hovermode="x unified"
        )
        st.plotly_chart(fig_traj, use_container_width=True)

    with col_b2:
        # Forecast Accuracy: Actual 168h vs Predicted 168h
        st.markdown("##### Model Accuracy on Hidden Ground-Truth (168h)")
        fig_acc = px.scatter(
            scored_df,
            x="val_168h",
            y="pred_val_168h",
            color="early_rejection_flag",
            title="Forecasted vs Actual Hidden 168h Values",
            labels={"val_168h": "Actual Ground-Truth 168h (µA)", "pred_val_168h": "Predicted 168h (µA)", "early_rejection_flag": "Early Reject @ 24h"},
            color_continuous_scale=[[0, "#2563EB"], [1, "#DC2626"]]
        )
        fig_acc.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(color="gray", dash="dot"))
        st.plotly_chart(fig_acc, use_container_width=True)

# ----------------- TAB 3: MODULE C -----------------
with tab3:
    st.subheader("Module C: QA Inspector Explainability Station")
    st.markdown("""
    Aerospace mission assurance forbids black-box models. Every screening decision must be fully explainable 
    with engineering root-cause justifications and feature impact breakdown.
    """)

    # Filter selector
    filter_type = st.radio("Filter Components By:", ["All", "Flagged Defects Only", "Normal Passed Only"], horizontal=True)
    if filter_type == "Flagged Defects Only":
        selectable_df = scored_df[scored_df["final_reject"] == 1]
    elif filter_type == "Normal Passed Only":
        selectable_df = scored_df[scored_df["final_reject"] == 0]
    else:
        selectable_df = scored_df

    selected_cid = st.selectbox("Select Component Serial ID to Audit:", selectable_df["component_id"].tolist())

    if selected_cid:
        row = scored_df[scored_df["component_id"] == selected_cid].iloc[0]
        audit_card = explainer.explain_component(row)

        col_audit1, col_audit2 = st.columns([1, 1])

        with col_audit1:
            st.markdown("#### Mission-Assurance Screening Certificate")
            if audit_card["verdict"] == "QUALIFIED FOR SPACE FLIGHT":
                st.markdown(f'<div class="verdict-pass">✅ {audit_card["verdict"]}<br><small>{audit_card["recommended_action"]}</small></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict-reject">❌ {audit_card["verdict"]}<br><small>{audit_card["recommended_action"]}</small></div>', unsafe_allow_html=True)

            st.markdown("##### Engineering Justifications:")
            for j in audit_card["justification"]:
                st.markdown(f"- {j}")

            st.markdown("##### Parametric Verification:")
            m = audit_card["metrics"]
            metric_table = pd.DataFrame([
                {"Parameter": "0h Leakage (µA)", "Value": m["val_0h"], "Criteria": f"Static Limit < {static_limit} µA", "Status": "PASS" if m["static_pass_0h"] else "FAIL"},
                {"Parameter": "0h DPAT Outlier", "Value": "Yes" if m["dpat_flag"] else "No", "Criteria": "Within 3.5σ Lot Normal", "Status": "ANOMALY" if m["dpat_flag"] else "PASS"},
                {"Parameter": "24h Leakage (µA)", "Value": m["val_24h"], "Criteria": f"Static Limit < {static_limit} µA", "Status": "PASS" if m["static_pass_24h"] else "FAIL"},
                {"Parameter": "Predicted 168h Value", "Value": f"{m['predicted_168h']} µA", "Criteria": f"< {static_limit} µA", "Status": "FAIL" if m["predicted_168h"] > static_limit else "PASS"},
                {"Parameter": "Safety Drift Slope", "Value": f"{m['safety_slope']} µA/h", "Criteria": f"< {safety_slope} µA/h", "Status": "EXCEEDED" if m["safety_slope"] > safety_slope else "NORMAL"},
            ])
            st.dataframe(metric_table, hide_index=True, use_container_width=True)

        with col_audit2:
            st.markdown("#### Feature Attribution Impact (SHAP)")
            st.markdown("Breakdown of physical factors driving the model's drift prediction:")
            attr_df = pd.DataFrame(audit_card["feature_attributions"])
            st.dataframe(attr_df, hide_index=True, use_container_width=True)

            # Attribution bar chart
            fig_imp = px.bar(
                attr_df,
                x="importance_weight",
                y="feature",
                orientation="h",
                title="Relative Feature Risk Weight",
                labels={"importance_weight": "Risk Weight", "feature": "Degradation Feature"},
                color="importance_weight",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig_imp, use_container_width=True)

            # Raw Inspector Document
            with st.expander("📄 View Official Signed Text Audit Report"):
                text_rep = explainer.generate_inspector_text_report(audit_card)
                st.code(text_rep)

# ----------------- TAB 4: DATA EXPLORER -----------------
with tab4:
    st.subheader("Wafer Lot Burn-In Telemetry Explorer")
    st.dataframe(
        scored_df[[
            "component_id", "lot_id", "wafer_id", "defect_type",
            "val_0h", "val_24h", "pred_val_168h", "val_168h",
            "dpat_outlier_0h", "early_rejection_flag", "final_reject"
        ]],
        use_container_width=True
    )
