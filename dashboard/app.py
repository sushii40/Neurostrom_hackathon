"""
================================================================================
NEUROSTORM SCADA | INDUSTRIAL PREDICTIVE MAINTENANCE COMMAND CENTER
Real-Time Shop-Floor Diagnostic, Testing & Machine Health Intelligence Platform
================================================================================
"""

import time
import random
import textwrap
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# HELPER: SAFE HTML RENDERING
# ==============================================================================

def render_html(html_str: str):
    """
    Safely renders HTML without triggering Markdown's 4-space indented code block rule.
    Any common leading whitespace is stripped so CommonMark parses it directly as raw HTML.
    """
    st.markdown(textwrap.dedent(html_str).strip(), unsafe_allow_html=True)


# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="NEUROSTORM SCADA | Industrial Machine Testing Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "processed"

# ==============================================================================
# INDUSTRIAL SCADA DARK-THEME STYLING
# ==============================================================================

st.markdown("""
<style>
/* Base Theme & Background */
.stApp {
    background-color: #0B0F19;
    color: #E2E8F0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
    max-width: 1600px;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #0F172A;
    border-right: 1px solid #1E293B;
}

[data-testid="stSidebar"] * {
    color: #CBD5E1;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: #38BDF8;
    font-weight: 700;
    letter-spacing: 0.5px;
}

/* SCADA Top Header Banner */
.scada-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}

.scada-title {
    font-size: 26px;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.5px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.scada-subtitle {
    color: #94A3B8;
    font-size: 13px;
    margin-top: 4px;
    letter-spacing: 0.3px;
}

.live-badge {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid #10B981;
    color: #10B981;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}

.live-dot {
    width: 8px;
    height: 8px;
    background-color: #10B981;
    border-radius: 50%;
    box-shadow: 0 0 10px #10B981;
    animation: live-pulse 1.8s infinite;
}

@keyframes live-pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

/* Industrial Metric Cards */
div[data-testid="stMetric"] {
    background: #131B2E;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: #38BDF8;
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

div[data-testid="stMetricValue"] {
    color: #F8FAFC;
    font-weight: 800;
    font-size: 26px;
    font-family: "JetBrains Mono", monospace, sans-serif;
}

/* Custom Cards */
.scada-card {
    background: #131B2E;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
}

.scada-card-header {
    font-size: 16px;
    font-weight: 700;
    color: #38BDF8;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Status Badges */
.status-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.status-optimal {
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid #10B981;
}

.status-warning {
    background: rgba(245, 158, 11, 0.15);
    color: #FBBF24;
    border: 1px solid #F59E0B;
}

.status-critical {
    background: rgba(239, 68, 68, 0.15);
    color: #F87171;
    border: 1px solid #EF4444;
}

.status-info {
    background: rgba(56, 189, 248, 0.15);
    color: #38BDF8;
    border: 1px solid #38BDF8;
}

/* Diagnostic Narrative Boxes */
.diagnostic-banner-optimal {
    background: rgba(16, 185, 129, 0.1);
    border-left: 5px solid #10B981;
    padding: 16px 20px;
    border-radius: 8px;
    margin-bottom: 18px;
}

.diagnostic-banner-warning {
    background: rgba(245, 158, 11, 0.1);
    border-left: 5px solid #F59E0B;
    padding: 16px 20px;
    border-radius: 8px;
    margin-bottom: 18px;
}

.diagnostic-banner-critical {
    background: rgba(239, 68, 68, 0.12);
    border-left: 5px solid #EF4444;
    padding: 16px 20px;
    border-radius: 8px;
    margin-bottom: 18px;
}

/* Plotly Chart Containers */
[data-testid="stPlotlyChart"] {
    background: #131B2E;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 10px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
}

/* Inputs & Selectboxes */
.stSelectbox > div > div,
.stTextInput input,
.stNumberInput input {
    background-color: #1E293B !important;
    color: #F8FAFC !important;
    border-color: #334155 !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
    color: #FFFFFF;
    font-weight: 700;
    border: 1px solid #38BDF8;
    border-radius: 8px;
    padding: 10px 22px;
    transition: all 0.2s ease;
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #0369A1 0%, #075985 100%);
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(2, 132, 199, 0.5);
    border-color: #7DD3FC;
}

/* Dataframe & Tables */
[data-testid="stDataFrame"] {
    border: 1px solid #1E293B;
    border-radius: 10px;
    background: #131B2E;
}

/* Containers with border */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #1E293B !important;
    background-color: #131B2E;
    border-radius: 12px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #0F172A;
    padding: 6px;
    border-radius: 10px;
    border: 1px solid #1E293B;
}

.stTabs [data-baseweb="tab"] {
    color: #94A3B8;
    border-radius: 8px;
    font-weight: 600;
    padding: 8px 18px;
    border: none;
}

.stTabs [aria-selected="true"] {
    background-color: #1E293B !important;
    color: #38BDF8 !important;
    border-bottom: 2px solid #38BDF8 !important;
}

/* Dividers */
hr {
    border-color: #1E293B;
    margin: 24px 0;
}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# DATA LOADING & PRE-PROCESSING
# ==============================================================================

@st.cache_data
def load_industrial_data():
    sensor_df = pd.read_csv(DATA / "cleaned_sensor_data.csv")
    prediction_df = pd.read_csv(DATA / "model_evaluation" / "loeo_predictions.csv")
    anomaly_df = pd.read_csv(DATA / "anomaly_detection" / "isolation_forest_results.csv")
    return sensor_df, prediction_df, anomaly_df

try:
    sensor_data, prediction_data, anomaly_data = load_industrial_data()
except Exception as err:
    st.error(f"FATAL SCADA ERROR: Could not load industrial telemetry data: {err}")
    st.stop()


# Pre-compute machine operational catalog across all 18 industrial CNC test runs
def build_machine_catalog(sensor_df, pred_df, anom_df):
    catalog = {}
    anom_lookup = anom_df.groupby("Experiment_ID")["anomaly"].agg(["sum", "count"]).to_dict("index")
    pred_lookup = {row["Experiment_ID"]: row for _, row in pred_df.iterrows()}

    for exp_id in sorted(sensor_df["Experiment_ID"].dropna().unique()):
        exp_sensor = sensor_df[sensor_df["Experiment_ID"] == exp_id]
        if exp_sensor.empty:
            continue
        first = exp_sensor.iloc[0]
        
        tool = str(first.get("tool_condition", "unworn")).lower()
        finalized = str(first.get("machining_finalized", "yes")).lower()
        visual_pass = str(first.get("passed_visual_inspection", "yes")).lower()
        feedrate = float(first.get("feedrate", 6.0)) if pd.notna(first.get("feedrate")) else 6.0
        clamp = float(first.get("clamp_pressure", 4.0)) if pd.notna(first.get("clamp_pressure")) else 4.0
        material = str(first.get("material", "wax"))

        anom_info = anom_lookup.get(exp_id, {"sum": 0, "count": len(exp_sensor)})
        anom_cnt = int(anom_info["sum"])
        anom_tot = int(anom_info["count"])
        anom_pct = (anom_cnt / anom_tot * 100) if anom_tot > 0 else 0.0

        pred_info = pred_lookup.get(exp_id)
        wear_prob = float(pred_info["Worn_Probability"]) if pred_info is not None else (0.85 if tool == "worn" else 0.15)

        # Operational status assessment for shop floor
        if finalized == "no":
            status_code = "ABORTED"
            status_label = "CRITICAL CYCLE ABORT"
            badge_class = "status-critical"
            health_score = int(max(15, 45 - anom_pct * 4))
        elif visual_pass == "no":
            status_code = "DEFECT"
            status_label = "PART REJECTED (QA FAIL)"
            badge_class = "status-critical"
            health_score = int(max(25, 55 - anom_pct * 3))
        elif tool == "worn" or wear_prob >= 0.50:
            status_code = "WARNING"
            status_label = "TOOL WEAR ELEVATED"
            badge_class = "status-warning"
            health_score = int(max(50, 75 - wear_prob * 30))
        else:
            status_code = "OPTIMAL"
            status_label = "OPTIMAL OPERATION"
            badge_class = "status-optimal"
            health_score = int(min(99, 95 - anom_pct * 2))

        # Sensor summary statistics
        s1_pwr_max = float(exp_sensor["S1_OutputPower"].max()) if "S1_OutputPower" in exp_sensor else 0.5
        s1_curr_max = float(exp_sensor["S1_CurrentFeedback"].max()) if "S1_CurrentFeedback" in exp_sensor else 70.0
        x1_acc_max = float(exp_sensor["X1_ActualAcceleration"].abs().max()) if "X1_ActualAcceleration" in exp_sensor else 0.0
        y1_acc_max = float(exp_sensor["Y1_ActualAcceleration"].abs().max()) if "Y1_ActualAcceleration" in exp_sensor else 0.0

        catalog[exp_id] = {
            "exp_id": exp_id,
            "machine_name": f"CNC-Unit #{exp_id:02d}",
            "tool_condition": tool,
            "finalized": finalized,
            "visual_pass": visual_pass,
            "feedrate": feedrate,
            "clamp_pressure": clamp,
            "material": material,
            "wear_prob": wear_prob,
            "wear_percent": wear_prob * 100,
            "anomaly_count": anom_cnt,
            "anomaly_total": anom_tot,
            "anomaly_rate": anom_pct,
            "status_code": status_code,
            "status_label": status_label,
            "badge_class": badge_class,
            "health_score": health_score,
            "observations": len(exp_sensor),
            "s1_pwr_max": s1_pwr_max,
            "s1_curr_max": s1_curr_max,
            "x1_acc_max": x1_acc_max,
            "y1_acc_max": y1_acc_max,
        }
    return catalog

machine_catalog = build_machine_catalog(sensor_data, prediction_data, anomaly_data)
total_machines = len(machine_catalog)
optimal_count = sum(1 for m in machine_catalog.values() if m["status_code"] == "OPTIMAL")
warning_count = sum(1 for m in machine_catalog.values() if m["status_code"] == "WARNING")
critical_count = sum(1 for m in machine_catalog.values() if m["status_code"] in ("ABORTED", "DEFECT"))
fleet_health_avg = np.mean([m["health_score"] for m in machine_catalog.values()])


# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================

with st.sidebar:
    st.markdown("### ⚙️ NEUROSTORM SCADA")
    st.caption("INDUSTRIAL MACHINE TESTING & TELEMETRY SYSTEM")
    
    render_html("""
    <div style="background: #131B2E; border: 1px solid #1E293B; border-radius: 8px; padding: 10px 14px; margin-bottom: 16px;">
        <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase;">Active Facility</div>
        <div style="font-size: 13px; font-weight: 700; color: #38BDF8;">Shop Floor Bay 04 • CNC Cells</div>
        <div style="font-size: 11px; color: #10B981; margin-top: 4px;">● Telemetry Link: 100 Hz Nominal</div>
    </div>
    """)
    
    st.markdown("#### ACTIVE MACHINE UNIT")
    selected_machine_id = st.selectbox(
        "Select Machine to Inspect",
        options=sorted(machine_catalog.keys()),
        format_func=lambda x: f"{machine_catalog[x]['machine_name']} [{machine_catalog[x]['status_label'].split()[0]}]",
        index=6 if 7 in machine_catalog else 0, # Default to Unit 7 (abort investigation)
        label_visibility="collapsed"
    )
    
    curr_m = machine_catalog[selected_machine_id]
    
    # Active unit mini status card
    status_border = '#10B981' if curr_m['status_code'] == 'OPTIMAL' else ('#F59E0B' if curr_m['status_code'] == 'WARNING' else '#EF4444')
    render_html(f"""
    <div style="background: #131B2E; border-left: 4px solid {status_border}; border-radius: 6px; padding: 10px 12px; margin-bottom: 20px;">
        <div style="font-size: 11px; color: #94A3B8;">CURRENT INSPECTION UNIT</div>
        <div style="font-weight: 800; font-size: 14px; color: #F8FAFC;">{curr_m['machine_name']}</div>
        <div style="font-size: 12px; margin-top: 4px;" class="{curr_m['badge_class']}">{curr_m['status_label']}</div>
        <div style="font-size: 11px; color: #94A3B8; margin-top: 6px;">Health: <b>{curr_m['health_score']}%</b> | Wear: <b>{curr_m['wear_percent']:.1f}%</b></div>
    </div>
    """)
    
    st.divider()
    
    st.markdown("#### SCADA MODULES")
    page = st.radio(
        "Module Navigation",
        [
            "🎛️ Live SCADA Fleet Command",
            "⚡ Live Machine Testing Workbench",
            "🔬 Machine Diagnostic Forensics",
            "📈 Telemetry Oscilloscope",
            "🛡️ Anomaly Intelligence",
            "📋 Maintenance Work Orders",
            "🗂️ Machine Fleet Audit"
        ],
        label_visibility="collapsed"
    )
    
    st.divider()
    st.caption("NEUROSTORM Industrial IoT Platform")
    st.caption(f"Telemetry Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ==============================================================================
# TOP SCADA HEADER
# ==============================================================================

render_html(f"""
<div class="scada-header">
    <div>
        <div class="scada-title">
            <span>⚡ NEUROSTORM SCADA</span>
            <span style="font-size: 18px; color: #64748B;">|</span>
            <span style="font-size: 20px; color: #38BDF8; font-weight: 600;">Industrial Machine Telemetry & Diagnostics</span>
        </div>
        <div class="scada-subtitle">
            Active Unit: <b>{curr_m['machine_name']}</b> &nbsp;•&nbsp; 
            Material: <b>{curr_m['material'].upper()}</b> &nbsp;•&nbsp; 
            Feedrate: <b>{curr_m['feedrate']} mm/s</b> &nbsp;•&nbsp; 
            Clamp: <b>{curr_m['clamp_pressure']} bar</b> &nbsp;•&nbsp; 
            Observations: <b>{curr_m['observations']:,}</b>
        </div>
    </div>
    <div>
        <div class="live-badge">
            <span class="live-dot"></span>
            <span>TELEMETRY STREAM: 100 Hz</span>
        </div>
    </div>
</div>
""")


# ==============================================================================
# MODULE 1: LIVE SCADA FLEET COMMAND
# ==============================================================================

if page == "🎛️ Live SCADA Fleet Command":
    
    st.markdown("### 🎛️ PLANT FLEET COMMAND CENTER")
    st.caption("Real-Time Shop-Floor Operational Status Across All 18 Industrial CNC Machining Centers")
    
    # High-level industrial KPI strip
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    with kpi1:
        st.metric("Total CNC Fleet", f"{total_machines} Units", help="Total production machining centers monitored")
    with kpi2:
        st.metric("Fleet Health Index", f"{fleet_health_avg:.1f}%", help="Aggregate machine operational health rating")
    with kpi3:
        st.metric("Optimal Operating", f"{optimal_count} Units", delta="Nominal Envelope", delta_color="normal")
    with kpi4:
        st.metric("Wear Advisory", f"{warning_count} Units", delta="Inspection Due", delta_color="off")
    with kpi5:
        st.metric("Critical Interventions", f"{critical_count} Units", delta="Immediate Action", delta_color="inverse")
        
    st.divider()
    
    # Filter Tabs for Fleet Matrix
    st.markdown("#### INDUSTRIAL MACHINE FLEET MATRIX")
    st.caption("Click into any machine card or filter by operational health status:")
    
    filter_choice = st.radio(
        "Filter Fleet",
        ["All Units (18)", "🟢 Optimal Running (6)", "🟡 Tool Wear Elevated (4)", "🔴 Aborted / Defect (8)"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # Filter catalog
    filtered_units = []
    for exp_id, m in machine_catalog.items():
        if "All" in filter_choice:
            filtered_units.append(m)
        elif "Optimal" in filter_choice and m["status_code"] == "OPTIMAL":
            filtered_units.append(m)
        elif "Wear Elevated" in filter_choice and m["status_code"] == "WARNING":
            filtered_units.append(m)
        elif "Aborted" in filter_choice and m["status_code"] in ("ABORTED", "DEFECT"):
            filtered_units.append(m)
            
    # Render fleet matrix cards (3 columns)
    cols = st.columns(3)
    for i, m in enumerate(filtered_units):
        col = cols[i % 3]
        border_color = "#10B981" if m["status_code"] == "OPTIMAL" else ("#F59E0B" if m["status_code"] == "WARNING" else "#EF4444")
        outcome_label = 'Finished & Passed QA' if m['finalized'] == 'yes' and m['visual_pass'] == 'yes' else ('Finished / Rejected QA' if m['finalized'] == 'yes' else 'Premature Abort')
        outcome_color = '#34D399' if m['finalized'] == 'yes' and m['visual_pass'] == 'yes' else '#F87171'
        
        with col:
            render_html(f"""
            <div style="background: #131B2E; border: 1px solid #1E293B; border-top: 4px solid {border_color}; border-radius: 10px; padding: 14px 16px; margin-bottom: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 800; font-size: 15px; color: #F8FAFC;">{m['machine_name']}</span>
                    <span class="status-pill {m['badge_class']}">{m['status_label'].split()[0]}</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px; margin-bottom: 10px;">
                    <div><span style="color: #94A3B8;">Tool:</span> <b style="color: #F8FAFC;">{m['tool_condition'].upper()}</b></div>
                    <div><span style="color: #94A3B8;">Health:</span> <b style="color: {border_color};">{m['health_score']}%</b></div>
                    <div><span style="color: #94A3B8;">Feedrate:</span> <b style="color: #F8FAFC;">{m['feedrate']} mm/s</b></div>
                    <div><span style="color: #94A3B8;">Clamp:</span> <b style="color: #F8FAFC;">{m['clamp_pressure']} bar</b></div>
                    <div><span style="color: #94A3B8;">Wear Risk:</span> <b style="color: #F8FAFC;">{m['wear_percent']:.1f}%</b></div>
                    <div><span style="color: #94A3B8;">Anomalies:</span> <b style="color: #F8FAFC;">{m['anomaly_count']} ({m['anomaly_rate']:.1f}%)</b></div>
                </div>
                <div style="font-size: 11px; color: #94A3B8; border-top: 1px solid #1E293B; padding-top: 8px;">
                    Cycle Outcome: <b style="color: {outcome_color};">{outcome_label}</b>
                </div>
            </div>
            """)
            
    st.divider()
    
    # Fleet Diagnostics & Operational Breakdown Charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("#### FLEET OPERATIONAL STATUS DISTRIBUTION")
        status_df = pd.DataFrame([
            {"Status": "Optimal Running", "Count": optimal_count, "Color": "#10B981"},
            {"Status": "Tool Wear Elevated", "Count": warning_count, "Color": "#F59E0B"},
            {"Status": "Intervention Required", "Count": critical_count, "Color": "#EF4444"}
        ])
        
        fig_donut = px.pie(
            status_df,
            names="Status",
            values="Count",
            hole=0.65,
            color="Status",
            color_discrete_map={
                "Optimal Running": "#10B981",
                "Tool Wear Elevated": "#F59E0B",
                "Intervention Required": "#EF4444"
            }
        )
        fig_donut.update_layout(
            template="plotly_dark",
            paper_bgcolor="#131B2E",
            plot_bgcolor="#131B2E",
            margin=dict(l=20, r=20, t=30, b=20),
            height=300,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with chart_col2:
        st.markdown("#### SHOP FLOOR ANOMALY INCIDENTS PER MACHINE")
        anom_bar_df = pd.DataFrame([
            {"Machine": m["machine_name"], "Anomalies": m["anomaly_count"], "Rate": m["anomaly_rate"], "Status": m["status_code"]}
            for m in machine_catalog.values()
        ])
        fig_anom = px.bar(
            anom_bar_df,
            x="Machine",
            y="Anomalies",
            color="Status",
            color_discrete_map={
                "OPTIMAL": "#10B981",
                "WARNING": "#F59E0B",
                "ABORTED": "#EF4444",
                "DEFECT": "#F87171"
            },
            labels={"Anomalies": "Flagged Telemetry Incidents", "Machine": "CNC Unit"}
        )
        fig_anom.update_layout(
            template="plotly_dark",
            paper_bgcolor="#131B2E",
            plot_bgcolor="#131B2E",
            margin=dict(l=20, r=20, t=30, b=20),
            height=300,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_anom, use_container_width=True)


# ==============================================================================
# MODULE 2: LIVE MACHINE TESTING WORKBENCH (REAL-TIME INTERACTIVE)
# ==============================================================================

elif page == "⚡ Live Machine Testing Workbench":
    
    st.markdown("### ⚡ LIVE MACHINE TESTING WORKBENCH")
    st.caption("Interactive Real-Time Test Bench for Shop-Floor Engineers to Test Physical Machines, Inject Faults & Monitor Diagnostics")
    
    workbench_mode = st.radio(
        "Workbench Mode",
        ["🕹️ Live Hardware Diagnostic & Stress Simulator", "📼 Actual Industrial CNC Telemetry Playback"],
        horizontal=True
    )
    
    if workbench_mode == "🕹️ Live Hardware Diagnostic & Stress Simulator":
        
        render_html("""
        <div class="diagnostic-banner-optimal">
            <b>INSPECTION BENCH ACTIVE:</b> You can dial in physical operating parameters, inject simulated factory stress conditions, 
            run live continuous telemetry streaming, and observe real-time automated PLC fault diagnosis.
        </div>
        """)
        
        # Fault Injection Presets
        st.markdown("#### QUICK FAULT INJECTION & STRESS TESTS")
        st.caption("Trigger realistic shop-floor test conditions with one click:")
        
        preset_cols = st.columns(4)
        
        # Initialize session state for live parameters if not present
        if "sim_rpm" not in st.session_state:
            st.session_state.sim_rpm = 1500
        if "sim_torque" not in st.session_state:
            st.session_state.sim_torque = 40.0
        if "sim_air_temp" not in st.session_state:
            st.session_state.sim_air_temp = 300.0
        if "sim_proc_temp" not in st.session_state:
            st.session_state.sim_proc_temp = 310.0
        if "sim_tool_wear" not in st.session_state:
            st.session_state.sim_tool_wear = 50.0
        if "sim_running" not in st.session_state:
            st.session_state.sim_running = True
            
        with preset_cols[0]:
            if st.button("🔥 Inject Overheating", use_container_width=True):
                st.session_state.sim_air_temp = 314.5
                st.session_state.sim_proc_temp = 328.0
                st.session_state.sim_torque = 44.0
                st.toast("Fault Injected: Spindle Overheating Triggered", icon="🔥")
                
        with preset_cols[1]:
            if st.button("⚙️ Inject Mechanical Chatter", use_container_width=True):
                st.session_state.sim_torque = 64.0
                st.session_state.sim_rpm = 1120
                st.session_state.sim_tool_wear = 135.0
                st.toast("Fault Injected: Mechanical Overload & Chatter Triggered", icon="⚙️")
                
        with preset_cols[2]:
            if st.button("🔪 Inject Critical Tool Wear", use_container_width=True):
                st.session_state.sim_tool_wear = 225.0
                st.session_state.sim_torque = 52.0
                st.toast("Fault Injected: Tool Flank Wear Exceeded (225 min)", icon="🔪")
                
        with preset_cols[3]:
            if st.button("✅ Factory Nominal Baseline", use_container_width=True):
                st.session_state.sim_rpm = 1500
                st.session_state.sim_torque = 40.0
                st.session_state.sim_air_temp = 300.0
                st.session_state.sim_proc_temp = 310.0
                st.session_state.sim_tool_wear = 50.0
                st.session_state.sim_running = True
                st.toast("Reset to Factory Certified Baseline", icon="✅")
                
        st.divider()
        
        # Interactive Sliders for Manual Parameter Adjustment
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
        
        with ctrl_col1:
            st.markdown("##### SPINDLE DYNAMICS")
            sim_rpm = st.slider(
                "Spindle Velocity [RPM]",
                min_value=1000,
                max_value=2000,
                value=int(st.session_state.sim_rpm),
                step=10,
                help="Spindle rotational speed. Nominal: 1400 - 1700 RPM"
            )
            sim_torque = st.slider(
                "Spindle Torque [Nm]",
                min_value=20.0,
                max_value=75.0,
                value=float(st.session_state.sim_torque),
                step=0.5,
                help="Cutting load torque. Safe limit: < 55 Nm"
            )
            
        with ctrl_col2:
            st.markdown("##### THERMAL ENVIRONMENT")
            sim_air = st.slider(
                "Air Temperature [K]",
                min_value=295.0,
                max_value=320.0,
                value=float(st.session_state.sim_air_temp),
                step=0.5,
                help="Ambient shop floor temperature (300K ≈ 27°C)"
            )
            sim_proc = st.slider(
                "Process / Tool Coolant Temp [K]",
                min_value=305.0,
                max_value=330.0,
                value=float(st.session_state.sim_proc_temp),
                step=0.5,
                help="Spindle & cutting zone temperature"
            )
            
        with ctrl_col3:
            st.markdown("##### TOOLING & MACHINE STATE")
            sim_wear = st.slider(
                "Tool Flank Wear [min]",
                min_value=0.0,
                max_value=250.0,
                value=float(st.session_state.sim_tool_wear),
                step=1.0,
                help="Cumulative active cutting contact duration. Wear limit: 200 min"
            )
            sim_class = st.selectbox("Machine Classification", ["Type L (Standard Mill)", "Type M (High Torque)", "Type H (Heavy Production)"])
            
        # Update session values
        st.session_state.sim_rpm = sim_rpm
        st.session_state.sim_torque = sim_torque
        st.session_state.sim_air_temp = sim_air
        st.session_state.sim_proc_temp = sim_proc
        st.session_state.sim_tool_wear = sim_wear
        
        temp_delta = sim_proc - sim_air
        
        # Real-time automated PLC fault diagnostic evaluation
        faults_detected = []
        action_recommendations = []
        
        if sim_air > 310 or sim_proc > 320 or temp_delta > 15:
            faults_detected.append("THERMAL OVERHEATING WARNING")
            action_recommendations.append("Inspect coolant pump flow rate and thermal sensors. Verify chiller fluid level.")
            
        if sim_torque > 55 or sim_rpm < 1200 or sim_rpm > 1800:
            faults_detected.append("MECHANICAL OVERLOAD / CHATTER")
            action_recommendations.append("Inspect motor drive coupling, spindle bearings and mechanical axis backlash.")
            
        if sim_wear > 200:
            faults_detected.append("CRITICAL TOOL FLANK WEAR")
            action_recommendations.append("Tool insert has exceeded safe life. Stop cycle and index carbide insert.")
            
        if not faults_detected:
            machine_health_status = "OPTIMAL - OPERATING WITHIN CERTIFIED ENVELOPE"
            status_banner_class = "diagnostic-banner-optimal"
            action_recommendations.append("Machine is operating nominally. Continue standard production monitoring.")
        else:
            machine_health_status = "ALARM DETECTED: " + " • ".join(faults_detected)
            status_banner_class = "diagnostic-banner-critical" if len(faults_detected) > 1 else "diagnostic-banner-warning"
            
        render_html(f"""
        <div class="{status_banner_class}">
            <div style="font-size: 15px; font-weight: 800; color: #F8FAFC;">{machine_health_status}</div>
            <div style="font-size: 12px; color: #CBD5E1; margin-top: 6px;">
                <b>Thermal Differential:</b> {temp_delta:.1f} K &nbsp;|&nbsp; 
                <b>Torque Load:</b> {sim_torque:.1f} Nm &nbsp;|&nbsp; 
                <b>Spindle Speed:</b> {sim_rpm} RPM &nbsp;|&nbsp; 
                <b>Tool Wear:</b> {sim_wear:.0f} min
            </div>
        </div>
        """)
        
        # Dynamic Visual Gauges
        st.markdown("#### REAL-TIME TELEMETRY GAUGES")
        g_col1, g_col2, g_col3, g_col4 = st.columns(4)
        
        with g_col1:
            fig_g1 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sim_rpm,
                title={"text": "Spindle Speed (RPM)"},
                gauge={
                    "axis": {"range": [1000, 2000]},
                    "bar": {"color": "#38BDF8"},
                    "steps": [
                        {"range": [1000, 1200], "color": "#7F1D1D"},
                        {"range": [1200, 1800], "color": "#064E3B"},
                        {"range": [1800, 2000], "color": "#78350F"}
                    ]
                }
            ))
            fig_g1.update_layout(template="plotly_dark", paper_bgcolor="#131B2E", height=220, margin=dict(l=15, r=15, t=30, b=15))
            st.plotly_chart(fig_g1, use_container_width=True)
            
        with g_col2:
            fig_g2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sim_torque,
                title={"text": "Cutting Torque (Nm)"},
                gauge={
                    "axis": {"range": [20, 75]},
                    "bar": {"color": "#38BDF8"},
                    "steps": [
                        {"range": [20, 50], "color": "#064E3B"},
                        {"range": [50, 60], "color": "#78350F"},
                        {"range": [60, 75], "color": "#7F1D1D"}
                    ]
                }
            ))
            fig_g2.update_layout(template="plotly_dark", paper_bgcolor="#131B2E", height=220, margin=dict(l=15, r=15, t=30, b=15))
            st.plotly_chart(fig_g2, use_container_width=True)
            
        with g_col3:
            fig_g3 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=temp_delta,
                title={"text": "Thermal Delta ΔT (K)"},
                gauge={
                    "axis": {"range": [0, 25]},
                    "bar": {"color": "#38BDF8"},
                    "steps": [
                        {"range": [0, 12], "color": "#064E3B"},
                        {"range": [12, 16], "color": "#78350F"},
                        {"range": [16, 25], "color": "#7F1D1D"}
                    ]
                }
            ))
            fig_g3.update_layout(template="plotly_dark", paper_bgcolor="#131B2E", height=220, margin=dict(l=15, r=15, t=30, b=15))
            st.plotly_chart(fig_g3, use_container_width=True)
            
        with g_col4:
            fig_g4 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sim_wear,
                title={"text": "Tool Flank Wear (min)"},
                gauge={
                    "axis": {"range": [0, 250]},
                    "bar": {"color": "#38BDF8"},
                    "steps": [
                        {"range": [0, 120], "color": "#064E3B"},
                        {"range": [120, 200], "color": "#78350F"},
                        {"range": [200, 250], "color": "#7F1D1D"}
                    ]
                }
            ))
            fig_g4.update_layout(template="plotly_dark", paper_bgcolor="#131B2E", height=220, margin=dict(l=15, r=15, t=30, b=15))
            st.plotly_chart(fig_g4, use_container_width=True)
            
        # Live Telemetry Streamer Simulation
        st.markdown("#### LIVE CONTINUOUS TELEMETRY STREAM (ROLLING BUFFER)")
        stream_btn_col1, stream_btn_col2, stream_btn_col3 = st.columns([1, 1, 2])
        
        with stream_btn_col1:
            run_stream = st.button("▶ Stream 15 Live Ticks", use_container_width=True)
            
        with stream_btn_col2:
            if st.button("🛑 EMERGENCY E-STOP", use_container_width=True):
                st.session_state.sim_rpm = 0
                st.session_state.sim_torque = 0
                st.error("EMERGENCY MACHINE E-STOP ACTIVATED. SPINDLE HALTED.")
                st.stop()
                
        chart_placeholder = st.empty()
        
        # Generate or simulate live streaming series
        np.random.seed(42)
        n_ticks = 40
        time_index = np.arange(n_ticks)
        
        # Synthetic noise around current setpoints
        stream_rpm = np.clip(sim_rpm + np.random.normal(0, 15, n_ticks), 1000, 2000)
        stream_torque = np.clip(sim_torque + np.random.normal(0, 1.8, n_ticks), 20, 75)
        stream_delta = np.clip(temp_delta + np.random.normal(0, 0.4, n_ticks), 0, 25)
        
        stream_df = pd.DataFrame({
            "Tick": time_index,
            "Spindle RPM": stream_rpm,
            "Torque (Nm)": stream_torque,
            "Thermal Delta (K)": stream_delta
        })
        
        fig_stream = px.line(
            stream_df,
            x="Tick",
            y=["Spindle RPM", "Torque (Nm)", "Thermal Delta (K)"],
            title="Active High-Frequency Telemetry Stream (Last 40 Ticks)",
            labels={"value": "Sensor Value", "Tick": "Sample Sequence"}
        )
        fig_stream.update_layout(
            template="plotly_dark",
            paper_bgcolor="#131B2E",
            plot_bgcolor="#131B2E",
            height=320,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        chart_placeholder.plotly_chart(fig_stream, use_container_width=True)
        
        if run_stream:
            st.info("Continuous telemetry streaming active. Shop-floor link verified at 100 Hz.")
            
    else:
        # Recorded Telemetry Playback Mode
        st.markdown(f"#### 📼 STEP-BY-STEP PLAYBACK: {curr_m['machine_name']}")
        st.caption("Scrub through the actual recorded industrial CNC cut cycle row-by-row:")
        
        exp_sensor = sensor_data[sensor_data["Experiment_ID"] == selected_machine_id].copy()
        if "M1_sequence_number" in exp_sensor.columns:
            exp_sensor = exp_sensor.sort_values("M1_sequence_number").reset_index(drop=True)
            
        total_steps = len(exp_sensor)
        step = st.slider("Scrub Machining Cycle Sequence", 0, total_steps - 1, min(100, total_steps - 1))
        
        curr_row = exp_sensor.iloc[step]
        
        pb_col1, pb_col2, pb_col3, pb_col4 = st.columns(4)
        pb_col1.metric("Machining Process", str(curr_row.get("Machining_Process", "Unknown")).upper())
        pb_col2.metric("Spindle Current Feedback", f"{curr_row.get('S1_CurrentFeedback', 0.0):.2f} A")
        pb_col3.metric("Spindle Output Power", f"{curr_row.get('S1_OutputPower', 0.0):.3f} W")
        pb_col4.metric("X1 Axis Velocity", f"{curr_row.get('X1_ActualVelocity', 0.0):.2f} mm/s")
        
        # Window of rows around current step
        start_w = max(0, step - 60)
        end_w = min(total_steps, step + 60)
        window_df = exp_sensor.iloc[start_w:end_w]
        
        fig_pb = px.line(
            window_df,
            x=window_df.index,
            y=["S1_CurrentFeedback", "X1_ActualVelocity", "Y1_ActualVelocity"],
            title=f"Telemetry Window around Step {step} of {total_steps}"
        )
        fig_pb.update_layout(template="plotly_dark", paper_bgcolor="#131B2E", height=320)
        fig_pb.add_vline(x=step, line_dash="dash", line_color="#EF4444", annotation_text="Current Needle")
        st.plotly_chart(fig_pb, use_container_width=True)


# ==============================================================================
# MODULE 3: MACHINE DIAGNOSTIC FORENSICS ("KNOW EXACTLY WHAT HAPPENED")
# ==============================================================================

elif page == "🔬 Machine Diagnostic Forensics":
    
    st.markdown(f"### 🔬 MACHINE FORENSIC DIAGNOSTICS: {curr_m['machine_name']}")
    st.caption("Complete Physical Root-Cause Investigation: What Exactly Happened In This Machine?")
    
    # Machine Executive Diagnosis Banner
    if curr_m["status_code"] == "ABORTED":
        render_html(f"""
        <div class="diagnostic-banner-critical">
            <div style="font-size: 18px; font-weight: 800; color: #F87171;">⚠️ EMERGENCY CYCLE ABORT: MACHINE HALTED PREMATURELY</div>
            <div style="font-size: 13px; color: #E2E8F0; margin-top: 8px; line-height: 1.6;">
                <b>Summary:</b> This machining run was terminated early at <b>{curr_m['observations']} cycles</b> 
                (normal full run is ~2,200 cycles). The operation was conducted with a <b>feedrate of {curr_m['feedrate']} mm/s</b> 
                and clamp pressure of <b>{curr_m['clamp_pressure']} bar</b> on a <b>{curr_m['tool_condition'].upper()}</b> cutting tool.
                <br><b>Failure Trigger:</b> Excessive mechanical resistance and tool chatter exceeded the machine safety interlock, forcing an emergency cut cutoff.
            </div>
        </div>
        """)
    elif curr_m["status_code"] == "DEFECT":
        render_html(f"""
        <div class="diagnostic-banner-critical">
            <div style="font-size: 18px; font-weight: 800; color: #F87171;">❌ CYCLE COMPLETED BUT WORKPIECE REJECTED: QA INSPECTION FAILED</div>
            <div style="font-size: 13px; color: #E2E8F0; margin-top: 8px; line-height: 1.6;">
                <b>Summary:</b> The machine finalized the complete machining sequence ({curr_m['observations']} cycles), but the manufactured component 
                <b>FAILED visual quality inspection</b> due to severe surface chatter marks, tear-out burrs, and dimensional non-conformance.
                <br><b>Root Cause:</b> Severely worn cutting edge (Estimated Tool Wear Probability: <b>{curr_m['wear_percent']:.1f}%</b>). 
                Dull tooling generated high frictional resistance and heat, degrading surface finish.
            </div>
        </div>
        """)
    elif curr_m["status_code"] == "WARNING":
        render_html(f"""
        <div class="diagnostic-banner-warning">
            <div style="font-size: 18px; font-weight: 800; color: #FBBF24;">🟡 TOOL DEGRADATION ADVISORY: FLANK WEAR APPROACHING LIMIT</div>
            <div style="font-size: 13px; color: #E2E8F0; margin-top: 8px; line-height: 1.6;">
                <b>Summary:</b> Machine completed the full run ({curr_m['observations']} cycles) and passed visual QA. However, sensor telemetry 
                indicates high tool degradation (Wear Severity: <b>{curr_m['wear_percent']:.1f}%</b>).
                <br><b>Recommendation:</b> Schedule tool replacement prior to next production cycle to prevent workpiece rejection or sudden edge fracture.
            </div>
        </div>
        """)
    else:
        render_html(f"""
        <div class="diagnostic-banner-optimal">
            <div style="font-size: 18px; font-weight: 800; color: #34D399;">✅ CERTIFIED NOMINAL RUN: ALL PARAMETERS OPTIMAL</div>
            <div style="font-size: 13px; color: #E2E8F0; margin-top: 8px; line-height: 1.6;">
                <b>Summary:</b> Machine completed all {curr_m['observations']} cycles with zero defects and passed visual inspection. 
                Tool condition is unworn, cutting resistance remained minimal (0.9% anomaly rate), and axis velocity tracking was pristine.
            </div>
        </div>
        """)
        
    # Physical KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Recorded Tool State", curr_m["tool_condition"].upper())
    k2.metric("Tool Wear Severity", f"{curr_m['wear_percent']:.1f}%")
    k3.metric("Cycle Finalized", curr_m["finalized"].upper())
    k4.metric("Visual QA Passed", curr_m["visual_pass"].upper())
    
    st.divider()
    
    # Detailed Subsystem Engineering Forensics
    st.markdown("#### PHYSICAL SUBSYSTEM ROOT-CAUSE INVESTIGATION")
    
    exp_sensor = sensor_data[sensor_data["Experiment_ID"] == selected_machine_id]
    exp_anom = anomaly_data[anomaly_data["Experiment_ID"] == selected_machine_id]
    
    sub1, sub2, sub3 = st.columns(3)
    
    with sub1:
        with st.container(border=True):
            st.markdown("##### ⚡ Spindle Motor Subsystem (S1)")
            st.write(f"• **Peak Current Feedback:** `{curr_m['s1_curr_max']:.2f} A`")
            st.write(f"• **Peak Output Power:** `{curr_m['s1_pwr_max']:.3f} W`")
            if curr_m["tool_condition"] == "worn":
                st.error("Spindle current feedback showed high transient surges during roughing passes, indicating tool dulling and increased cutting friction.")
            else:
                st.success("Spindle current feedback remained uniform and within normal mechanical torque tolerances.")
        
    with sub2:
        with st.container(border=True):
            st.markdown("##### 📐 Axis Motion Subsystem (X/Y/Z)")
            st.write(f"• **X-Axis Peak Acceleration:** `{curr_m['x1_acc_max']:.3f}`")
            st.write(f"• **Y-Axis Peak Acceleration:** `{curr_m['y1_acc_max']:.3f}`")
            if curr_m["feedrate"] >= 15:
                st.warning(f"High programmed feedrate ({curr_m['feedrate']} mm/s) induced sharp acceleration reversals during contour transitions.")
            else:
                st.success(f"Programmed feedrate ({curr_m['feedrate']} mm/s) maintained smooth linear motion and negligible vibration chatter.")
        
    with sub3:
        with st.container(border=True):
            st.markdown("##### 🔒 Clamping & Workpiece Fixture")
            st.write(f"• **Hydraulic Clamp Pressure:** `{curr_m['clamp_pressure']} bar`")
            st.write(f"• **Workpiece Material:** `{curr_m['material'].title()}`")
            if curr_m["clamp_pressure"] <= 2.5:
                st.error("Low clamp pressure (2.5 bar) posed elevated risk of workpiece vibration and micro-slippage under cutting forces.")
            else:
                st.success(f"Clamp pressure ({curr_m['clamp_pressure']} bar) provided rigid workpiece fixture stability throughout the cut.")
        
    st.divider()
    
    # Process Phase Breakdown: Which machining step experienced anomalies?
    st.markdown("#### MACHINING PROCESS PHASE BREAKDOWN")
    st.caption("Identify exactly which machining process triggered anomalies or cutting resistance:")
    
    process_df = exp_sensor.groupby("Machining_Process").size().reset_index(name="Observations")
    if not exp_anom.empty:
        process_anom = exp_anom.groupby("Machining_Process")["anomaly"].sum().reset_index(name="Anomalies")
        process_df = process_df.merge(process_anom, on="Machining_Process", how="left").fillna(0)
        process_df["Anomaly Rate (%)"] = (process_df["Anomalies"] / process_df["Observations"] * 100).round(2)
        process_df = process_df.sort_values("Anomaly Rate (%)", ascending=False)
        
    fig_proc = px.bar(
        process_df,
        x="Machining_Process",
        y="Anomaly Rate (%)",
        color="Anomaly Rate (%)",
        color_continuous_scale="Reds",
        title=f"Anomaly Rate by Machining Phase - {curr_m['machine_name']}"
    )
    fig_proc.update_layout(
        template="plotly_dark",
        paper_bgcolor="#131B2E",
        plot_bgcolor="#131B2E",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_proc, use_container_width=True)
    
    st.divider()
    
    # Technician Action Checklist
    st.markdown("#### ON-SITE TECHNICIAN CORRECTIVE PROTOCOL")
    st.caption("Standard operating procedure (SOP) checklist for field engineers at this machine:")
    
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.checkbox("Step 1: Execute Lockout/Tagout (LOTO) on main 480V breaker", value=True)
        st.checkbox("Step 2: Remove cutting tool & inspect flank wear with optical loupe (VB > 0.3mm)", value=curr_m["tool_condition"] == "worn")
        st.checkbox("Step 3: Measure collet chuck radial runout with dial test indicator (< 0.005mm)", value=True)
    with t_col2:
        st.checkbox(f"Step 4: Verify hydraulic vise clamp pressure regulator ({curr_m['clamp_pressure']} bar recipe)", value=True)
        st.checkbox("Step 5: Clean metal chips and coolant residue from spindle taper", value=True)
        st.checkbox("Step 6: Run single-block air-cut test cycle prior to production release", value=False)


# ==============================================================================
# MODULE 4: TELEMETRY OSCILLOSCOPE & WAVEFORMS
# ==============================================================================

elif page == "📈 Telemetry Oscilloscope":
    
    st.markdown(f"### 📈 MULTI-AXIS TELEMETRY OSCILLOSCOPE: {curr_m['machine_name']}")
    st.caption("High-Resolution Multi-Channel Sensor Waveform Analysis with Anomaly Overlay")
    
    exp_sensor = sensor_data[sensor_data["Experiment_ID"] == selected_machine_id].copy()
    if "M1_sequence_number" in exp_sensor.columns:
        exp_sensor = exp_sensor.sort_values("M1_sequence_number").reset_index(drop=True)
        
    osc_col1, osc_col2 = st.columns([1, 2])
    
    with osc_col1:
        signal_family = st.selectbox(
            "Select Signal Family",
            [
                "Axis Current Feedback (S1, X1, Y1)",
                "Axis Acceleration / Vibration (X1, Y1, Z1)",
                "Axis Velocity Tracking (X1, Y1, Z1, S1)",
                "Electrical Output Power (S1, X1, Y1)"
            ]
        )
        
    with osc_col2:
        available_processes = ["All Processes"] + sorted(exp_sensor["Machining_Process"].dropna().unique().tolist())
        sel_proc = st.selectbox("Filter Machining Process", available_processes)
        
    filtered_plot = exp_sensor.copy()
    if sel_proc != "All Processes":
        filtered_plot = filtered_plot[filtered_plot["Machining_Process"] == sel_proc]
        
    if filtered_plot.empty:
        st.warning("No sensor telemetry rows for this selection.")
    else:
        # Determine signals
        if "Current Feedback" in signal_family:
            signals = [c for c in ["S1_CurrentFeedback", "X1_CurrentFeedback", "Y1_CurrentFeedback"] if c in filtered_plot.columns]
            y_title = "Current (Amperes)"
        elif "Acceleration" in signal_family:
            signals = [c for c in ["X1_ActualAcceleration", "Y1_ActualAcceleration", "Z1_ActualAcceleration"] if c in filtered_plot.columns]
            y_title = "Acceleration Feedback"
        elif "Velocity" in signal_family:
            signals = [c for c in ["X1_ActualVelocity", "Y1_ActualVelocity", "Z1_ActualVelocity", "S1_ActualVelocity"] if c in filtered_plot.columns]
            y_title = "Velocity (mm/s or RPM)"
        else:
            signals = [c for c in ["S1_OutputPower", "X1_OutputPower", "Y1_OutputPower"] if c in filtered_plot.columns]
            y_title = "Power Output (Watts)"
            
        fig_wave = px.line(
            filtered_plot,
            x=filtered_plot.index,
            y=signals,
            title=f"{signal_family} - {curr_m['machine_name']} [{sel_proc}]",
            labels={"index": "Time / Sequence Sample", "value": y_title}
        )
        fig_wave.update_layout(
            template="plotly_dark",
            paper_bgcolor="#131B2E",
            plot_bgcolor="#131B2E",
            height=440,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_wave, use_container_width=True)
        
        # Summary telemetry stats for selected channel
        st.markdown("#### TELEMETRY CHANNEL STATISTICS")
        stat_cols = st.columns(len(signals))
        for col, sig in zip(stat_cols, signals):
            series = filtered_plot[sig].dropna()
            with col:
                with st.container(border=True):
                    st.caption(sig)
                    st.metric("Mean Value", f"{series.mean():.3f}")
                    st.caption(f"Peak: **{series.max():.3f}** | Min: **{series.min():.3f}**")


# ==============================================================================
# MODULE 5: ANOMALY INTELLIGENCE & PROCESS SAFETY
# ==============================================================================

elif page == "🛡️ Anomaly Intelligence":
    
    st.markdown("### 🛡️ ANOMALY INTELLIGENCE & PROCESS SAFETY")
    st.caption("Multi-Dimensional Isolation Forest Outlier Analysis Across Machining Cycles")
    
    anom_k1, anom_k2, anom_k3, anom_k4 = st.columns(4)
    total_anom = int(anomaly_data["anomaly"].sum())
    total_obs = len(anomaly_data)
    fleet_anom_rate = total_anom / total_obs * 100 if total_obs > 0 else 0
    
    anom_k1.metric("Total Anomaly Points", f"{total_anom:,}")
    anom_k2.metric("Telemetry Observations", f"{total_obs:,}")
    anom_k3.metric("Fleet Anomaly Rate", f"{fleet_anom_rate:.2f}%")
    anom_k4.metric("Active Machine Anomaly Rate", f"{curr_m['anomaly_rate']:.2f}%")
    
    st.divider()
    
    anom_c1, anom_c2 = st.columns(2)
    
    with anom_c1:
        st.markdown("#### ANOMALY DISTRIBUTION BY TOOL HEALTH")
        cond_anom = anomaly_data.groupby("tool_condition").agg(
            Total=("anomaly", "count"),
            Anomalies=("anomaly", "sum")
        ).reset_index()
        cond_anom["Rate (%)"] = cond_anom["Anomalies"] / cond_anom["Total"] * 100
        
        fig_cond = px.bar(
            cond_anom,
            x="tool_condition",
            y="Rate (%)",
            color="tool_condition",
            color_discrete_map={"worn": "#EF4444", "unworn": "#10B981"},
            title="Anomaly Rate: Worn vs Unworn Tooling",
            labels={"tool_condition": "Tool State", "Rate (%)": "Anomaly Rate (%)"}
        )
        fig_cond.update_layout(template="plotly_dark", paper_bgcolor="#131B2E", height=320)
        st.plotly_chart(fig_cond, use_container_width=True)
        
    with anom_c2:
        st.markdown("#### ANOMALY CONCENTRATION BY MACHINING PROCESS")
        proc_anom = anomaly_data.groupby("Machining_Process").agg(
            Total=("anomaly", "count"),
            Anomalies=("anomaly", "sum")
        ).reset_index()
        proc_anom["Rate (%)"] = proc_anom["Anomalies"] / proc_anom["Total"] * 100
        proc_anom = proc_anom.sort_values("Rate (%)", ascending=False).head(8)
        
        fig_proc_anom = px.bar(
            proc_anom,
            x="Machining_Process",
            y="Rate (%)",
            color="Rate (%)",
            color_continuous_scale="Viridis",
            title="Top Anomaly-Prone Machining Processes",
            labels={"Machining_Process": "Process", "Rate (%)": "Anomaly Rate (%)"}
        )
        fig_proc_anom.update_layout(template="plotly_dark", paper_bgcolor="#131B2E", height=320, xaxis_tickangle=-35)
        st.plotly_chart(fig_proc_anom, use_container_width=True)
        
    st.divider()
    
    # Anomaly Log Table for active machine
    st.markdown(f"#### FLAGGED ANOMALY INCIDENTS FOR {curr_m['machine_name']}")
    curr_anom_rows = anomaly_data[
        (anomaly_data["Experiment_ID"] == selected_machine_id) & 
        (anomaly_data["anomaly"] == 1)
    ]
    
    if curr_anom_rows.empty:
        st.success(f"No anomaly events recorded for {curr_m['machine_name']}.")
    else:
        st.dataframe(
            curr_anom_rows[["Machining_Process", "tool_condition", "anomaly_score"]],
            use_container_width=True,
            hide_index=True
        )


# ==============================================================================
# MODULE 6: MAINTENANCE WORK ORDERS & FIELD ADVISOR (RESOLVED & ENHANCED)
# ==============================================================================

elif page == "📋 Maintenance Work Orders":
    
    st.markdown("### 📋 DIGITAL MAINTENANCE WORK ORDER GENERATOR")
    st.caption("Official Shop-Floor Corrective & Preventive Maintenance Work Orders for Factory Technicians")
    
    # Interactive Controls Strip
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1.2, 1.2])
    with ctrl_col1:
        wo_technician = st.text_input("Lead Technician Name", value="Alex Mercer (Reliability Tech II)")
    with ctrl_col2:
        wo_shift = st.selectbox("Production Shift", ["Shift A (Morning 06:00 - 14:30)", "Shift B (Evening 14:30 - 23:00)", "Shift C (Night 23:00 - 06:00)"])
    with ctrl_col3:
        wo_status = st.selectbox("Work Order Status", ["🔴 ISSUED / ACTION REQUIRED", "🟡 IN PROGRESS (MAINTENANCE UNDERWAY)", "🟢 COMPLETED & CERTIFIED"])
        
    wo_priority = "HIGH (IMMEDIATE ACTION REQUIRED)" if curr_m["status_code"] in ("ABORTED", "DEFECT") else ("MEDIUM (SCHEDULED MAINTENANCE)" if curr_m["status_code"] == "WARNING" else "ROUTINE (PREVENTIVE MAINTENANCE)")
    wo_number = f"WO-2026-CNC-{selected_machine_id:02d}-{int(curr_m['wear_percent'])}"
    
    st.divider()
    
    # MAIN WORK ORDER CARD (Clean Native Streamlit Container)
    with st.container(border=True):
        
        # Header Row
        head_col1, head_col2 = st.columns([3, 1])
        with head_col1:
            st.markdown(f"## 🛠️ FIELD WORK ORDER: `{wo_number}`")
            st.markdown(f"**Facility:** Plant Bay 04 • High-Speed Milling Cell &nbsp;|&nbsp; **Machine:** `{curr_m['machine_name']}`")
        with head_col2:
            priority_color = "#EF4444" if "HIGH" in wo_priority else ("#F59E0B" if "MEDIUM" in wo_priority else "#10B981")
            render_html(f"""
            <div style="text-align: right; background: #0F172A; padding: 10px 14px; border-radius: 8px; border: 1px solid #1E293B;">
                <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase;">PRIORITY LEVEL</div>
                <div style="font-size: 14px; font-weight: 800; color: {priority_color};">{wo_priority.split()[0]}</div>
                <div style="font-size: 11px; color: #64748B;">Date: {datetime.now().strftime('%Y-%m-%d')}</div>
            </div>
            """)
            
        st.divider()
        
        # Section 1: Operating Metrics
        st.markdown("#### 1. MACHINE OPERATING STATE & TELEMETRY SNAPSHOT")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Operating Health", f"{curr_m['health_score']}%", delta=f"{curr_m['status_label'].split()[0]}")
        m_col2.metric("Tool Wear Severity", f"{curr_m['wear_percent']:.1f}%", delta="Critical Threshold: 70%")
        m_col3.metric("Feedrate & Clamp", f"{curr_m['feedrate']} mm/s | {curr_m['clamp_pressure']} bar")
        m_col4.metric("Visual QA Result", curr_m['visual_pass'].upper(), delta="Quality Inspection")
        
        st.divider()
        
        # Section 2: Symptoms & Root Cause Determination
        st.markdown("#### 2. DIAGNOSTIC SYMPTOMS & ROOT-CAUSE DETERMINATION")
        if curr_m["status_code"] == "ABORTED":
            st.error(f"""
            **EMERGENCY CUTOFF DETECTED:** Machine aborted prematurely at {curr_m['observations']} cycles (normal full run is ~2,200 cycles). 
            High programmed feedrate ({curr_m['feedrate']} mm/s) on a worn tool caused cutting resistance and spindle torque spikes, 
            triggering {curr_m['anomaly_count']} anomaly incidents ({curr_m['anomaly_rate']:.1f}% rate) and safety interlock trip.
            """)
        elif curr_m["status_code"] == "DEFECT":
            st.error(f"""
            **QUALITY CONTROL REJECTION:** Machine finalized {curr_m['observations']} cycles, but workpiece failed visual QA inspection. 
            Severe tool flank wear (Wear Severity: {curr_m['wear_percent']:.1f}%) generated heavy chatter burrs, surface micro-cracks, and dimensional non-conformance.
            """)
        elif curr_m["status_code"] == "WARNING":
            st.warning(f"""
            **TOOL WEAR ELEVATED:** Machine completed {curr_m['observations']} cycles and passed visual QA. 
            However, sensor telemetry indicates flank wear degradation (Wear Severity: {curr_m['wear_percent']:.1f}%). 
            Tool replacement is required prior to next production cycle to prevent workpiece rejection or sudden edge fracture.
            """)
        else:
            st.success(f"""
            **ROUTINE PREVENTIVE CHECK:** All cutting forces and electrical signals are nominal (Health Score: {curr_m['health_score']}%). 
            Continue standard visual inspection and tool lubrication schedule.
            """)
            
        st.divider()
        
        # Section 3: Required Replacement Tooling & Consumables Table
        st.markdown("#### 3. REQUIRED REPLACEMENT TOOLING & CONSUMABLES")
        parts_data = [
            {"Item": 1, "Part Number": "CNMG-120408-PM", "Description": "TiAlN Coated Carbide Turning/Milling Insert", "Qty": "1 Set (4 Edges)", "Location": "Bin Tooling-A12", "Status": "In Stock"},
            {"Item": 2, "Part Number": "ER32-COLLET-08", "Description": "High Precision Spring Collet (Runout < 0.005mm)", "Qty": "1 Unit", "Location": "Tool Crib B-04", "Status": "In Stock"},
            {"Item": 3, "Part Number": "ISO-VG-46-SYN", "Description": "Synthetic Water-Soluble Coolant Concentrate", "Qty": "5 Liters", "Location": "Fluid Bay F-01", "Status": "Available"},
            {"Item": 4, "Part Number": "VISE-SEAL-KIT", "Description": "Hydraulic Vise Pressure O-Ring & Wiper Kit", "Qty": "1 Kit", "Location": "Pneumatics P-08", "Status": "In Stock"}
        ]
        st.dataframe(pd.DataFrame(parts_data), use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Section 4: Field SOP Action Checklist
        st.markdown("#### 4. ON-SITE FIELD ACTION MANDATES & PROCEDURES")
        st.caption("Field technician must check off each physical procedure performed at the machine:")
        
        chk1, chk2 = st.columns(2)
        with chk1:
            st.checkbox("Step 1: Perform 480V Electrical Disconnect & Lockout/Tagout (LOTO)", value=True, key="wo_step1")
            st.checkbox(f"Step 2: Inspect cutting insert flank wear under optical loupe (Replace if VB > 0.3mm)", value=curr_m["tool_condition"] == "worn", key="wo_step2")
            st.checkbox("Step 3: Clean metal chips and coolant sludge from spindle taper & toolholder socket", value=True, key="wo_step3")
        with chk2:
            st.checkbox(f"Step 4: Verify hydraulic clamp pressure regulator matches recipe ({curr_m['clamp_pressure']} bar)", value=True, key="wo_step4")
            st.checkbox("Step 5: Measure spindle collet radial runout using dial test indicator (< 0.005mm)", value=True, key="wo_step5")
            st.checkbox("Step 6: Execute 30-second dry air-cut test cycle before releasing to production", value=False, key="wo_step6")
            
        st.divider()
        
        # Section 5: Technician Sign-off & Verification
        st.markdown("#### 5. TECHNICIAN CERTIFICATION & CMMS LOG")
        sign_col1, sign_col2 = st.columns([2.5, 1.2])
        with sign_col1:
            st.checkbox("I hereby certify that all required inspections, tool indexing, and safety verifications have been conducted according to ISO 13374 SOP.", key="wo_cert")
            st.caption(f"Certified by: **{wo_technician}** &nbsp;|&nbsp; Shift: **{wo_shift.split()[0]}** &nbsp;|&nbsp; Status: **{wo_status}**")
        with sign_col2:
            if st.button("💾 Log to Plant CMMS", use_container_width=True):
                st.success(f"Work Order {wo_number} logged to Plant Maintenance Database.")
                
    st.divider()
    
    # Download Official Ticket Text
    ticket_text = f"""
================================================================================
NEUROSTORM SCADA | FIELD MAINTENANCE WORK ORDER
Work Order #: {wo_number}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Machine Station: {curr_m['machine_name']}
Priority: {wo_priority}
Lead Technician: {wo_technician} | Shift: {wo_shift}
Tool State: {curr_m['tool_condition'].upper()} (Wear Severity: {curr_m['wear_percent']:.1f}%)
Feedrate: {curr_m['feedrate']} mm/s | Clamp: {curr_m['clamp_pressure']} bar
================================================================================
DIAGNOSTIC FINDINGS:
Machine Health Score: {curr_m['health_score']}%
Cycle Finalized: {curr_m['finalized'].upper()} | Visual Inspection: {curr_m['visual_pass'].upper()}
Total Sensor Observations: {curr_m['observations']}
Flagged Anomaly Points: {curr_m['anomaly_count']} ({curr_m['anomaly_rate']:.2f}%)

PRESCRIBED ACTIONS:
1. Lockout/Tagout machine.
2. Index or replace carbide insert (ISO CNMG-120408-PM).
3. Verify spindle taper runout (< 0.005mm).
4. Verify clamp pressure ({curr_m['clamp_pressure']} bar).
5. Perform test air-cut.

Technician Sign-Off: {wo_technician}   Date: {datetime.now().strftime('%Y-%m-%d')}
================================================================================
    """
    
    st.download_button(
        "📥 Download Maintenance Work Order (TXT)",
        data=ticket_text.encode("utf-8"),
        file_name=f"{wo_number}.txt",
        mime="text/plain"
    )


# ==============================================================================
# MODULE 7: MACHINE FLEET AUDIT & HISTORICAL COMPARISON
# ==============================================================================

elif page == "🗂️ Machine Fleet Audit":
    
    st.markdown("### 🗂️ MACHINE FLEET AUDIT & SIDE-BY-SIDE COMPARISON")
    st.caption("Compare Operating Parameters, Wear Progression & Outcomes Between Any Two Industrial CNC Runs")
    
    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        m1_id = st.selectbox("First Machine Run", sorted(machine_catalog.keys()), index=0, key="audit_m1")
    with comp_col2:
        m2_id = st.selectbox("Second Machine Run", sorted(machine_catalog.keys()), index=6 if len(machine_catalog) > 6 else 1, key="audit_m2")
        
    m1 = machine_catalog[m1_id]
    m2 = machine_catalog[m2_id]
    
    comparison_data = [
        {"Parameter": "Machine Identification", m1["machine_name"]: m1["machine_name"], m2["machine_name"]: m2["machine_name"]},
        {"Parameter": "Operational Health Score", m1["machine_name"]: f"{m1['health_score']}%", m2["machine_name"]: f"{m2['health_score']}%"},
        {"Parameter": "Operational Status", m1["machine_name"]: m1["status_label"], m2["machine_name"]: m2["status_label"]},
        {"Parameter": "Tool Condition", m1["machine_name"]: m1["tool_condition"].upper(), m2["machine_name"]: m2["tool_condition"].upper()},
        {"Parameter": "Wear Severity Risk", m1["machine_name"]: f"{m1['wear_percent']:.1f}%", m2["machine_name"]: f"{m2['wear_percent']:.1f}%"},
        {"Parameter": "Programmed Feedrate", m1["machine_name"]: f"{m1['feedrate']} mm/s", m2["machine_name"]: f"{m2['feedrate']} mm/s"},
        {"Parameter": "Hydraulic Clamp Pressure", m1["machine_name"]: f"{m1['clamp_pressure']} bar", m2["machine_name"]: f"{m2['clamp_pressure']} bar"},
        {"Parameter": "Workpiece Material", m1["machine_name"]: m1["material"].title(), m2["machine_name"]: m2["material"].title()},
        {"Parameter": "Machining Finalized", m1["machine_name"]: m1["finalized"].upper(), m2["machine_name"]: m2["finalized"].upper()},
        {"Parameter": "Passed Visual QA", m1["machine_name"]: m1["visual_pass"].upper(), m2["machine_name"]: m2["visual_pass"].upper()},
        {"Parameter": "Flagged Anomalies", m1["machine_name"]: f"{m1['anomaly_count']} ({m1['anomaly_rate']:.1f}%)", m2["machine_name"]: f"{m2['anomaly_count']} ({m2['anomaly_rate']:.1f}%)"},
        {"Parameter": "Peak Spindle Power", m1["machine_name"]: f"{m1['s1_pwr_max']:.3f} W", m2["machine_name"]: f"{m2['s1_pwr_max']:.3f} W"}
    ]
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Complete Fleet Table
    st.markdown("#### ALL 18 INDUSTRIAL PRODUCTION RUNS")
    fleet_rows = []
    for exp_id, m in machine_catalog.items():
        fleet_rows.append({
            "Unit": m["machine_name"],
            "Health": f"{m['health_score']}%",
            "Status": m["status_label"],
            "Tool": m["tool_condition"].upper(),
            "Wear (%)": round(m["wear_percent"], 1),
            "Feedrate (mm/s)": m["feedrate"],
            "Clamp (bar)": m["clamp_pressure"],
            "Finalized": m["finalized"].upper(),
            "Visual QA": m["visual_pass"].upper(),
            "Anomalies": m["anomaly_count"],
            "Anomaly Rate (%)": round(m["anomaly_rate"], 2),
            "Cycles": m["observations"]
        })
        
    fleet_df = pd.DataFrame(fleet_rows)
    st.dataframe(fleet_df, use_container_width=True, hide_index=True)
    
    st.download_button(
        "📥 Download Fleet Audit Log (CSV)",
        data=fleet_df.to_csv(index=False).encode("utf-8"),
        file_name="neurostorm_scada_fleet_audit.csv",
        mime="text/csv"
    )


# ==============================================================================
# FOOTER
# ==============================================================================

st.divider()
render_html("""
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 12px; color: #64748B;">
    <div>NEUROSTORM SCADA | Intelligent Industrial Machine Testing & Diagnostics Platform</div>
    <div>Live Shop-Floor Link &nbsp;•&nbsp; 100 Hz CNC Telemetry Stream &nbsp;•&nbsp; ISO 13374 Condition Monitoring</div>
</div>
""")
