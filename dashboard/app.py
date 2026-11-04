"""
================================================================================
VERTEX CNC SCADA | PRECISION TELEMETRY & PREDICTIVE MAINTENANCE SYSTEM
Industrial Machine Telemetry, Testing & Diagnostic Intelligence Platform
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
# HELPER: SAFE HTML & ZERO-DEPENDENCY TABLE RENDERING
# ==============================================================================

def render_html(html_str: str):
    """
    Safely renders HTML without triggering Markdown's 4-space indented code block rule.
    Any common leading whitespace is stripped so CommonMark parses it directly as raw HTML.
    """
    st.markdown(textwrap.dedent(html_str).strip(), unsafe_allow_html=True)


def render_clean_table(df: pd.DataFrame, hide_index: bool = True):
    """
    Renders a clean, white industrial HTML table without any dependency on PyArrow.
    """
    table_html = df.to_html(index=not hide_index, classes="scada-table", border=0, escape=True)
    render_html(f'<div class="scada-table-container">{table_html}</div>')


def safe_dataframe(df: pd.DataFrame, hide_index: bool = True):
    """
    Safely renders dataframe with fallback to zero-dependency HTML table.
    Guarantees no PyArrow or Arrow serialization error will ever crash the dashboard.
    """
    df_clean = df.copy()
    try:
        import pyarrow as pa
        # Validate that PyArrow can serialize without error in Python
        _ = pa.Table.from_pandas(df_clean.head(1))
        st.dataframe(df_clean, hide_index=hide_index)
        return
    except BaseException:
        pass

    try:
        render_clean_table(df_clean, hide_index=hide_index)
    except BaseException:
        try:
            st.table(df_clean)
        except BaseException:
            st.write(df_clean.to_dict("records"))


# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="VERTEX CNC | Industrial Telemetry & Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "processed"

# ==============================================================================
# CLEAN INDUSTRIAL DESIGN SYSTEM
# ==============================================================================

st.markdown("""
<style>
/* Base Theme & Layout */
.stApp {
    background-color: #F8FAFC;
    color: #1E293B;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Adjust Streamlit Header & Prevent Clipping on Scroll */
header[data-testid="stHeader"] {
    background-color: transparent !important;
    height: 2.2rem !important;
    z-index: 10 !important;
}

.block-container {
    padding-top: 2.6rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1440px !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

[data-testid="stSidebar"] * {
    color: #334155;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h5 {
    color: #0F172A;
    font-weight: 700;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 6px;
}

/* Sidebar Brand Header Card */
.sidebar-brand-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 2px 0 14px 0;
    border-bottom: 1px solid #F1F5F9;
    margin-bottom: 14px;
}

.brand-title-text {
    font-size: 17px;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.3px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.brand-badge {
    font-size: 10px;
    font-weight: 700;
    color: #2563EB;
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    padding: 1px 6px;
    border-radius: 4px;
    letter-spacing: 0.5px;
}

.brand-sub-text {
    font-size: 11px;
    color: #64748B;
    font-weight: 500;
}

/* Sidebar Radio Navigation Items */
[data-testid="stSidebar"] [role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 4px;
    background: transparent;
    border: none;
    padding: 0;
}

[data-testid="stSidebar"] [role="radiogroup"] > label {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 0;
    cursor: pointer;
    transition: all 0.15s ease-in-out;
}

[data-testid="stSidebar"] [role="radiogroup"] > label:hover {
    background: #F8FAFC;
    border-color: #E2E8F0;
}

[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
    background: #EFF6FF !important;
    border: 1px solid #BFDBFE !important;
    box-shadow: 0 1px 2px rgba(37, 99, 235, 0.05) !important;
}

[data-testid="stSidebar"] [role="radiogroup"] > label span {
    font-size: 13px;
    font-weight: 500;
    color: #475569;
}

[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) span {
    font-weight: 600;
    color: #1D4ED8 !important;
}

[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child {
    display: none !important;
}

/* Page Header Row (Main Page) */
.main-page-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 14px;
    margin-bottom: 16px;
    border-bottom: 1px solid #E2E8F0;
}

.main-page-title {
    font-size: 22px;
    font-weight: 800;
    color: #0F172A;
    margin: 0;
    letter-spacing: -0.4px;
}

.main-page-desc {
    font-size: 13px;
    color: #64748B;
    margin: 3px 0 0 0;
}

.main-page-status {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 12px;
    color: #334155;
    font-weight: 500;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}

/* Active Unit Status Banner (Main Page) */
.active-unit-banner {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 9px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 18px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}

.unit-banner-left {
    display: flex;
    align-items: center;
    gap: 10px;
}

.unit-banner-name {
    font-weight: 700;
    font-size: 14px;
    color: #0F172A;
}

.unit-banner-right {
    font-size: 12px;
    color: #475569;
    display: flex;
    align-items: center;
    gap: 14px;
}

/* Clean Resilient SCADA Table (Zero-Dependency) */
.scada-table-container {
    width: 100%;
    overflow-x: auto;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    margin: 12px 0 16px 0;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
}

.scada-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    color: #1E293B;
    text-align: left;
}

.scada-table thead tr {
    background-color: #F8FAFC;
    border-bottom: 2px solid #E2E8F0;
}

.scada-table th {
    padding: 10px 14px;
    font-weight: 600;
    color: #475569;
    text-align: left;
    white-space: nowrap;
    border: none;
}

.scada-table td {
    padding: 9px 14px;
    border-bottom: 1px solid #F1F5F9;
    border-top: none;
    border-left: none;
    border-right: none;
    color: #334155;
}

.scada-table tbody tr:last-child td {
    border-bottom: none;
}

.scada-table tbody tr:hover {
    background-color: #F8FAFC;
}

.live-pill {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    color: #15803D;
    padding: 5px 12px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.live-dot-simple {
    width: 7px;
    height: 7px;
    background-color: #16A34A;
    border-radius: 50%;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 14px 18px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}

div[data-testid="stMetricLabel"] {
    color: #64748B;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

div[data-testid="stMetricValue"] {
    color: #0F172A;
    font-weight: 700;
    font-size: 24px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* Status Badges */
.status-pill {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}

.status-optimal {
    background: #ECFDF5;
    color: #047857;
    border: 1px solid #A7F3D0;
}

.status-warning {
    background: #FFFBEB;
    color: #B45309;
    border: 1px solid #FDE68A;
}

.status-critical {
    background: #FEF2F2;
    color: #B91C1C;
    border: 1px solid #FECACA;
}

.status-info {
    background: #EFF6FF;
    color: #1D4ED8;
    border: 1px solid #BFDBFE;
}

/* Diagnostic Narrative Callouts */
.diagnostic-banner-optimal {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-left: 4px solid #16A34A;
    padding: 14px 18px;
    border-radius: 6px;
    margin-bottom: 16px;
    color: #15803D;
}

.diagnostic-banner-warning {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 4px solid #D97706;
    padding: 14px 18px;
    border-radius: 6px;
    margin-bottom: 16px;
    color: #92400E;
}

.diagnostic-banner-critical {
    background: #FEF2F2;
    border: 1px solid #FECACA;
    border-left: 4px solid #DC2626;
    padding: 14px 18px;
    border-radius: 6px;
    margin-bottom: 16px;
    color: #991B1B;
}

/* Plotly Chart Containers */
[data-testid="stPlotlyChart"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 8px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
}

/* Form Controls & Inputs */
.stSelectbox > div > div,
.stTextInput input,
.stNumberInput input {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 6px !important;
}

/* Buttons */
.stButton > button {
    background: #2563EB;
    color: #FFFFFF;
    font-weight: 500;
    border: 1px solid #2563EB;
    border-radius: 6px;
    padding: 8px 16px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

.stButton > button:hover {
    background: #1D4ED8;
    border-color: #1D4ED8;
    color: #FFFFFF;
}

/* Dataframe & Tables */
[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    background: #FFFFFF;
}

/* Containers with border */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #E2E8F0 !important;
    background-color: #FFFFFF !important;
    border-radius: 8px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background-color: #F1F5F9;
    padding: 4px;
    border-radius: 6px;
    border: 1px solid #E2E8F0;
}

.stTabs [data-baseweb="tab"] {
    color: #64748B;
    border-radius: 4px;
    font-weight: 500;
    padding: 6px 14px;
    border: none;
}

.stTabs [aria-selected="true"] {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
}

/* Dividers */
hr {
    border-color: #E2E8F0;
    margin: 18px 0;
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
    st.error(f"Error loading telemetry data: {err}")
    st.stop()


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
# SIDEBAR: BRAND, NAVIGATION, AND ACTIVE STATION QUICK VIEW
# ==============================================================================

with st.sidebar:
    # 1. Brand Logo & Title (VERTEX CNC SCADA)
    render_html("""
    <div class="sidebar-brand-card">
        <svg width="34" height="34" viewBox="0 0 34 34" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="34" height="34" rx="8" fill="#2563EB"/>
            <path d="M17 7L24 11V23L17 27L10 23V11L17 7Z" stroke="white" stroke-width="2" stroke-linejoin="round"/>
            <path d="M17 12V22" stroke="white" stroke-width="2" stroke-linecap="round"/>
            <path d="M12 17H22" stroke="#93C5FD" stroke-width="2" stroke-linecap="round"/>
            <circle cx="17" cy="17" r="2.2" fill="white"/>
        </svg>
        <div class="brand-info">
            <div class="brand-title-text">
                <span>VERTEX CNC</span>
                <span class="brand-badge">SCADA</span>
            </div>
            <div class="brand-sub-text">Precision Machine Telemetry</div>
        </div>
    </div>
    """)

    st.markdown("##### NAVIGATION")
    nav_modules = [
        "Fleet Overview",
        "Testing Workbench",
        "Diagnostic Forensics",
        "Telemetry Oscilloscope",
        "Anomaly Detection",
        "Maintenance Work Orders",
        "Fleet Audit & History"
    ]

    page = st.radio(
        "Navigation Menu",
        nav_modules,
        index=0,
        key="sidebar_navigation_radio",
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("##### ACTIVE STATION")
    selected_machine_id = st.selectbox(
        "Active Inspection Station",
        options=sorted(machine_catalog.keys()),
        format_func=lambda x: f"{machine_catalog[x]['machine_name']} — {machine_catalog[x]['status_label'].split()[0]}",
        index=6 if 7 in machine_catalog else 0, # Default to Unit 7
        key="sidebar_machine_selector",
        label_visibility="collapsed"
    )
    curr_m = machine_catalog[selected_machine_id]

    status_border = '#16A34A' if curr_m['status_code'] == 'OPTIMAL' else ('#D97706' if curr_m['status_code'] == 'WARNING' else '#DC2626')
    render_html(f"""
    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid {status_border}; border-radius: 6px; padding: 10px 12px; margin-top: 6px;">
        <div style="font-size: 11px; color: #64748B; font-weight: 600; text-transform: uppercase;">Selected Station</div>
        <div style="font-weight: 700; font-size: 14px; color: #0F172A; margin-top: 2px;">{curr_m['machine_name']}</div>
        <div style="font-size: 11px; margin-top: 3px;" class="{curr_m['badge_class']}">{curr_m['status_label']}</div>
        <div style="font-size: 12px; color: #475569; margin-top: 8px; line-height: 1.6;">
            Health: <b>{curr_m['health_score']}%</b><br>
            Tool State: <b>{curr_m['tool_condition'].upper()}</b><br>
            Wear Severity: <b>{curr_m['wear_percent']:.1f}%</b><br>
            Flagged Anomalies: <b>{curr_m['anomaly_count']}</b>
        </div>
    </div>
    """)

    st.divider()

    render_html("""
    <div style="font-size: 12px; color: #64748B; line-height: 1.6;">
        <div><b>Facility:</b> Bay 04 • High-Speed Cells</div>
        <div style="margin-top: 4px;">
            <span class="live-pill" style="font-size: 11px; padding: 2px 8px;">
                <span class="live-dot-simple"></span> Stream Online • 100 Hz
            </span>
        </div>
    </div>
    """)
    st.caption(f"Clock: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ==============================================================================
# MAIN VIEWPORT: HEADER & ACTIVE STATUS BANNER
# ==============================================================================

page_descriptions = {
    "Fleet Overview": "Real-time health telemetry and condition monitoring across all 18 industrial CNC machining cells",
    "Testing Workbench": "Spindle stress simulator, dynamic telemetry streaming, and edge anomaly detection testbench",
    "Diagnostic Forensics": "Root-cause investigation, electrical power feedback, and technician action protocols",
    "Telemetry Oscilloscope": "High-resolution multi-channel sensor waveform analysis and electrical signal tracking",
    "Anomaly Detection": "Multi-dimensional Isolation Forest outlier analysis and abnormal cycle classification",
    "Maintenance Work Orders": "Standardized corrective & preventive maintenance tickets for reliability engineers",
    "Fleet Audit & History": "Comparative audit of operating parameters, tool wear progression, and QA results across runs"
}

render_html(f"""
<div class="main-page-header">
    <div>
        <h2 class="main-page-title">{page}</h2>
        <p class="main-page-desc">{page_descriptions.get(page, '')}</p>
    </div>
    <div class="main-page-status">
        <span class="live-dot-simple"></span>
        <span>Plant Bay 04 &nbsp;•&nbsp; 100 Hz CNC Telemetry</span>
    </div>
</div>
""")

if page in ("Testing Workbench", "Diagnostic Forensics", "Telemetry Oscilloscope", "Maintenance Work Orders"):
    status_border = '#16A34A' if curr_m['status_code'] == 'OPTIMAL' else ('#D97706' if curr_m['status_code'] == 'WARNING' else '#DC2626')
    render_html(f"""
    <div class="active-unit-banner" style="border-left: 4px solid {status_border};">
        <div class="unit-banner-left">
            <span class="unit-banner-name">{curr_m['machine_name']}</span>
            <span class="status-pill {curr_m['badge_class']}">{curr_m['status_label']}</span>
        </div>
        <div class="unit-banner-right">
            <span>Health: <b style="color: {status_border};">{curr_m['health_score']}%</b></span>
            <span>Wear Risk: <b>{curr_m['wear_percent']:.1f}%</b></span>
            <span>Feedrate: <b>{curr_m['feedrate']} mm/s</b></span>
            <span>Clamp: <b>{curr_m['clamp_pressure']} bar</b></span>
        </div>
    </div>
    """)


# ==============================================================================
# MODULE 1: FLEET OVERVIEW
# ==============================================================================

if page == "Fleet Overview":
    
    # KPI Strip
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    with kpi1:
        st.metric("Total CNC Fleet", f"{total_machines} Units")
    with kpi2:
        st.metric("Fleet Health Index", f"{fleet_health_avg:.1f}%")
    with kpi3:
        st.metric("Optimal Operating", f"{optimal_count} Units", delta="Nominal", delta_color="normal")
    with kpi4:
        st.metric("Wear Advisory", f"{warning_count} Units", delta="Inspection Due", delta_color="off")
    with kpi5:
        st.metric("Critical Interventions", f"{critical_count} Units", delta="Action Required", delta_color="inverse")
        
    st.divider()
    
    st.markdown("#### Machine Fleet Matrix")
    
    filter_choice = st.radio(
        "Filter Fleet",
        ["All Units (18)", "Optimal (6)", "Tool Wear Elevated (4)", "Aborted / Defect (8)"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
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
            
    cols = st.columns(3)
    for i, m in enumerate(filtered_units):
        col = cols[i % 3]
        border_color = "#16A34A" if m["status_code"] == "OPTIMAL" else ("#D97706" if m["status_code"] == "WARNING" else "#DC2626")
        outcome_label = 'Finished & Passed QA' if m['finalized'] == 'yes' and m['visual_pass'] == 'yes' else ('Finished / Rejected QA' if m['finalized'] == 'yes' else 'Premature Abort')
        outcome_color = '#15803D' if m['finalized'] == 'yes' and m['visual_pass'] == 'yes' else '#B91C1C'
        
        with col:
            render_html(f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-top: 3px solid {border_color}; border-radius: 8px; padding: 14px 16px; margin-bottom: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-weight: 600; font-size: 14px; color: #0F172A;">{m['machine_name']}</span>
                    <span class="status-pill {m['badge_class']}">{m['status_label'].split()[0]}</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px; margin-bottom: 10px;">
                    <div><span style="color: #64748B;">Tool:</span> <b style="color: #1E293B;">{m['tool_condition'].upper()}</b></div>
                    <div><span style="color: #64748B;">Health:</span> <b style="color: {border_color};">{m['health_score']}%</b></div>
                    <div><span style="color: #64748B;">Feedrate:</span> <b style="color: #1E293B;">{m['feedrate']} mm/s</b></div>
                    <div><span style="color: #64748B;">Clamp:</span> <b style="color: #1E293B;">{m['clamp_pressure']} bar</b></div>
                    <div><span style="color: #64748B;">Wear Risk:</span> <b style="color: #1E293B;">{m['wear_percent']:.1f}%</b></div>
                    <div><span style="color: #64748B;">Anomalies:</span> <b style="color: #1E293B;">{m['anomaly_count']} ({m['anomaly_rate']:.1f}%)</b></div>
                </div>
                <div style="font-size: 11px; color: #64748B; border-top: 1px solid #F1F5F9; padding-top: 8px;">
                    Cycle Outcome: <b style="color: {outcome_color};">{outcome_label}</b>
                </div>
            </div>
            """)
            
    st.divider()
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("#### Operational Status Distribution")
        status_df = pd.DataFrame([
            {"Status": "Optimal Running", "Count": optimal_count},
            {"Status": "Tool Wear Elevated", "Count": warning_count},
            {"Status": "Intervention Required", "Count": critical_count}
        ])
        
        fig_donut = px.pie(
            status_df,
            names="Status",
            values="Count",
            hole=0.60,
            color="Status",
            color_discrete_map={
                "Optimal Running": "#16A34A",
                "Tool Wear Elevated": "#D97706",
                "Intervention Required": "#DC2626"
            }
        )
        fig_donut.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#334155", family="sans-serif"),
            margin=dict(l=20, r=20, t=20, b=20),
            height=280,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with chart_col2:
        st.markdown("#### Anomaly Incidents per Machine")
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
                "OPTIMAL": "#16A34A",
                "WARNING": "#D97706",
                "ABORTED": "#DC2626",
                "DEFECT": "#EA580C"
            },
            labels={"Anomalies": "Flagged Incidents", "Machine": "CNC Unit"}
        )
        fig_anom.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#334155", family="sans-serif"),
            margin=dict(l=20, r=20, t=20, b=20),
            height=280,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_anom, use_container_width=True)


# ==============================================================================
# MODULE 2: TESTING WORKBENCH
# ==============================================================================

elif page == "Testing Workbench":
    workbench_mode = st.radio(
        "Workbench Mode",
        ["Interactive Testing Simulator", "Recorded CNC Telemetry Playback"],
        horizontal=True
    )
    
    if workbench_mode == "Interactive Testing Simulator":
        
        render_html("""
        <div class="diagnostic-banner-optimal">
            <b>Testing Simulator Active:</b> Adjust operating parameters or apply stress test presets to evaluate real-time PLC diagnostic rules.
        </div>
        """)
        
        st.markdown("#### Stress Test Presets")
        preset_cols = st.columns(4)
        
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
            if st.button("Overheating Stress", use_container_width=True):
                st.session_state.sim_air_temp = 314.5
                st.session_state.sim_proc_temp = 328.0
                st.session_state.sim_torque = 44.0
                st.toast("Applied: Spindle Overheating condition")
                
        with preset_cols[1]:
            if st.button("Mechanical Chatter Stress", use_container_width=True):
                st.session_state.sim_torque = 64.0
                st.session_state.sim_rpm = 1120
                st.session_state.sim_tool_wear = 135.0
                st.toast("Applied: Mechanical Overload & Chatter condition")
                
        with preset_cols[2]:
            if st.button("Tool Wear Limit Stress", use_container_width=True):
                st.session_state.sim_tool_wear = 225.0
                st.session_state.sim_torque = 52.0
                st.toast("Applied: Tool Flank Wear Exceeded (225 min)")
                
        with preset_cols[3]:
            if st.button("Nominal Baseline", use_container_width=True):
                st.session_state.sim_rpm = 1500
                st.session_state.sim_torque = 40.0
                st.session_state.sim_air_temp = 300.0
                st.session_state.sim_proc_temp = 310.0
                st.session_state.sim_tool_wear = 50.0
                st.session_state.sim_running = True
                st.toast("Reset to Nominal Baseline")
                
        st.divider()
        
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
        
        with ctrl_col1:
            st.markdown("##### Spindle Dynamics")
            sim_rpm = st.slider(
                "Spindle Velocity [RPM]",
                min_value=1000,
                max_value=2000,
                value=int(st.session_state.sim_rpm),
                step=10,
                help="Nominal: 1400 - 1700 RPM"
            )
            sim_torque = st.slider(
                "Spindle Torque [Nm]",
                min_value=20.0,
                max_value=75.0,
                value=float(st.session_state.sim_torque),
                step=0.5,
                help="Safe limit: < 55 Nm"
            )
            
        with ctrl_col2:
            st.markdown("##### Thermal Environment")
            sim_air = st.slider(
                "Air Temperature [K]",
                min_value=295.0,
                max_value=320.0,
                value=float(st.session_state.sim_air_temp),
                step=0.5,
                help="Ambient shop temperature (300K ≈ 27°C)"
            )
            sim_proc = st.slider(
                "Process / Coolant Temp [K]",
                min_value=305.0,
                max_value=330.0,
                value=float(st.session_state.sim_proc_temp),
                step=0.5,
                help="Cutting zone temperature"
            )
            
        with ctrl_col3:
            st.markdown("##### Tooling & Machine State")
            sim_wear = st.slider(
                "Tool Flank Wear [min]",
                min_value=0.0,
                max_value=250.0,
                value=float(st.session_state.sim_tool_wear),
                step=1.0,
                help="Safe wear threshold: 200 min"
            )
            sim_class = st.selectbox("Machine Classification", ["Type L (Standard Mill)", "Type M (High Torque)", "Type H (Heavy Production)"])
            
        st.session_state.sim_rpm = sim_rpm
        st.session_state.sim_torque = sim_torque
        st.session_state.sim_air_temp = sim_air
        st.session_state.sim_proc_temp = sim_proc
        st.session_state.sim_tool_wear = sim_wear
        
        temp_delta = sim_proc - sim_air
        
        faults_detected = []
        action_recommendations = []
        
        if sim_air > 310 or sim_proc > 320 or temp_delta > 15:
            faults_detected.append("THERMAL OVERHEATING WARNING")
            action_recommendations.append("Inspect coolant pump flow rate and chiller fluid levels.")
            
        if sim_torque > 55 or sim_rpm < 1200 or sim_rpm > 1800:
            faults_detected.append("MECHANICAL OVERLOAD / CHATTER")
            action_recommendations.append("Inspect motor drive coupling, spindle bearings and axis backlash.")
            
        if sim_wear > 200:
            faults_detected.append("CRITICAL TOOL FLANK WEAR")
            action_recommendations.append("Tool insert has exceeded safe life. Stop cycle and index carbide insert.")
            
        if not faults_detected:
            machine_health_status = "OPTIMAL: Operating within certified parameters"
            status_banner_class = "diagnostic-banner-optimal"
            action_recommendations.append("Machine is operating nominally. Continue standard monitoring.")
        else:
            machine_health_status = "ALARM DETECTED: " + " • ".join(faults_detected)
            status_banner_class = "diagnostic-banner-critical" if len(faults_detected) > 1 else "diagnostic-banner-warning"
            
        render_html(f"""
        <div class="{status_banner_class}">
            <div style="font-size: 14px; font-weight: 700;">{machine_health_status}</div>
            <div style="font-size: 12px; margin-top: 4px; opacity: 0.9;">
                <b>Thermal Differential:</b> {temp_delta:.1f} K &nbsp;|&nbsp; 
                <b>Torque Load:</b> {sim_torque:.1f} Nm &nbsp;|&nbsp; 
                <b>Spindle Speed:</b> {sim_rpm} RPM &nbsp;|&nbsp; 
                <b>Tool Wear:</b> {sim_wear:.0f} min
            </div>
        </div>
        """)
        
        st.markdown("#### Real-Time Telemetry Indicators")
        g_col1, g_col2, g_col3, g_col4 = st.columns(4)
        
        with g_col1:
            fig_g1 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sim_rpm,
                title={"text": "Spindle Speed (RPM)", "font": {"size": 13, "color": "#334155"}},
                gauge={
                    "axis": {"range": [1000, 2000], "tickcolor": "#94A3B8"},
                    "bar": {"color": "#2563EB"},
                    "steps": [
                        {"range": [1000, 1200], "color": "#FEE2E2"},
                        {"range": [1200, 1800], "color": "#DCFCE7"},
                        {"range": [1800, 2000], "color": "#FEF3C7"}
                    ]
                }
            ))
            fig_g1.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", height=200, margin=dict(l=15, r=15, t=30, b=15))
            st.plotly_chart(fig_g1, use_container_width=True)
            
        with g_col2:
            fig_g2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sim_torque,
                title={"text": "Cutting Torque (Nm)", "font": {"size": 13, "color": "#334155"}},
                gauge={
                    "axis": {"range": [20, 75], "tickcolor": "#94A3B8"},
                    "bar": {"color": "#2563EB"},
                    "steps": [
                        {"range": [20, 50], "color": "#DCFCE7"},
                        {"range": [50, 60], "color": "#FEF3C7"},
                        {"range": [60, 75], "color": "#FEE2E2"}
                    ]
                }
            ))
            fig_g2.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", height=200, margin=dict(l=15, r=15, t=30, b=15))
            st.plotly_chart(fig_g2, use_container_width=True)
            
        with g_col3:
            fig_g3 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=temp_delta,
                title={"text": "Thermal Delta ΔT (K)", "font": {"size": 13, "color": "#334155"}},
                gauge={
                    "axis": {"range": [0, 25], "tickcolor": "#94A3B8"},
                    "bar": {"color": "#2563EB"},
                    "steps": [
                        {"range": [0, 12], "color": "#DCFCE7"},
                        {"range": [12, 16], "color": "#FEF3C7"},
                        {"range": [16, 25], "color": "#FEE2E2"}
                    ]
                }
            ))
            fig_g3.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", height=200, margin=dict(l=15, r=15, t=30, b=15))
            st.plotly_chart(fig_g3, use_container_width=True)
            
        with g_col4:
            fig_g4 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sim_wear,
                title={"text": "Tool Flank Wear (min)", "font": {"size": 13, "color": "#334155"}},
                gauge={
                    "axis": {"range": [0, 250], "tickcolor": "#94A3B8"},
                    "bar": {"color": "#2563EB"},
                    "steps": [
                        {"range": [0, 120], "color": "#DCFCE7"},
                        {"range": [120, 200], "color": "#FEF3C7"},
                        {"range": [200, 250], "color": "#FEE2E2"}
                    ]
                }
            ))
            fig_g4.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", height=200, margin=dict(l=15, r=15, t=30, b=15))
            st.plotly_chart(fig_g4, use_container_width=True)
            
        st.markdown("#### Rolling Telemetry Buffer")
        stream_btn_col1, stream_btn_col2, _ = st.columns([1, 1, 2])
        
        with stream_btn_col1:
            run_stream = st.button("Simulate Sample Stream", use_container_width=True)
            
        with stream_btn_col2:
            if st.button("Emergency Stop", use_container_width=True):
                st.session_state.sim_rpm = 0
                st.session_state.sim_torque = 0
                st.error("Emergency machine stop activated. Spindle halted.")
                st.stop()
                
        chart_placeholder = st.empty()
        
        np.random.seed(42)
        n_ticks = 40
        time_index = np.arange(n_ticks)
        
        stream_rpm = np.clip(sim_rpm + np.random.normal(0, 15, n_ticks), 1000, 2000)
        stream_torque = np.clip(sim_torque + np.random.normal(0, 1.8, n_ticks), 20, 75)
        stream_delta = np.clip(temp_delta + np.random.normal(0, 0.4, n_ticks), 0, 25)
        
        stream_df = pd.DataFrame({
            "Sample": time_index,
            "Spindle RPM": stream_rpm,
            "Torque (Nm)": stream_torque,
            "Thermal Delta (K)": stream_delta
        })
        
        fig_stream = px.line(
            stream_df,
            x="Sample",
            y=["Spindle RPM", "Torque (Nm)", "Thermal Delta (K)"],
            title="Telemetry Stream (Last 40 Samples)",
            labels={"value": "Sensor Reading", "Sample": "Sample Index"},
            color_discrete_sequence=["#2563EB", "#D97706", "#DC2626"]
        )
        fig_stream.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#334155", family="sans-serif"),
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        chart_placeholder.plotly_chart(fig_stream, use_container_width=True)
        
        if run_stream:
            st.info("Continuous telemetry streaming active. Connection healthy at 100 Hz.")
            
    else:
        st.markdown(f"#### Recorded Telemetry Playback: {curr_m['machine_name']}")
        st.caption("Scrub through the recorded cut cycle row-by-row:")
        
        exp_sensor = sensor_data[sensor_data["Experiment_ID"] == selected_machine_id].copy()
        if "M1_sequence_number" in exp_sensor.columns:
            exp_sensor = exp_sensor.sort_values("M1_sequence_number").reset_index(drop=True)
            
        total_steps = len(exp_sensor)
        step = st.slider("Cycle Sequence Step", 0, total_steps - 1, min(100, total_steps - 1))
        
        curr_row = exp_sensor.iloc[step]
        
        pb_col1, pb_col2, pb_col3, pb_col4 = st.columns(4)
        pb_col1.metric("Machining Process", str(curr_row.get("Machining_Process", "Unknown")).upper())
        pb_col2.metric("Spindle Current Feedback", f"{curr_row.get('S1_CurrentFeedback', 0.0):.2f} A")
        pb_col3.metric("Spindle Output Power", f"{curr_row.get('S1_OutputPower', 0.0):.3f} W")
        pb_col4.metric("X1 Axis Velocity", f"{curr_row.get('X1_ActualVelocity', 0.0):.2f} mm/s")
        
        start_w = max(0, step - 60)
        end_w = min(total_steps, step + 60)
        window_df = exp_sensor.iloc[start_w:end_w]
        
        fig_pb = px.line(
            window_df,
            x=window_df.index,
            y=["S1_CurrentFeedback", "X1_ActualVelocity", "Y1_ActualVelocity"],
            title=f"Telemetry Window around Step {step} of {total_steps}",
            color_discrete_sequence=["#2563EB", "#059669", "#D97706"]
        )
        fig_pb.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#334155", family="sans-serif"),
            height=300
        )
        fig_pb.add_vline(x=step, line_dash="dash", line_color="#DC2626", annotation_text="Selected Step")
        st.plotly_chart(fig_pb, use_container_width=True)


# ==============================================================================
# MODULE 3: DIAGNOSTIC FORENSICS
# ==============================================================================

elif page == "Diagnostic Forensics":
    if curr_m["status_code"] == "ABORTED":
        render_html(f"""
        <div class="diagnostic-banner-critical">
            <div style="font-size: 15px; font-weight: 700; color: #991B1B;">Emergency Cycle Abort: Machine Halted Prematurely</div>
            <div style="font-size: 13px; color: #334155; margin-top: 6px; line-height: 1.5;">
                <b>Summary:</b> This run was terminated early at <b>{curr_m['observations']} cycles</b> (expected full run is ~2,200 cycles). 
                The operation was conducted at <b>{curr_m['feedrate']} mm/s feedrate</b> and <b>{curr_m['clamp_pressure']} bar clamp pressure</b> with a <b>{curr_m['tool_condition']}</b> cutting tool.
                <br><b>Failure Trigger:</b> Cutting resistance and tool chatter exceeded the machine safety interlock, forcing an emergency cutoff.
            </div>
        </div>
        """)
    elif curr_m["status_code"] == "DEFECT":
        render_html(f"""
        <div class="diagnostic-banner-critical">
            <div style="font-size: 15px; font-weight: 700; color: #991B1B;">Cycle Completed with QA Rejection</div>
            <div style="font-size: 13px; color: #334155; margin-top: 6px; line-height: 1.5;">
                <b>Summary:</b> The machine finalized the complete machining sequence ({curr_m['observations']} cycles), but the manufactured part 
                <b>failed visual quality inspection</b> due to surface chatter marks and dimensional non-conformance.
                <br><b>Root Cause:</b> Worn cutting tool (Wear Probability: <b>{curr_m['wear_percent']:.1f}%</b>) generated high friction and heat.
            </div>
        </div>
        """)
    elif curr_m["status_code"] == "WARNING":
        render_html(f"""
        <div class="diagnostic-banner-warning">
            <div style="font-size: 15px; font-weight: 700; color: #92400E;">Tool Degradation Advisory: Flank Wear Approaching Limit</div>
            <div style="font-size: 13px; color: #334155; margin-top: 6px; line-height: 1.5;">
                <b>Summary:</b> Machine completed the full run ({curr_m['observations']} cycles) and passed visual inspection, but sensor telemetry 
                indicates high tool degradation (Wear Severity: <b>{curr_m['wear_percent']:.1f}%</b>).
                <br><b>Recommendation:</b> Schedule tool replacement before the next production cycle.
            </div>
        </div>
        """)
    else:
        render_html(f"""
        <div class="diagnostic-banner-optimal">
            <div style="font-size: 15px; font-weight: 700; color: #15803D;">Certified Nominal Run: All Parameters Optimal</div>
            <div style="font-size: 13px; color: #334155; margin-top: 6px; line-height: 1.5;">
                <b>Summary:</b> Machine completed all {curr_m['observations']} cycles with zero defects and passed visual inspection. 
                Tool condition is unworn, cutting resistance remained minimal ({curr_m['anomaly_rate']:.1f}% anomaly rate), and motion tracking was nominal.
            </div>
        </div>
        """)
        
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tool State", curr_m["tool_condition"].upper())
    k2.metric("Tool Wear Severity", f"{curr_m['wear_percent']:.1f}%")
    k3.metric("Cycle Finalized", curr_m["finalized"].upper())
    k4.metric("Visual QA Passed", curr_m["visual_pass"].upper())
    
    st.divider()
    
    st.markdown("#### Subsystem Investigation")
    
    exp_sensor = sensor_data[sensor_data["Experiment_ID"] == selected_machine_id]
    exp_anom = anomaly_data[anomaly_data["Experiment_ID"] == selected_machine_id]
    
    sub1, sub2, sub3 = st.columns(3)
    
    with sub1:
        with st.container(border=True):
            st.markdown("##### Spindle Motor Subsystem (S1)")
            st.write(f"• **Peak Current Feedback:** `{curr_m['s1_curr_max']:.2f} A`")
            st.write(f"• **Peak Output Power:** `{curr_m['s1_pwr_max']:.3f} W`")
            if curr_m["tool_condition"] == "worn":
                st.error("Spindle current feedback showed high transient surges during roughing passes, indicating tool dulling and increased friction.")
            else:
                st.success("Spindle current feedback remained uniform and within normal mechanical torque tolerances.")
        
    with sub2:
        with st.container(border=True):
            st.markdown("##### Axis Motion Subsystem (X/Y/Z)")
            st.write(f"• **X-Axis Peak Acceleration:** `{curr_m['x1_acc_max']:.3f}`")
            st.write(f"• **Y-Axis Peak Acceleration:** `{curr_m['y1_acc_max']:.3f}`")
            if curr_m["feedrate"] >= 15:
                st.warning(f"High programmed feedrate ({curr_m['feedrate']} mm/s) induced sharp acceleration reversals during contour transitions.")
            else:
                st.success(f"Programmed feedrate ({curr_m['feedrate']} mm/s) maintained smooth linear motion and minimal chatter.")
        
    with sub3:
        with st.container(border=True):
            st.markdown("##### Clamping & Workpiece Fixture")
            st.write(f"• **Clamp Pressure:** `{curr_m['clamp_pressure']} bar`")
            st.write(f"• **Workpiece Material:** `{curr_m['material'].title()}`")
            if curr_m["clamp_pressure"] <= 2.5:
                st.error("Low clamp pressure (2.5 bar) posed elevated risk of workpiece vibration and micro-slippage under cutting forces.")
            else:
                st.success(f"Clamp pressure ({curr_m['clamp_pressure']} bar) provided rigid workpiece fixture stability throughout the cut.")
        
    st.divider()
    
    st.markdown("#### Machining Process Breakdown")
    st.caption("Distribution of observations and anomaly rates by machining stage:")
    
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
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#334155", family="sans-serif"),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_proc, use_container_width=True)
    
    st.divider()
    
    st.markdown("#### Technician Action Protocol")
    st.caption("Standard operating procedure checklist for maintenance engineers:")
    
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.checkbox("Step 1: Execute Lockout/Tagout (LOTO) on main power breaker", value=True)
        st.checkbox("Step 2: Remove cutting tool & inspect flank wear under optical loupe", value=curr_m["tool_condition"] == "worn")
        st.checkbox("Step 3: Measure collet chuck radial runout with dial indicator", value=True)
    with t_col2:
        st.checkbox(f"Step 4: Verify hydraulic clamp pressure regulator ({curr_m['clamp_pressure']} bar)", value=True)
        st.checkbox("Step 5: Clean chips and coolant residue from spindle taper", value=True)
        st.checkbox("Step 6: Run single-block air-cut test cycle before production release", value=False)


# ==============================================================================
# MODULE 4: TELEMETRY OSCILLOSCOPE
# ==============================================================================

elif page == "Telemetry Oscilloscope":
    exp_sensor = sensor_data[sensor_data["Experiment_ID"] == selected_machine_id].copy()
    if "M1_sequence_number" in exp_sensor.columns:
        exp_sensor = exp_sensor.sort_values("M1_sequence_number").reset_index(drop=True)
        
    osc_col1, osc_col2 = st.columns([1, 2])
    
    with osc_col1:
        signal_family = st.selectbox(
            "Select Signal Family",
            [
                "Axis Current Feedback (S1, X1, Y1)",
                "Axis Acceleration Feedback (X1, Y1, Z1)",
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
            labels={"index": "Sample Sequence", "value": y_title},
            color_discrete_sequence=["#2563EB", "#059669", "#D97706", "#7C3AED"]
        )
        fig_wave.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#334155", family="sans-serif"),
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_wave, use_container_width=True)
        
        st.markdown("#### Channel Summary Statistics")
        stat_cols = st.columns(len(signals))
        for col, sig in zip(stat_cols, signals):
            series = filtered_plot[sig].dropna()
            with col:
                with st.container(border=True):
                    st.caption(sig)
                    st.metric("Mean", f"{series.mean():.3f}")
                    st.caption(f"Peak: **{series.max():.3f}** | Min: **{series.min():.3f}**")


# ==============================================================================
# MODULE 5: ANOMALY DETECTION
# ==============================================================================

elif page == "Anomaly Detection":
    
    anom_k1, anom_k2, anom_k3, anom_k4 = st.columns(4)
    total_anom = int(anomaly_data["anomaly"].sum())
    total_obs = len(anomaly_data)
    fleet_anom_rate = total_anom / total_obs * 100 if total_obs > 0 else 0
    
    anom_k1.metric("Total Flagged Points", f"{total_anom:,}")
    anom_k2.metric("Telemetry Samples", f"{total_obs:,}")
    anom_k3.metric("Fleet Anomaly Rate", f"{fleet_anom_rate:.2f}%")
    anom_k4.metric("Selected Machine Rate", f"{curr_m['anomaly_rate']:.2f}%")
    
    st.divider()
    
    anom_c1, anom_c2 = st.columns(2)
    
    with anom_c1:
        st.markdown("#### Anomaly Rate by Tool Condition")
        cond_anom = anomaly_data.groupby("tool_condition").agg(
            Total=("anomaly", "count"),
            Anomalies=("anomaly", "sum")
        ).reset_index()
        cond_anom["Rate (%)"] = (cond_anom["Anomalies"] / cond_anom["Total"] * 100).round(2)
        
        fig_cond = px.bar(
            cond_anom,
            x="tool_condition",
            y="Rate (%)",
            color="tool_condition",
            color_discrete_map={"worn": "#DC2626", "unworn": "#16A34A"},
            title="Anomaly Rate: Worn vs Unworn Tooling",
            labels={"tool_condition": "Tool State", "Rate (%)": "Anomaly Rate (%)"}
        )
        fig_cond.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#334155", family="sans-serif"),
            height=300
        )
        st.plotly_chart(fig_cond, width="stretch")
        
    with anom_c2:
        st.markdown("#### Top Anomaly-Prone Processes")
        proc_anom = anomaly_data.groupby("Machining_Process").agg(
            Total=("anomaly", "count"),
            Anomalies=("anomaly", "sum")
        ).reset_index()
        proc_anom["Rate (%)"] = (proc_anom["Anomalies"] / proc_anom["Total"] * 100).round(2)
        proc_anom = proc_anom.sort_values("Rate (%)", ascending=False).head(8)
        
        fig_proc_anom = px.bar(
            proc_anom,
            x="Machining_Process",
            y="Rate (%)",
            color="Rate (%)",
            color_continuous_scale="Blues",
            title="Processes by Anomaly Percentage",
            labels={"Machining_Process": "Process", "Rate (%)": "Anomaly Rate (%)"}
        )
        fig_proc_anom.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#334155", family="sans-serif"),
            height=300,
            xaxis_tickangle=-35
        )
        st.plotly_chart(fig_proc_anom, width="stretch")
        
    st.divider()
    
    st.markdown(f"#### Anomaly Incidents for {curr_m['machine_name']}")
    curr_anom_rows = anomaly_data[
        (anomaly_data["Experiment_ID"] == selected_machine_id) & 
        (anomaly_data["anomaly"] == 1)
    ]
    
    if curr_anom_rows.empty:
        st.success(f"No anomaly events recorded for {curr_m['machine_name']}.")
    else:
        st.caption(f"Showing {len(curr_anom_rows)} flagged telemetry cycles:")
        display_anom = curr_anom_rows[["Machining_Process", "tool_condition", "anomaly_score"]].copy()
        display_anom["anomaly_score"] = display_anom["anomaly_score"].round(4)
        display_anom["tool_condition"] = display_anom["tool_condition"].str.upper()
        display_anom.columns = ["Machining Process", "Tool State", "Anomaly Score"]
        safe_dataframe(display_anom.head(50), hide_index=True)


# ==============================================================================
# MODULE 6: MAINTENANCE WORK ORDERS
# ==============================================================================

elif page == "Maintenance Work Orders":
    
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1.2, 1.2])
    with ctrl_col1:
        wo_technician = st.text_input("Lead Technician Name", value="Alex Mercer (Reliability Tech II)")
    with ctrl_col2:
        wo_shift = st.selectbox("Production Shift", ["Shift A (Morning 06:00 - 14:30)", "Shift B (Evening 14:30 - 23:00)", "Shift C (Night 23:00 - 06:00)"])
    with ctrl_col3:
        wo_status = st.selectbox("Work Order Status", ["Issued / Action Required", "In Progress", "Completed & Certified"])
        
    wo_priority = "HIGH (IMMEDIATE ACTION REQUIRED)" if curr_m["status_code"] in ("ABORTED", "DEFECT") else ("MEDIUM (SCHEDULED MAINTENANCE)" if curr_m["status_code"] == "WARNING" else "ROUTINE (PREVENTIVE MAINTENANCE)")
    wo_number = f"WO-2026-CNC-{selected_machine_id:02d}-{int(curr_m['wear_percent'])}"
    
    st.divider()
    
    with st.container(border=True):
        
        head_col1, head_col2 = st.columns([3, 1])
        with head_col1:
            st.markdown(f"### Field Work Order: `{wo_number}`")
            st.caption(f"**Facility:** Plant Bay 04 • High-Speed Milling Cell &nbsp;|&nbsp; **Machine:** `{curr_m['machine_name']}`")
        with head_col2:
            priority_color = "#DC2626" if "HIGH" in wo_priority else ("#D97706" if "MEDIUM" in wo_priority else "#16A34A")
            render_html(f"""
            <div style="text-align: right; background: #F8FAFC; padding: 10px 14px; border-radius: 6px; border: 1px solid #E2E8F0;">
                <div style="font-size: 11px; color: #64748B; text-transform: uppercase; font-weight: 600;">Priority Level</div>
                <div style="font-size: 15px; font-weight: 700; color: {priority_color}; margin-top: 2px;">{wo_priority.split()[0]}</div>
                <div style="font-size: 11px; color: #94A3B8; margin-top: 2px;">Date: {datetime.now().strftime('%Y-%m-%d')}</div>
            </div>
            """)
            
        st.divider()
        
        st.markdown("#### 1. Machine Operating State Snapshot")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Operating Health", f"{curr_m['health_score']}%", delta=f"{curr_m['status_label'].split()[0]}")
        m_col2.metric("Tool Wear Severity", f"{curr_m['wear_percent']:.1f}%")
        m_col3.metric("Feedrate & Clamp", f"{curr_m['feedrate']} mm/s | {curr_m['clamp_pressure']} bar")
        m_col4.metric("Visual QA Result", curr_m['visual_pass'].upper())
        
        st.divider()
        
        st.markdown("#### 2. Symptoms & Diagnostic Determination")
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
            Tool replacement is required prior to next production cycle to prevent workpiece rejection.
            """)
        else:
            st.success(f"""
            **ROUTINE PREVENTIVE CHECK:** All cutting forces and electrical signals are nominal (Health Score: {curr_m['health_score']}%). 
            Continue standard visual inspection and tool lubrication schedule.
            """)
            
        st.divider()
        
        st.markdown("#### 3. Required Tooling & Consumables")
        parts_data = [
            {"Item": 1, "Part Number": "CNMG-120408-PM", "Description": "TiAlN Coated Carbide Milling Insert", "Qty": "1 Set (4 Edges)", "Location": "Bin Tooling-A12", "Status": "In Stock"},
            {"Item": 2, "Part Number": "ER32-COLLET-08", "Description": "High Precision Spring Collet (< 0.005mm)", "Qty": "1 Unit", "Location": "Tool Crib B-04", "Status": "In Stock"},
            {"Item": 3, "Part Number": "ISO-VG-46-SYN", "Description": "Synthetic Water-Soluble Coolant Concentrate", "Qty": "5 Liters", "Location": "Fluid Bay F-01", "Status": "Available"},
            {"Item": 4, "Part Number": "VISE-SEAL-KIT", "Description": "Hydraulic Vise Pressure Seal Kit", "Qty": "1 Kit", "Location": "Pneumatics P-08", "Status": "In Stock"}
        ]
        safe_dataframe(pd.DataFrame(parts_data), hide_index=True)
        
        st.divider()
        
        st.markdown("#### 4. Field Action Procedures")
        st.caption("Verify each procedure performed at the machine:")
        
        chk1, chk2 = st.columns(2)
        with chk1:
            st.checkbox("Step 1: Perform 480V Electrical Disconnect & Lockout/Tagout (LOTO)", value=True, key="wo_step1")
            st.checkbox("Step 2: Inspect cutting insert flank wear under optical loupe", value=curr_m["tool_condition"] == "worn", key="wo_step2")
            st.checkbox("Step 3: Clean chips and coolant residue from spindle taper", value=True, key="wo_step3")
        with chk2:
            st.checkbox(f"Step 4: Verify hydraulic clamp pressure matches recipe ({curr_m['clamp_pressure']} bar)", value=True, key="wo_step4")
            st.checkbox("Step 5: Measure spindle collet radial runout using dial indicator (< 0.005mm)", value=True, key="wo_step5")
            st.checkbox("Step 6: Execute 30-second dry air-cut test cycle before releasing machine", value=False, key="wo_step6")
            
        st.divider()
        
        st.markdown("#### 5. Certification & Sign-off")
        sign_col1, sign_col2 = st.columns([2.5, 1.2])
        with sign_col1:
            st.checkbox("I hereby certify that all required inspections, tool indexing, and safety verifications have been conducted.", key="wo_cert")
            st.caption(f"Technician: **{wo_technician}** &nbsp;|&nbsp; Shift: **{wo_shift.split()[0]}** &nbsp;|&nbsp; Status: **{wo_status}**")
        with sign_col2:
            if st.button("Log to Maintenance DB", width="stretch"):
                st.success(f"Work Order {wo_number} logged successfully.")
                
    st.divider()
    
    ticket_text = f"""
================================================================================
FIELD MAINTENANCE WORK ORDER: {wo_number}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Machine: {curr_m['machine_name']}
Priority: {wo_priority}
Lead Technician: {wo_technician} | Shift: {wo_shift}
Tool State: {curr_m['tool_condition'].upper()} (Wear Severity: {curr_m['wear_percent']:.1f}%)
Feedrate: {curr_m['feedrate']} mm/s | Clamp: {curr_m['clamp_pressure']} bar
================================================================================
DIAGNOSTIC FINDINGS:
Health Score: {curr_m['health_score']}%
Cycle Finalized: {curr_m['finalized'].upper()} | Visual QA: {curr_m['visual_pass'].upper()}
Sensor Observations: {curr_m['observations']}
Flagged Anomaly Points: {curr_m['anomaly_count']} ({curr_m['anomaly_rate']:.2f}%)

PRESCRIBED ACTIONS:
1. Lockout/Tagout machine.
2. Index or replace carbide insert (ISO CNMG-120408-PM).
3. Verify spindle taper runout (< 0.005mm).
4. Verify clamp pressure regulator ({curr_m['clamp_pressure']} bar).
5. Perform test air-cut.

Technician Sign-Off: {wo_technician}   Date: {datetime.now().strftime('%Y-%m-%d')}
================================================================================
    """
    
    st.download_button(
        "Download Work Order (TXT)",
        data=ticket_text.encode("utf-8"),
        file_name=f"{wo_number}.txt",
        mime="text/plain"
    )


# ==============================================================================
# MODULE 7: FLEET AUDIT & HISTORY
# ==============================================================================

elif page == "Fleet Audit & History":
    
    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        m1_id = st.selectbox("First Machine Run", sorted(machine_catalog.keys()), index=0, key="audit_m1")
    with comp_col2:
        m2_id = st.selectbox("Second Machine Run", sorted(machine_catalog.keys()), index=6 if len(machine_catalog) > 6 else 1, key="audit_m2")
        
    m1 = machine_catalog[m1_id]
    m2 = machine_catalog[m2_id]
    
    comp_df = pd.DataFrame({
        "Operating Parameter": [
            "Machine Station",
            "Operational Health Score",
            "Operational Status",
            "Tool Condition",
            "Wear Severity Risk",
            "Programmed Feedrate",
            "Hydraulic Clamp Pressure",
            "Workpiece Material",
            "Machining Finalized",
            "Passed Visual QA",
            "Flagged Anomalies",
            "Peak Spindle Power"
        ],
        f"Unit A ({m1['machine_name']})": [
            m1["machine_name"],
            f"{m1['health_score']}%",
            m1["status_label"],
            m1["tool_condition"].upper(),
            f"{m1['wear_percent']:.1f}%",
            f"{m1['feedrate']} mm/s",
            f"{m1['clamp_pressure']} bar",
            m1["material"].title(),
            m1["finalized"].upper(),
            m1["visual_pass"].upper(),
            f"{m1['anomaly_count']} ({m1['anomaly_rate']:.1f}%)",
            f"{m1['s1_pwr_max']:.3f} W"
        ],
        f"Unit B ({m2['machine_name']})": [
            m2["machine_name"],
            f"{m2['health_score']}%",
            m2["status_label"],
            m2["tool_condition"].upper(),
            f"{m2['wear_percent']:.1f}%",
            f"{m2['feedrate']} mm/s",
            f"{m2['clamp_pressure']} bar",
            m2["material"].title(),
            m2["finalized"].upper(),
            m2["visual_pass"].upper(),
            f"{m2['anomaly_count']} ({m2['anomaly_rate']:.1f}%)",
            f"{m2['s1_pwr_max']:.3f} W"
        ]
    })
    
    safe_dataframe(comp_df, hide_index=True)
    
    st.divider()
    
    st.markdown("#### Complete Fleet Production Log (18 Runs)")
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
    safe_dataframe(fleet_df, hide_index=True)
    
    st.download_button(
        "Download Fleet Log (CSV)",
        data=fleet_df.to_csv(index=False).encode("utf-8"),
        file_name="cnc_fleet_audit_log.csv",
        mime="text/csv"
    )


# ==============================================================================
# FOOTER
# ==============================================================================

st.divider()
render_html("""
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 12px; color: #94A3B8; padding: 10px 0;">
    <div>VERTEX CNC SCADA &nbsp;•&nbsp; Industrial Machine Telemetry & Predictive Maintenance Platform</div>
    <div>100 Hz Telemetry Link &nbsp;•&nbsp; ISO 13374 Condition Monitoring</div>
</div>
""")
