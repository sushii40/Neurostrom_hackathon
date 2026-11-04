from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NeuroStorm | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "processed"

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* GLOBAL APPLICATION */

.stApp {
    background-color: #F8F6F2;
    color: #393631;
    font-family: "Inter", "Segoe UI", sans-serif;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background-color: #EDE5DC;
    border-right: 1px solid #DED2C6;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}

[data-testid="stSidebar"] * {
    color: #393631;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
    color: #A66F59;
    font-weight: 800;
    letter-spacing: 1px;
}

/* MAIN HEADINGS */

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #393631;
    letter-spacing: -0.8px;
    margin-bottom: 5px;
}

.subtitle {
    color: #898078;
    font-size: 14px;
    margin-bottom: 30px;
}

.section-title {
    font-size: 21px;
    font-weight: 750;
    color: #49413B;
    margin-top: 30px;
    margin-bottom: 16px;
}

/* METRIC CARDS */

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E9E0D7;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 3px 12px rgba(90, 70, 50, 0.045);
    transition: all 0.25s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 7px 20px rgba(90, 70, 50, 0.09);
    border-color: #C99C86;
}

div[data-testid="stMetricLabel"] {
    color: #887D73;
    font-size: 13px;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #393631;
    font-weight: 800;
    font-size: 28px;
}

/* NAVIGATION */

[data-testid="stSidebar"] [role="radiogroup"] label {
    background: transparent;
    border-radius: 9px;
    padding: 9px 12px;
    transition: all 0.2s ease;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #DFD2C5;
}

/* BUTTONS */

.stButton > button {
    background: #B77B65;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 9px 20px;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #98644F;
    color: white;
    border: none;
    transform: translateY(-2px);
}

/* INPUTS */

.stSelectbox > div > div,
.stTextInput input,
.stNumberInput input {
    background: #FFFFFF;
    border-radius: 9px;
    border-color: #DED4C9;
    color: #393631;
}

/* DATAFRAMES */

[data-testid="stDataFrame"] {
    border: 1px solid #E8DED4;
    border-radius: 12px;
    overflow: hidden;
}

/* CHART CONTAINERS */

[data-testid="stPlotlyChart"] {
    background: #FFFFFF;
    border: 1px solid #EAE2D9;
    border-radius: 14px;
    padding: 12px;
    box-shadow: 0 3px 12px rgba(90, 70, 50, 0.035);
}

/* ALERTS */

[data-testid="stAlert"] {
    border-radius: 10px;
}

/* DIVIDERS */

hr {
    border-color: #E4DAD0;
}

/* CAPTIONS */

.stCaption {
    color: #93877D;
}

/* FOOTER */

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():

    sensor = pd.read_csv(
        DATA / "cleaned_sensor_data.csv"
    )

    experiments = pd.read_csv(
        DATA / "features" / "process_aware_features.csv"
    )

    anomalies = pd.read_csv(
        DATA / "anomaly_detection" / "isolation_forest_results.csv"
    )

    predictions = pd.read_csv(
        DATA / "model_evaluation" / "loeo_predictions.csv"
    )

    stability = pd.read_csv(
        DATA / "model_evaluation" / "feature_stability.csv"
    )

    comparison = pd.read_csv(
        DATA / "model_evaluation" / "advanced_model_comparison.csv"
    )

    return (
        sensor,
        experiments,
        anomalies,
        predictions,
        stability,
        comparison
    )


try:

    (
        sensor_data,
        experiment_data,
        anomaly_data,
        prediction_data,
        feature_data,
        model_comparison
    ) = load_data()

except Exception as error:

    st.error(f"Unable to load project data: {error}")
    st.stop()


# ============================================================
# CALCULATE PROJECT METRICS
# ============================================================

total_experiments = sensor_data["Experiment_ID"].nunique()

condition_counts = (
    sensor_data.groupby("Experiment_ID")["tool_condition"]
    .first()
    .value_counts()
)

worn_count = int(condition_counts.get("worn", 0))
unworn_count = int(condition_counts.get("unworn", 0))

total_observations = len(sensor_data)

total_anomalies = int(anomaly_data["anomaly"].sum())

anomaly_percentage = (
    total_anomalies / len(anomaly_data) * 100
    if len(anomaly_data) else 0
)

y_true = (
    prediction_data["Actual"].map({
        "unworn": 0,
        "worn": 1
    })
)

y_pred = (
    prediction_data["Predicted"].map({
        "unworn": 0,
        "worn": 1
    })
)

probabilities = prediction_data["Worn_Probability"]

accuracy = accuracy_score(y_true, y_pred)
balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, probabilities)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("# ⚙️ NEUROSTORM")

    st.caption("INTELLIGENT INDUSTRIAL ANALYTICS")

    st.divider()

    st.markdown("### NAVIGATION")

    page = st.radio(
        "Select Module",
        [
            "Dashboard Overview",
            "Machine Monitoring",
            "Tool Wear Prediction",
            "Maintenance Advisor",
            "Anomaly Detection",
            "Machine Performance",
            "Experiment History"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### DATASET STATUS")

    st.success("Dataset Loaded")

    st.metric(
        "Experiments",
        total_experiments
    )

    st.metric(
        "Sensor Observations",
        f"{total_observations:,}"
    )

    st.divider()

    st.caption("NEUROSTORM PREDICTIVE MAINTENANCE")
    st.caption("Machining Tool Wear Analytics")


# ============================================================
# DASHBOARD OVERVIEW
# ============================================================

if page == "Dashboard Overview":

    st.markdown(
        '<div class="main-title">Industrial Predictive Maintenance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Machining Tool Wear Monitoring | '
        'Sensor Analytics | Intelligent Condition Assessment</div>',
        unsafe_allow_html=True
    )

    # KPI CARDS

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Experiments",
            total_experiments,
            help="Number of independent machining experiments"
        )

    with col2:
        st.metric(
            "Worn Tools",
            worn_count,
            help="Experiments labelled as worn"
        )

    with col3:
        st.metric(
            "Unworn Tools",
            unworn_count,
            help="Experiments labelled as unworn"
        )

    with col4:
        st.metric(
            "Sensor Observations",
            f"{total_observations:,}"
        )

    st.markdown('<div class="section-title">Tool Condition Analytics</div>',
                unsafe_allow_html=True)

    left, right = st.columns(2)

    # TOOL CONDITION CHART

    with left:

        condition_df = pd.DataFrame({
            "Condition": ["Worn", "Unworn"],
            "Experiments": [worn_count, unworn_count]
        })

        fig = px.pie(
            condition_df,
            names="Condition",
            values="Experiments",
            hole=0.62,
            title="Tool Condition Distribution",
            color="Condition",
            color_discrete_map={
                "Worn": "#f97316",
                "Unworn": "#22c55e"
            }
        )

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font_color="#49413B",
            legend_title_text=""
        )

        st.plotly_chart(fig, use_container_width=True)

    # ANOMALY CHART

    with right:

        anomaly_summary = pd.DataFrame({
            "Category": ["Normal", "Anomaly"],
            "Observations": [
                len(anomaly_data) - total_anomalies,
                total_anomalies
            ]
        })

        fig = px.bar(
            anomaly_summary,
            x="Category",
            y="Observations",
            color="Category",
            title="Isolation Forest Anomaly Summary",
            color_discrete_map={
                "Normal": "#0ea5e9",
                "Anomaly": "#ef4444"
            },
            text="Observations"
        )

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            showlegend=False,
            font_color="#49413B"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Model Performance Overview</div>',
                unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("LOEO Accuracy", f"{accuracy * 100:.1f}%")
    m2.metric("Balanced Accuracy", f"{balanced_accuracy * 100:.1f}%")
    m3.metric("F1 Score", f"{f1:.3f}")
    m4.metric("ROC-AUC", f"{roc_auc:.3f}")

    st.caption(
        "Metrics calculated from Leave-One-Experiment-Out predictions "
        "across 18 independent experiments. These are cross-validation "
        "results, not independent external test results."
    )

    st.markdown('<div class="section-title">Recent Experiment Predictions</div>',
                unsafe_allow_html=True)

    recent = prediction_data.sort_values(
        "Experiment_ID",
        ascending=False
    ).head(8)

    display_recent = recent.rename(columns={
        "Experiment_ID": "Experiment",
        "Actual": "Actual Condition",
        "Predicted": "Predicted Condition",
        "Worn_Probability": "Worn Probability",
        "Selected_Model": "Selected Algorithm"
    })

    st.dataframe(
        display_recent,
        use_container_width=True,
        hide_index=True
    )

    st.markdown('<div class="section-title">Anomaly Detection Summary</div>',
                unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)

    a1.metric(
        "Anomalies Detected",
        f"{total_anomalies:,}"
    )

    a2.metric(
        "Observations Analyzed",
        f"{len(anomaly_data):,}"
    )

    a3.metric(
        "Anomaly Rate",
        f"{anomaly_percentage:.2f}%"
    )


# ============================================================
# OTHER MODULES - TO BE DEVELOPED
# ============================================================

elif page == "Machine Monitoring":

    st.markdown(
        '<div class="main-title">Machine Monitoring</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Machine A | Historical Sensor Monitoring and Process Intelligence</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Monitoring uses recorded sensor observations from the selected "
        "experiment. This is dataset replay, not live hardware telemetry."
    )

    # --------------------------------------------------------
    # EXPERIMENT AND PROCESS SELECTION
    # --------------------------------------------------------

    experiments_available = sorted(
        sensor_data["Experiment_ID"].dropna().unique().tolist()
    )

    col1, col2 = st.columns(2)

    with col1:
        selected_experiment = st.selectbox(
            "Select Machine / Experiment",
            experiments_available,
            format_func=lambda x: f"Machine A | Experiment {int(x)}"
        )

    experiment_rows = sensor_data[
        sensor_data["Experiment_ID"] == selected_experiment
    ].copy()

    available_processes = sorted(
        experiment_rows["Machining_Process"].dropna().unique().tolist()
    )

    process_options = ["All Processes"] + available_processes

    with col2:
        selected_process = st.selectbox(
            "Select Machining Process",
            process_options
        )

    if selected_process != "All Processes":

        monitoring_data = experiment_rows[
            experiment_rows["Machining_Process"] == selected_process
        ].copy()

    else:
        monitoring_data = experiment_rows.copy()

    if monitoring_data.empty:

        st.warning("No sensor observations available for this selection.")
        st.stop()

    if "M1_sequence_number" in monitoring_data.columns:

        monitoring_data = monitoring_data.sort_values(
            "M1_sequence_number"
        )

    # --------------------------------------------------------
    # MACHINE CONDITION
    # --------------------------------------------------------

    tool_condition = (
        experiment_rows["tool_condition"].dropna().iloc[0]
    )

    material = (
        experiment_rows["material"].dropna().iloc[0]
        if experiment_rows["material"].notna().any()
        else "Not available"
    )

    condition_col1, condition_col2, condition_col3 = st.columns(3)

    with condition_col1:
        st.metric(
            "Tool Condition",
            str(tool_condition).upper()
        )

    with condition_col2:
        st.metric(
            "Material",
            str(material)
        )

    with condition_col3:
        st.metric(
            "Sensor Observations",
            f"{len(monitoring_data):,}"
        )

    if tool_condition == "worn":

        st.warning(
            "This experiment is labelled WORN in the historical dataset. "
            "Inspect the tool and review the sensor behaviour."
        )

    else:

        st.success(
            "This experiment is labelled UNWORN in the historical dataset."
        )

    st.divider()

    # --------------------------------------------------------
    # OPERATING CONDITIONS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Operating Conditions</div>',
        unsafe_allow_html=True
    )

    op1, op2, op3 = st.columns(3)

    feedrate = experiment_rows["feedrate"].iloc[0]
    pressure = experiment_rows["clamp_pressure"].iloc[0]

    with op1:
        st.metric(
            "Feedrate",
            f"{feedrate:.3f}" if pd.notna(feedrate) else "N/A"
        )

    with op2:
        st.metric(
            "Clamp Pressure",
            f"{pressure:.3f}" if pd.notna(pressure) else "N/A"
        )

    with op3:
        st.metric(
            "Machining Processes",
            f"{experiment_rows['Machining_Process'].nunique()}"
        )

    st.divider()

    # --------------------------------------------------------
    # SENSOR READINGS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Current Sensor Snapshot</div>',
        unsafe_allow_html=True
    )

    latest = monitoring_data.iloc[-1]

    sensor_groups = [
        ("X Axis", [
            "X1_ActualVelocity",
            "X1_ActualAcceleration",
            "X1_CurrentFeedback",
            "X1_OutputPower"
        ]),
        ("Y Axis", [
            "Y1_ActualVelocity",
            "Y1_ActualAcceleration",
            "Y1_CurrentFeedback",
            "Y1_OutputPower"
        ]),
        ("Z Axis", [
            "Z1_ActualVelocity",
            "Z1_ActualAcceleration"
        ]),
        ("Spindle / S Axis", [
            "S1_ActualVelocity",
            "S1_ActualAcceleration",
            "S1_CurrentFeedback",
            "S1_OutputPower"
        ])
    ]

    for group_name, columns in sensor_groups:

        available_columns = [
            column for column in columns
            if column in monitoring_data.columns
        ]

        if not available_columns:
            continue

        st.markdown(f"#### {group_name}")

        metric_columns = st.columns(len(available_columns))

        for metric_col, column in zip(metric_columns, available_columns):

            value = latest[column]

            with metric_col:

                st.metric(
                    label=column.replace(group_name[0] + "1_", ""),
                    value=f"{value:.4f}" if pd.notna(value) else "N/A"
                )

    st.caption(
        "Values represent the latest recorded observation in the selected "
        "experiment/process. Units are not assumed where dataset "
        "documentation does not specify them."
    )

    st.divider()

    # --------------------------------------------------------
    # SENSOR TREND ANALYSIS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Sensor Trend Analysis</div>',
        unsafe_allow_html=True
    )

    available_signals = [
        column for column in monitoring_data.columns
        if column.endswith((
            "_ActualVelocity",
            "_ActualAcceleration",
            "_CurrentFeedback",
            "_OutputPower",
            "_ActualPosition"
        ))
    ]

    chart_col1, chart_col2 = st.columns([1, 2])

    with chart_col1:

        selected_signal = st.selectbox(
            "Choose Sensor Signal",
            available_signals
        )

    x_axis = (
        "M1_sequence_number"
        if "M1_sequence_number" in monitoring_data.columns
        else monitoring_data.index.name or "index"
    )

    chart_data = monitoring_data.copy()

    if x_axis == "index":
        chart_data["index"] = chart_data.index

    with chart_col2:

        st.caption(f"Trend: {selected_signal}")

    trend = px.line(
        chart_data,
        x=x_axis,
        y=selected_signal,
        title=f"{selected_signal} over Machining Sequence",
        labels={
            x_axis: "Machining Sequence",
            selected_signal: "Recorded Sensor Value"
        }
    )

    trend.update_layout(
        template="plotly_white",
        height=390,
        margin=dict(l=20, r=20, t=55, b=20)
    )

    st.plotly_chart(trend, use_container_width=True)

    # --------------------------------------------------------
    # CURRENT FEEDBACK COMPARISON
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Axis Current Feedback</div>',
        unsafe_allow_html=True
    )

    current_columns = [
        column for column in [
            "X1_CurrentFeedback",
            "Y1_CurrentFeedback",
            "S1_CurrentFeedback"
        ]
        if column in monitoring_data.columns
    ]

    if current_columns:

        current_data = monitoring_data[
            [x_axis] + current_columns
        ].melt(
            id_vars=x_axis,
            var_name="Axis",
            value_name="Current Feedback"
        )

        current_chart = px.line(
            current_data,
            x=x_axis,
            y="Current Feedback",
            color="Axis",
            title="Current Feedback Across Machine Axes"
        )

        current_chart.update_layout(
            template="plotly_white",
            height=380
        )

        st.plotly_chart(
            current_chart,
            use_container_width=True
        )

    # --------------------------------------------------------
    # RAW SENSOR OBSERVATIONS
    # --------------------------------------------------------

    with st.expander("View Recorded Sensor Observations"):

        st.dataframe(
            monitoring_data,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Selected Sensor Data",
            data=monitoring_data.to_csv(index=False).encode("utf-8"),
            file_name=f"experiment_{selected_experiment}_monitoring.csv",
            mime="text/csv"
        )

    st.caption(
        "NEUROSTORM | Historical Machine Monitoring"
    )


elif page == "Tool Wear Prediction":

    st.markdown(
        '<div class="main-title">AI Tool Wear Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Machine Learning | Experiment-Level Tool Health Assessment</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Predictions shown here are Leave-One-Experiment-Out cross-validation "
        "results. Each experiment was held out during its prediction. "
        "These results demonstrate model evaluation, not live deployment."
    )

    # --------------------------------------------------------
    # PREPARE PREDICTION DATA
    # --------------------------------------------------------

    predictions = prediction_data.copy()

    predictions["Actual"] = (
        predictions["Actual"].astype(str).str.lower()
    )

    predictions["Predicted"] = (
        predictions["Predicted"].astype(str).str.lower()
    )

    predictions["Worn_Probability"] = pd.to_numeric(
        predictions["Worn_Probability"],
        errors="coerce"
    )

    predictions["Prediction_Status"] = np.where(
        predictions["Actual"] == predictions["Predicted"],
        "Correct",
        "Incorrect"
    )

    predictions["Risk_Level"] = pd.cut(
        predictions["Worn_Probability"],
        bins=[-0.001, 0.30, 0.70, 1.001],
        labels=["Lower", "Moderate", "Higher"]
    )

    # --------------------------------------------------------
    # EXPERIMENT SELECTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Select Experiment</div>',
        unsafe_allow_html=True
    )

    available_experiments = sorted(
        predictions["Experiment_ID"].unique().tolist()
    )

    selected_prediction_experiment = st.selectbox(
        "Choose Experiment",
        available_experiments,
        format_func=lambda x: f"Machine A | Experiment {int(x)}"
    )

    selected_prediction = predictions[
        predictions["Experiment_ID"] == selected_prediction_experiment
    ].iloc[0]

    # --------------------------------------------------------
    # SELECTED EXPERIMENT RESULTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Experiment Assessment</div>',
        unsafe_allow_html=True
    )

    actual_condition = selected_prediction["Actual"]
    predicted_condition = selected_prediction["Predicted"]
    worn_probability = selected_prediction["Worn_Probability"]
    selected_model = selected_prediction["Selected_Model"]
    prediction_status = selected_prediction["Prediction_Status"]

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.metric(
            "Actual Condition",
            actual_condition.upper()
        )

    with p2:
        st.metric(
            "Predicted Condition",
            predicted_condition.upper()
        )

    with p3:
        st.metric(
            "Worn Probability",
            f"{worn_probability * 100:.2f}%"
        )

    with p4:
        st.metric(
            "Prediction Result",
            prediction_status.upper()
        )

    if prediction_status == "Correct":

        st.success(
            "The model correctly classified this held-out experiment."
        )

    else:

        st.error(
            "The model incorrectly classified this experiment. "
            "This case should be reviewed before relying on the prediction."
        )

    st.markdown(
        f"**Selected model:** `{selected_model}`"
    )

    # --------------------------------------------------------
    # WORN PROBABILITY GAUGE
    # --------------------------------------------------------

    gauge_col1, gauge_col2 = st.columns([1, 1])

    with gauge_col1:

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(worn_probability * 100),
                number={"suffix": "%"},
                title={"text": "Estimated Worn Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#B77B65"},
                    "steps": [
                        {"range": [0, 30], "color": "#E6EDE4"},
                        {"range": [30, 70], "color": "#F5EBD8"},
                        {"range": [70, 100], "color": "#F1DDD6"}
                    ]
                }
            )
        )

        gauge.update_layout(
            template="plotly_white",
            height=330
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

    with gauge_col2:

        st.markdown("#### Probability Interpretation")

        if worn_probability >= 0.70:

            st.error(
                "Higher estimated wear probability. "
                "Prioritize tool inspection and review the sensor history."
            )

        elif worn_probability >= 0.30:

            st.warning(
                "Moderate estimated wear probability. "
                "Review process behaviour and inspect the tool if warranted."
            )

        else:

            st.success(
                "Lower estimated wear probability. "
                "Continue routine inspection and monitoring."
            )

        st.caption(
            "These probability bands are dashboard interpretation thresholds, "
            "not validated maintenance or safety limits."
        )

        st.markdown("#### Model Information")

        st.write(f"**Algorithm:** {selected_model}")
        st.write(f"**Experiment:** {int(selected_prediction_experiment)}")
        st.write(f"**Actual label:** {actual_condition}")
        st.write(f"**Predicted label:** {predicted_condition}")

    st.divider()

    # --------------------------------------------------------
    # OVERALL MODEL PERFORMANCE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Overall Prediction Performance</div>',
        unsafe_allow_html=True
    )

    y_true_local = predictions["Actual"].map({
        "unworn": 0,
        "worn": 1
    })

    y_pred_local = predictions["Predicted"].map({
        "unworn": 0,
        "worn": 1
    })

    valid = y_true_local.notna() & y_pred_local.notna()

    local_accuracy = accuracy_score(
        y_true_local[valid],
        y_pred_local[valid]
    )

    local_balanced_accuracy = balanced_accuracy_score(
        y_true_local[valid],
        y_pred_local[valid]
    )

    local_f1 = f1_score(
        y_true_local[valid],
        y_pred_local[valid],
        zero_division=0
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("LOEO Accuracy", f"{local_accuracy * 100:.1f}%")
    m2.metric("Balanced Accuracy", f"{local_balanced_accuracy * 100:.1f}%")
    m3.metric("F1 Score", f"{local_f1:.3f}")
    m4.metric("Experiments Evaluated", len(predictions))

    st.caption(
        "Metrics are calculated from the held-out experiment predictions "
        "shown in this dataset."
    )

    st.divider()

    # --------------------------------------------------------
    # ACTUAL VS PREDICTED COMPARISON
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Actual vs Predicted Tool Condition</div>',
        unsafe_allow_html=True
    )

    comparison_table = pd.crosstab(
        predictions["Actual"].str.upper(),
        predictions["Predicted"].str.upper()
    )

    comparison_col1, comparison_col2 = st.columns(2)

    with comparison_col1:

        st.markdown("#### Classification Comparison")

        st.dataframe(
            comparison_table,
            use_container_width=True
        )

    with comparison_col2:

        st.markdown("#### Experiment Prediction Status")

        status_chart = px.histogram(
            predictions,
            x="Prediction_Status",
            color="Prediction_Status",
            title="Correct vs Incorrect Predictions"
        )

        status_chart.update_layout(
            template="plotly_white",
            height=300,
            showlegend=False
        )

        st.plotly_chart(
            status_chart,
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------------
    # EXPERIMENT PROBABILITY ANALYSIS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Wear Probability Across Experiments</div>',
        unsafe_allow_html=True
    )

    probability_chart = px.bar(
        predictions.sort_values("Experiment_ID"),
        x="Experiment_ID",
        y="Worn_Probability",
        color="Actual",
        hover_data=[
            "Predicted",
            "Selected_Model",
            "Prediction_Status"
        ],
        title="Estimated Worn Probability by Experiment",
        labels={
            "Experiment_ID": "Experiment",
            "Worn_Probability": "Worn Probability",
            "Actual": "Actual Tool Condition"
        }
    )

    probability_chart.update_layout(
        template="plotly_white",
        height=420,
        yaxis_tickformat=".0%"
    )

    probability_chart.add_hline(
        y=0.5,
        line_dash="dash",
        annotation_text="50% reference"
    )

    st.plotly_chart(
        probability_chart,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL SELECTION ANALYSIS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Model Selection Analysis</div>',
        unsafe_allow_html=True
    )

    model_counts = (
        predictions["Selected_Model"]
        .value_counts()
        .reset_index()
    )

    model_counts.columns = ["Model", "Experiments Selected"]

    model_chart = px.bar(
        model_counts,
        x="Model",
        y="Experiments Selected",
        color="Model",
        title="Model Selected Across Held-Out Experiments"
    )

    model_chart.update_layout(
        template="plotly_white",
        height=350,
        showlegend=False
    )

    st.plotly_chart(
        model_chart,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # INCORRECT PREDICTIONS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Experiments Requiring Review</div>',
        unsafe_allow_html=True
    )

    incorrect = predictions[
        predictions["Prediction_Status"] == "Incorrect"
    ]

    if not incorrect.empty:

        st.warning(
            f"{len(incorrect)} experiments were incorrectly classified "
            "during cross-validation."
        )

        st.dataframe(
            incorrect[
                [
                    "Experiment_ID",
                    "Actual",
                    "Predicted",
                    "Worn_Probability",
                    "Selected_Model"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "No incorrect classifications in the available predictions."
        )

    # --------------------------------------------------------
    st.caption(
        "NEUROSTORM | AI Tool Wear Prediction | LOEO Model Evaluation"
    )



elif page == "Maintenance Advisor":

    st.markdown(
        '<div class="main-title">Intelligent Maintenance Advisor</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Evidence-Based Tool Health Assessment and Maintenance Guidance</div>',
        unsafe_allow_html=True
    )

    st.info(
        "This advisor combines held-out tool-wear predictions and historical "
        "Isolation Forest anomaly results. Recommendations are preliminary "
        "inspection guidance, not confirmed fault diagnoses."
    )

    # --------------------------------------------------------
    # EXPERIMENT SELECTION
    # --------------------------------------------------------

    available_ids = sorted(
        prediction_data["Experiment_ID"].unique().tolist()
    )

    selected_id = st.selectbox(
        "Select Machine A Experiment",
        available_ids,
        format_func=lambda x: f"Experiment {int(x)}"
    )

    prediction_row = prediction_data[
        prediction_data["Experiment_ID"] == selected_id
    ].iloc[0]

    worn_probability = float(prediction_row["Worn_Probability"])
    predicted_condition = str(prediction_row["Predicted"]).lower()
    actual_condition = str(prediction_row["Actual"]).lower()
    selected_model = prediction_row["Selected_Model"]

    # --------------------------------------------------------
    # ANOMALY ANALYSIS
    # --------------------------------------------------------

    selected_anomalies = anomaly_data[
        anomaly_data["Experiment_ID"] == selected_id
    ].copy()

    observation_count = len(selected_anomalies)
    anomaly_count = int(selected_anomalies["anomaly"].sum())

    anomaly_rate = (
        anomaly_count / observation_count * 100
        if observation_count else 0
    )

    overall_anomaly_rate = (
        anomaly_data["anomaly"].mean() * 100
        if len(anomaly_data) else 0
    )

    # --------------------------------------------------------
    # CONDITION SUMMARY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Machine Health Assessment</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Predicted Tool Condition",
        predicted_condition.upper()
    )

    c2.metric(
        "Worn Probability",
        f"{worn_probability * 100:.2f}%"
    )

    c3.metric(
        "Flagged Anomalies",
        f"{anomaly_count:,}"
    )

    c4.metric(
        "Experiment Anomaly Rate",
        f"{anomaly_rate:.2f}%"
    )

    st.caption(
        f"Prediction model: {selected_model} | "
        f"Historical reference label: {actual_condition.upper()}"
    )

    st.caption(
        "The historical reference label is the dataset ground truth. "
        "It is shown for evaluation and is not an input to the recommendation rules."
    )

    st.divider()

    # --------------------------------------------------------
    # RISK INTERPRETATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Assessment Summary</div>',
        unsafe_allow_html=True
    )

    if worn_probability >= 0.70:
        wear_level = "Higher"
        st.error(
            "Higher estimated tool-wear probability. "
            "Prioritize tool-condition inspection."
        )

    elif worn_probability >= 0.30:
        wear_level = "Moderate"
        st.warning(
            "Moderate estimated tool-wear probability. "
            "Review the tool and machining history."
        )

    else:
        wear_level = "Lower"
        st.success(
            "Lower estimated tool-wear probability. "
            "Continue routine monitoring and inspection."
        )

    if anomaly_rate > overall_anomaly_rate * 1.5:
        anomaly_level = "Elevated"
        st.warning(
            "The selected experiment has a higher anomaly rate "
            "than the dataset-wide reference."
        )

    elif anomaly_count > 0:
        anomaly_level = "Detected"
        st.info(
            "Unusual observations were detected. "
            "Review their associated sensor readings and process stages."
        )

    else:
        anomaly_level = "None detected"
        st.success(
            "No observations were flagged in this experiment."
        )

    st.caption(
        f"Dataset-wide anomaly reference: {overall_anomaly_rate:.2f}%. "
        "The 1.5x comparison is an exploratory dashboard heuristic, "
        "not a validated safety threshold."
    )

    st.divider()

    # --------------------------------------------------------
    # PROCESS-LEVEL INVESTIGATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Processes Requiring Attention</div>',
        unsafe_allow_html=True
    )

    process_summary = (
        selected_anomalies.groupby("Machining_Process")
        .agg(
            Observations=("anomaly", "count"),
            Anomalies=("anomaly", "sum")
        )
        .reset_index()
    )

    if not process_summary.empty:

        process_summary["Anomaly_Rate"] = (
            process_summary["Anomalies"]
            / process_summary["Observations"] * 100
        )

        process_summary = process_summary.sort_values(
            ["Anomaly_Rate", "Anomalies"],
            ascending=False
        )

        st.dataframe(
            process_summary,
            use_container_width=True,
            hide_index=True
        )

        top_process = process_summary.iloc[0]

        st.write(
            f"**Highest observed process anomaly rate:** "
            f"{top_process['Machining_Process']} "
            f"({top_process['Anomaly_Rate']:.2f}%)"
        )

    else:
        st.info("No process-level anomaly records are available.")

    st.divider()

    # --------------------------------------------------------
    # MAINTENANCE RECOMMENDATIONS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Recommended Precautions</div>',
        unsafe_allow_html=True
    )

    recommendations = []

    if worn_probability >= 0.70:

        recommendations.append({
            "Priority": "High",
            "Recommendation": "Inspect the cutting tool",
            "Reason": "The model estimates a higher probability of tool wear.",
            "Action": "Inspect the cutting edge for visible wear or damage. "
                      "Follow the manufacturer's tool replacement criteria."
        })

    elif worn_probability >= 0.30:

        recommendations.append({
            "Priority": "Medium",
            "Recommendation": "Schedule a tool-condition inspection",
            "Reason": "The estimated tool-wear probability is moderate.",
            "Action": "Review tool condition and compare the machining results "
                      "with previous inspection records."
        })

    else:

        recommendations.append({
            "Priority": "Routine",
            "Recommendation": "Continue scheduled tool inspections",
            "Reason": "The model estimates a lower probability of tool wear.",
            "Action": "Maintain the normal inspection schedule. "
                      "A low model probability does not guarantee tool health."
        })

    if anomaly_count > 0:

        recommendations.append({
            "Priority": "Medium",
            "Recommendation": "Investigate unusual sensor observations",
            "Reason": f"{anomaly_count} observations were flagged by Isolation Forest.",
            "Action": "Review the associated sensor readings and repeatability "
                      "of the affected machining process."
        })

    if anomaly_rate > overall_anomaly_rate * 1.5:

        recommendations.append({
            "Priority": "High",
            "Recommendation": "Review the affected machining process",
            "Reason": "The experiment anomaly rate is elevated relative to "
                      "the dataset-wide reference.",
            "Action": "Review process settings, workpiece clamping, tool "
                      "condition and machining records before the next run."
        })

    if not recommendations:
        recommendations.append({
            "Priority": "Routine",
            "Recommendation": "Continue routine monitoring",
            "Reason": "No elevated wear probability or flagged anomalies "
                      "were identified by the available analyses.",
            "Action": "Continue standard inspection and monitor future "
                      "sensor observations."
        })

    for index, item in enumerate(recommendations, start=1):

        with st.container(border=True):

            priority = item["Priority"]

            if priority == "High":
                st.error(f"{priority} Priority | {item['Recommendation']}")

            elif priority == "Medium":
                st.warning(f"{priority} Priority | {item['Recommendation']}")

            else:
                st.success(f"{priority} Priority | {item['Recommendation']}")

            st.write(f"**Evidence:** {item['Reason']}")
            st.write(f"**Suggested precaution:** {item['Action']}")

    st.divider()

    # --------------------------------------------------------
    # IMPORTANT LIMITATIONS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Diagnostic Limitations</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The available dataset supports tool-wear classification and "
        "unusual sensor-behaviour detection. It does not contain confirmed "
        "mechanical failure-mode labels."
    )

    st.write(
        "Therefore, this advisor does not claim to diagnose bearing failure, "
        "spindle overheating, tool breakage or other specific faults."
    )

    st.write(
        "Temperature and torque measurements are not available in the "
        "current cleaned dataset. Recommendations are preliminary checks "
        "based on the available evidence."
    )

    st.caption(
        "NEUROSTORM | Intelligent Maintenance Advisor"
    )


elif page == "Anomaly Detection":

    st.markdown(
        '<div class="main-title">Anomaly Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Isolation Forest | Abnormal Sensor Behaviour Analysis</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Anomalies are unusual observations identified by Isolation Forest. "
        "They are indicators for investigation, not confirmed machine failures."
    )

    # --------------------------------------------------------
    # MATCH ANOMALY RESULTS WITH SENSOR OBSERVATIONS
    # --------------------------------------------------------

    anomaly_sensor = sensor_data[
        sensor_data["Machining_Process"].astype(str).str.lower() != "starting"
    ].copy().reset_index(drop=True)

    anomaly_results = anomaly_data.copy().reset_index(drop=True)

    key_columns = [
        "Experiment_ID",
        "Machining_Process",
        "tool_condition"
    ]

    matching_verified = (
        len(anomaly_sensor) == len(anomaly_results)
        and all(
            anomaly_sensor[column].astype(str).equals(
                anomaly_results[column].astype(str)
            )
            for column in key_columns
        )
    )

    if matching_verified:

        anomaly_sensor["anomaly"] = anomaly_results["anomaly"].values
        anomaly_sensor["anomaly_score"] = anomaly_results["anomaly_score"].values

    else:

        st.warning(
            "Sensor-to-anomaly row alignment could not be verified. "
            "Sensor-level anomaly comparisons are disabled to avoid "
            "displaying incorrect readings."
        )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    filter1, filter2 = st.columns(2)

    with filter1:

        experiment_options = ["All Experiments"] + sorted(
            anomaly_data["Experiment_ID"].unique().tolist()
        )

        selected_anomaly_experiment = st.selectbox(
            "Filter by Experiment",
            experiment_options
        )

    with filter2:

        process_options = ["All Processes"] + sorted(
            anomaly_data["Machining_Process"].dropna().unique().tolist()
        )

        selected_anomaly_process = st.selectbox(
            "Filter by Machining Process",
            process_options
        )

    filtered = anomaly_data.copy()

    if selected_anomaly_experiment != "All Experiments":

        filtered = filtered[
            filtered["Experiment_ID"] == selected_anomaly_experiment
        ]

    if selected_anomaly_process != "All Processes":

        filtered = filtered[
            filtered["Machining_Process"] == selected_anomaly_process
        ]

    flagged = filtered[filtered["anomaly"] == 1]
    normal = filtered[filtered["anomaly"] == 0]

    total = len(filtered)
    anomaly_count = len(flagged)

    anomaly_rate = (
        anomaly_count / total * 100
        if total else 0
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Anomaly Overview</div>',
        unsafe_allow_html=True
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Observations Analysed", f"{total:,}")
    k2.metric("Flagged Anomalies", f"{anomaly_count:,}")
    k3.metric("Normal Observations", f"{len(normal):,}")
    k4.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")

    st.divider()

    # --------------------------------------------------------
    # EXPERIMENT ANALYSIS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Experiment-wise Anomaly Analysis</div>',
        unsafe_allow_html=True
    )

    experiment_summary = (
        filtered.groupby("Experiment_ID")
        .agg(
            Total_Observations=("anomaly", "count"),
            Anomalies=("anomaly", "sum")
        )
        .reset_index()
    )

    experiment_summary["Anomaly_Rate"] = (
        experiment_summary["Anomalies"]
        / experiment_summary["Total_Observations"] * 100
    )

    experiment_summary = experiment_summary.sort_values(
        "Anomaly_Rate",
        ascending=False
    )

    fig_experiment = px.bar(
        experiment_summary,
        x="Experiment_ID",
        y="Anomaly_Rate",
        color="Anomaly_Rate",
        title="Anomaly Rate by Experiment",
        labels={
            "Experiment_ID": "Experiment",
            "Anomaly_Rate": "Anomaly Rate (%)"
        }
    )

    fig_experiment.update_layout(
        template="plotly_white",
        height=380
    )

    st.plotly_chart(
        fig_experiment,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # PROCESS ANALYSIS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Machining Process Analysis</div>',
        unsafe_allow_html=True
    )

    process_summary = (
        filtered.groupby("Machining_Process")
        .agg(
            Total_Observations=("anomaly", "count"),
            Anomalies=("anomaly", "sum")
        )
        .reset_index()
    )

    process_summary["Anomaly_Rate"] = (
        process_summary["Anomalies"]
        / process_summary["Total_Observations"] * 100
    )

    process_summary = process_summary.sort_values(
        "Anomaly_Rate",
        ascending=False
    )

    fig_process = px.bar(
        process_summary,
        x="Machining_Process",
        y="Anomaly_Rate",
        color="Anomaly_Rate",
        title="Anomaly Rate Across Machining Processes",
        labels={
            "Machining_Process": "Machining Process",
            "Anomaly_Rate": "Anomaly Rate (%)"
        }
    )

    fig_process.update_layout(
        template="plotly_white",
        height=400,
        xaxis_tickangle=-35
    )

    st.plotly_chart(
        fig_process,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # TOOL CONDITION COMPARISON
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Anomalies by Tool Condition</div>',
        unsafe_allow_html=True
    )

    condition_summary = (
        filtered.groupby("tool_condition")
        .agg(
            Total_Observations=("anomaly", "count"),
            Anomalies=("anomaly", "sum")
        )
        .reset_index()
    )

    condition_summary["Anomaly_Rate"] = (
        condition_summary["Anomalies"]
        / condition_summary["Total_Observations"] * 100
    )

    condition_col1, condition_col2 = st.columns(2)

    with condition_col1:

        fig_condition = px.bar(
            condition_summary,
            x="tool_condition",
            y="Anomaly_Rate",
            color="tool_condition",
            title="Anomaly Rate by Tool Condition",
            labels={
                "tool_condition": "Tool Condition",
                "Anomaly_Rate": "Anomaly Rate (%)"
            }
        )

        fig_condition.update_layout(
            template="plotly_white",
            height=350
        )

        st.plotly_chart(
            fig_condition,
            use_container_width=True
        )

    with condition_col2:

        fig_score = px.histogram(
            filtered,
            x="anomaly_score",
            color="anomaly",
            nbins=35,
            title="Isolation Forest Score Distribution",
            labels={
                "anomaly_score": "Anomaly Score",
                "anomaly": "Anomaly Flag"
            }
        )

        fig_score.update_layout(
            template="plotly_white",
            height=350
        )

        st.plotly_chart(
            fig_score,
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------------
    # SENSOR-LEVEL ANOMALY INVESTIGATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Sensor Behaviour Investigation</div>',
        unsafe_allow_html=True
    )

    if matching_verified:

        filtered_sensor = anomaly_sensor.copy()

        if selected_anomaly_experiment != "All Experiments":

            filtered_sensor = filtered_sensor[
                filtered_sensor["Experiment_ID"]
                == selected_anomaly_experiment
            ]

        if selected_anomaly_process != "All Processes":

            filtered_sensor = filtered_sensor[
                filtered_sensor["Machining_Process"]
                == selected_anomaly_process
            ]

        flagged_sensor = filtered_sensor[
            filtered_sensor["anomaly"] == 1
        ].copy()

        sensor_candidates = [
            "X1_CurrentFeedback",
            "Y1_CurrentFeedback",
            "S1_CurrentFeedback",
            "X1_ActualVelocity",
            "Y1_ActualVelocity",
            "Z1_ActualVelocity",
            "S1_ActualVelocity",
            "X1_OutputPower",
            "Y1_OutputPower",
            "S1_OutputPower",
            "X1_ActualAcceleration",
            "Y1_ActualAcceleration",
            "Z1_ActualAcceleration",
            "S1_ActualAcceleration"
        ]

        sensor_candidates = [
            column for column in sensor_candidates
            if column in filtered_sensor.columns
        ]

        if not flagged_sensor.empty and sensor_candidates:

            selected_sensor = st.selectbox(
                "Select Sensor for Investigation",
                sensor_candidates
            )

            sensor_plot_data = filtered_sensor.copy()

            sensor_plot_data["Observation Type"] = np.where(
                sensor_plot_data["anomaly"] == 1,
                "Flagged Anomaly",
                "Normal Observation"
            )

            x_column = (
                "M1_sequence_number"
                if "M1_sequence_number" in sensor_plot_data.columns
                else "Experiment_ID"
            )

            fig_sensor = px.scatter(
                sensor_plot_data,
                x=x_column,
                y=selected_sensor,
                color="Observation Type",
                hover_data=[
                    "Experiment_ID",
                    "Machining_Process",
                    "anomaly_score"
                ],
                title=f"{selected_sensor}: Normal vs Flagged Observations"
            )

            fig_sensor.update_layout(
                template="plotly_white",
                height=420
            )

            st.plotly_chart(
                fig_sensor,
                use_container_width=True
            )

            st.caption(
                "Highlighted points are observations flagged by Isolation Forest. "
                "Their sensor values are not necessarily the cause of the anomaly."
            )

        else:

            st.info(
                "No flagged sensor observations are available for the current filters."
            )

    # --------------------------------------------------------
    # ANOMALY RECORDS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Flagged Anomaly Records</div>',
        unsafe_allow_html=True
    )

    if not flagged.empty:

        if st.checkbox("Show only flagged anomaly records", value=True):

            st.dataframe(
                flagged.sort_values(
                    "anomaly_score",
                    ascending=True
                ),
                use_container_width=True,
                hide_index=True
            )

        if matching_verified and not flagged_sensor.empty:

            with st.expander("View associated sensor readings"):

                st.dataframe(
                    flagged_sensor,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "Download Anomaly Sensor Records",
                    data=flagged_sensor.to_csv(index=False).encode("utf-8"),
                    file_name="neurostorm_anomaly_sensor_records.csv",
                    mime="text/csv"
                )

        st.download_button(
            "Download Anomaly Analysis",
            data=flagged.to_csv(index=False).encode("utf-8"),
            file_name="neurostorm_anomaly_results.csv",
            mime="text/csv"
        )

    else:

        st.success(
            "No anomalies were flagged for the selected filters."
        )

    st.caption(
        "NEUROSTORM | Anomaly Intelligence | Isolation Forest Analysis"
    )



elif page == "Machine Performance":

    st.title("Machine Performance")
    st.caption(
        "Historical machine condition and operating performance "
        "based on the selected experiment."
    )

    st.info(
        "This page displays recorded dataset observations. "
        "It is not a live machine-monitoring connection."
    )

    machine_experiments = sorted(
        sensor_data["Experiment_ID"].dropna().unique(),
        key=str
    )

    selected_exp = st.selectbox(
        "Select Machine Experiment",
        machine_experiments,
        format_func=lambda x: f"Machine A | Experiment {x}"
    )

    selected_sensor = sensor_data[
        sensor_data["Experiment_ID"].astype(str) == str(selected_exp)
    ].copy()

    if "M1_sequence_number" in selected_sensor.columns:
        selected_sensor = selected_sensor.sort_values(
            "M1_sequence_number"
        )

    if selected_sensor.empty:
        st.warning("No sensor observations are available.")
    else:

        latest = selected_sensor.iloc[-1]

        prediction_row = prediction_data[
            prediction_data["Experiment_ID"].astype(str) == str(selected_exp)
        ]

        anomaly_rows = anomaly_data[
            anomaly_data["Experiment_ID"].astype(str) == str(selected_exp)
        ]

        anomaly_count = int(anomaly_rows["anomaly"].sum())
        anomaly_total = len(anomaly_rows)
        anomaly_rate = (
            anomaly_count / anomaly_total * 100
            if anomaly_total else 0
        )

        tool_condition = latest.get("tool_condition", "Not recorded")

        predicted_condition = "Not available"
        worn_probability = None

        if not prediction_row.empty:
            pred = prediction_row.iloc[0]
            predicted_condition = (
                "Worn" if str(pred["Predicted"]).strip().lower() in ("1", "worn", "true") else "Unworn"
            )
            worn_probability = float(pred["Worn_Probability"]) * 100

        st.subheader("Current Recorded Condition")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Recorded Tool Condition", str(tool_condition).title())
        c2.metric("Predicted Condition", predicted_condition)

        c3.metric(
            "Worn Probability",
            f"{worn_probability:.1f}%"
            if worn_probability is not None else "N/A"
        )

        c4.metric("Flagged Anomalies", f"{anomaly_count:,}")

        st.divider()

        st.subheader("Operating Parameters")

        parameter_columns = st.columns(4)

        parameters = [
            ("Material", "material"),
            ("Feedrate", "feedrate"),
            ("Clamp Pressure", "clamp_pressure"),
            ("Sensor Observations", None)
        ]

        for column, (label, field) in zip(parameter_columns, parameters):
            if field:
                value = latest.get(field, "Not recorded")
                column.metric(label, str(value))
            else:
                column.metric(label, f"{len(selected_sensor):,}")

        st.divider()

        st.subheader("Anomaly Assessment")

        a1, a2, a3 = st.columns(3)

        a1.metric("Anomalies Detected", f"{anomaly_count:,}")
        a2.metric("Observations Analysed", f"{anomaly_total:,}")
        a3.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")

        st.caption(
            "Anomaly flags indicate unusual sensor observations. "
            "They do not independently confirm a mechanical failure."
        )

        st.divider()

        st.subheader("Latest Recorded Sensor Readings")

        axis_metrics = [
            ("X1_ActualVelocity", "X Axis Velocity"),
            ("Y1_ActualVelocity", "Y Axis Velocity"),
            ("Z1_ActualVelocity", "Z Axis Velocity"),
            ("S1_ActualVelocity", "S Axis Velocity"),
            ("X1_ActualAcceleration", "X Axis Acceleration"),
            ("Y1_ActualAcceleration", "Y Axis Acceleration"),
            ("Z1_ActualAcceleration", "Z Axis Acceleration"),
            ("S1_ActualAcceleration", "S Axis Acceleration"),
            ("X1_CurrentFeedback", "X Axis Current"),
            ("Y1_CurrentFeedback", "Y Axis Current"),
            ("S1_CurrentFeedback", "S Axis Current"),
            ("X1_OutputPower", "X Axis Output Power"),
            ("Y1_OutputPower", "Y Axis Output Power"),
            ("S1_OutputPower", "S Axis Output Power")
        ]

        available_metrics = [
            (field, label)
            for field, label in axis_metrics
            if field in selected_sensor.columns
        ]

        if available_metrics:
            metric_columns = st.columns(3)

            for index, (field, label) in enumerate(available_metrics):
                value = latest.get(field)

                if pd.notna(value):
                    metric_columns[index % 3].metric(
                        label,
                        f"{float(value):.3f}"
                    )

        st.divider()

        st.subheader("Sensor Performance Trend")

        trend_options = [
            col for col in [
                "X1_ActualVelocity",
                "Y1_ActualVelocity",
                "Z1_ActualVelocity",
                "S1_ActualVelocity",
                "X1_ActualAcceleration",
                "Y1_ActualAcceleration",
                "X1_CurrentFeedback",
                "Y1_CurrentFeedback",
                "X1_OutputPower",
                "Y1_OutputPower"
            ]
            if col in selected_sensor.columns
        ]

        if trend_options:
            selected_metric = st.selectbox(
                "Select Sensor Parameter",
                trend_options
            )

            x_axis = (
                "M1_sequence_number"
                if "M1_sequence_number" in selected_sensor.columns
                else selected_sensor.index.name or "Observation"
            )

            chart_data = selected_sensor.reset_index()

            if x_axis not in chart_data.columns:
                chart_data["Observation"] = range(len(chart_data))
                x_axis = "Observation"

            fig = px.line(
                chart_data,
                x=x_axis,
                y=selected_metric,
                title=f"{selected_metric} - Experiment {selected_exp}"
            )

            fig.update_layout(
                template="plotly_white",
                xaxis_title="Recorded Observation",
                yaxis_title=selected_metric
            )

            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Performance Assessment")

        if worn_probability is None:
            st.info(
                "A model prediction is not available for this experiment."
            )
        elif worn_probability >= 70:
            st.error(
                "High predicted tool-wear probability. "
                "A tool-condition inspection is recommended."
            )
        elif worn_probability >= 30:
            st.warning(
                "Moderate predicted tool-wear probability. "
                "Continue monitoring and review the tool condition."
            )
        else:
            st.success(
                "Low predicted tool-wear probability in the available model output."
            )

        st.caption(
            "The assessment uses model probability as an exploratory indicator. "
            "It is not a validated safety threshold or a confirmed diagnosis."
        )

        with st.expander("View Latest Recorded Sensor Row"):
            st.dataframe(
                selected_sensor.tail(1),
                use_container_width=True
            )

    st.caption("NEUROSTORM | Machine Performance")


elif page == "Experiment History":

    st.title("Experiment History")
    st.caption(
        "Recorded experimental runs, operating conditions, "
        "tool condition and prediction results."
    )

    st.info(
        "The dataset identifies experimental runs using Experiment_ID. "
        "Machine A is a conceptual dashboard label, not a physical machine ID."
    )

    experiment_ids = sorted(
        sensor_data["Experiment_ID"].dropna().unique(),
        key=str
    )

    prediction_lookup = {
        str(row["Experiment_ID"]): row
        for _, row in prediction_data.iterrows()
    }

    anomaly_lookup = (
        anomaly_data.groupby("Experiment_ID")["anomaly"]
        .agg(["sum", "count"])
        .to_dict("index")
    )

    summary_rows = []

    for exp_id in experiment_ids:

        exp_sensor = sensor_data[
            sensor_data["Experiment_ID"].astype(str) == str(exp_id)
        ]

        if exp_sensor.empty:
            continue

        first_row = exp_sensor.iloc[0]

        def get_value(field):
            return (
                first_row[field]
                if field in exp_sensor.columns
                else "Not recorded"
            )

        anomaly_info = anomaly_lookup.get(exp_id, {})
        anomaly_count = int(anomaly_info.get("sum", 0))
        anomaly_total = int(anomaly_info.get("count", 0))

        anomaly_rate = (
            anomaly_count / anomaly_total * 100
            if anomaly_total else 0
        )

        pred = prediction_lookup.get(str(exp_id))

        if pred is not None:
            predicted_condition = (
                "Worn" if str(pred["Predicted"]).strip().lower() in ("1", "worn", "true") else "Unworn"
            )
            worn_probability = float(pred["Worn_Probability"]) * 100
            prediction_status = (
                "Correct"
                if (
                    ("worn" if str(pred["Actual"]).strip().lower() in ("1", "worn", "true") else "unworn")
                    ==
                    ("worn" if str(pred["Predicted"]).strip().lower() in ("1", "worn", "true") else "unworn")
                )
                else "Incorrect"
            )
        else:
            predicted_condition = "Not available"
            worn_probability = None
            prediction_status = "Not available"

        summary_rows.append({
            "Machine": "Machine A",
            "Experiment_ID": exp_id,
            "Material": get_value("material"),
            "Feedrate": get_value("feedrate"),
            "Clamp Pressure": get_value("clamp_pressure"),
            "Tool Condition": get_value("tool_condition"),
            "Machining Finalized": get_value("machining_finalized"),
            "Visual Inspection": get_value("passed_visual_inspection"),
            "Sensor Observations": len(exp_sensor),
            "Anomaly Count": anomaly_count,
            "Anomaly Rate (%)": round(anomaly_rate, 2),
            "Predicted Condition": predicted_condition,
            "Worn Probability (%)": (
                round(worn_probability, 2)
                if worn_probability is not None else None
            ),
            "Prediction Status": prediction_status
        })

    history_df = pd.DataFrame(summary_rows)

    st.subheader("All Recorded Experiments")

    st.metric("Total Experiments", len(history_df))

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Experiment History",
        data=csv_data,
        file_name="neurostorm_experiment_history.csv",
        mime="text/csv"
    )

    st.divider()

    st.subheader("Compare Two Experiments")

    comparison_columns = st.columns(2)

    first_exp = comparison_columns[0].selectbox(
        "First Experiment",
        experiment_ids,
        key="history_first_exp"
    )

    second_options = [
        exp for exp in experiment_ids
        if str(exp) != str(first_exp)
    ]

    second_exp = comparison_columns[1].selectbox(
        "Second Experiment",
        second_options,
        key="history_second_exp"
    )

    first_record = history_df[
        history_df["Experiment_ID"].astype(str) == str(first_exp)
    ].iloc[0]

    second_record = history_df[
        history_df["Experiment_ID"].astype(str) == str(second_exp)
    ].iloc[0]

    comparison_fields = [
        "Material",
        "Feedrate",
        "Clamp Pressure",
        "Tool Condition",
        "Machining Finalized",
        "Visual Inspection",
        "Anomaly Count",
        "Anomaly Rate (%)",
        "Predicted Condition",
        "Worn Probability (%)"
    ]

    comparison_rows = []

    for field in comparison_fields:
        first_value = first_record[field]
        second_value = second_record[field]

        if pd.isna(first_value) and pd.isna(second_value):
            changed = False
        elif pd.isna(first_value) or pd.isna(second_value):
            changed = True
        else:
            changed = str(first_value) != str(second_value)

        comparison_rows.append({
            "Parameter": field,
            f"Experiment {first_exp}": first_value,
            f"Experiment {second_exp}": second_value,
            "Difference": "Changed" if changed else "Same"
        })

    comparison_df = pd.DataFrame(comparison_rows)

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Machining Process History")

    timeline_exp = st.selectbox(
        "Select Experiment for Process History",
        experiment_ids,
        key="history_timeline_exp"
    )

    timeline_data = sensor_data[
        sensor_data["Experiment_ID"].astype(str) == str(timeline_exp)
    ].copy()

    if "Machining_Process" in timeline_data.columns:

        process_history = (
            timeline_data.groupby("Machining_Process", sort=False)
            .size()
            .reset_index(name="Observation Count")
            .rename(columns={"Machining_Process": "Process"})
        )

        if "M1_CURRENT_FEEDRATE" in timeline_data.columns:

            feedrate_history = (
                timeline_data.groupby("Machining_Process", sort=False)
                ["M1_CURRENT_FEEDRATE"]
                .mean()
                .reset_index(name="Average Feedrate")
                .rename(columns={"Machining_Process": "Process"})
            )

            process_history = process_history.merge(
                feedrate_history,
                on="Process",
                how="left"
            )

        st.dataframe(
            process_history,
            use_container_width=True,
            hide_index=True
        )

    st.warning(
        "Important: These are recorded differences between experimental runs. "
        "The dataset does not contain a timestamped operator change log, "
        "maintenance history or physical machine identifiers. "
        "Do not interpret these comparisons as proof of who changed a setting or when."
    )

    st.caption("NEUROSTORM | Experiment History")




# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "NEUROSTORM | Adaptive Multimodal Predictive Maintenance | "
    "Industrial Tool Wear Analytics"
)



