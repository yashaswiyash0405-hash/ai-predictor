
"""
INDUSTRIAL SENSOR INTELLIGENCE PLATFORM
=======================================
Upload-based sensor analytics, machine learning,
anomaly detection and future trend forecasting.

Run:
    streamlit run sensor_upload_ai.py

Install:
    python -m pip install streamlit pandas numpy scikit-learn openpyxl
"""

import io
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Industrial Sensor Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #0B1120;
    color: #E5E7EB;
}

[data-testid="stHeader"] {
    background: rgba(11,17,32,0.95);
}

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #263247;
}

[data-testid="stSidebar"] * {
    color: #D1D5DB;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: #F9FAFB !important;
    font-weight: 700 !important;
}

p, label, span {
    color: #CBD5E1;
}

.main-header {
    padding: 24px 28px;
    background: linear-gradient(120deg, #17243B, #172C43);
    border: 1px solid #30445F;
    border-radius: 14px;
    margin-bottom: 25px;
}

.main-header h1 {
    font-size: 30px;
    margin: 0;
    letter-spacing: -0.8px;
}

.main-header p {
    color: #9CAFC5;
    margin: 8px 0 0 0;
    font-size: 14px;
}

.section-label {
    color: #60A5FA;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.metric-card {
    background: #151F30;
    border: 1px solid #29394F;
    border-radius: 12px;
    padding: 19px;
    min-height: 125px;
}

.metric-label {
    font-size: 12px;
    color: #94A3B8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.metric-value {
    font-family: 'Manrope', sans-serif;
    font-size: 27px;
    font-weight: 800;
    color: #F8FAFC;
    margin-top: 12px;
}

.metric-description {
    font-size: 11px;
    color: #94A3B8;
    margin-top: 5px;
}

.status-card {
    background: #151F30;
    border: 1px solid #29394F;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 15px;
}

.status-normal {
    border-left: 4px solid #10B981;
}

.status-warning {
    border-left: 4px solid #F59E0B;
}

.status-danger {
    border-left: 4px solid #EF4444;
}

div[data-testid="stTabs"] button {
    color: #94A3B8;
    font-weight: 600;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #60A5FA;
    border-bottom-color: #3B82F6;
}

.stButton > button,
.stDownloadButton > button {
    background: #2563EB;
    color: white;
    border: none;
    border-radius: 7px;
    font-weight: 600;
    padding: 9px 18px;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #1D4ED8;
    color: white;
}

[data-testid="stDataFrame"] {
    border: 1px solid #29394F;
    border-radius: 10px;
}

[data-testid="stMetric"] {
    background: #151F30;
    border: 1px solid #29394F;
    padding: 15px;
    border-radius: 10px;
}

[data-testid="stMetricLabel"] {
    color: #94A3B8;
}

[data-testid="stMetricValue"] {
    color: #F8FAFC;
}

hr {
    border-color: #29394F;
}

.footer {
    border-top: 1px solid #29394F;
    padding-top: 18px;
    margin-top: 35px;
    color: #64748B;
    font-size: 12px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def metric_card(label, value, description):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-description">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def guess_column(columns, keywords):
    for column in columns:
        name = column.lower().replace("_", "").replace(" ", "")
        if any(word in name for word in keywords):
            return column
    return None


def make_future_index(frame, count, timestamp_column):
    if timestamp_column == "(Row order)":
        return np.arange(len(frame), len(frame) + count)

    parsed = pd.to_datetime(
        frame[timestamp_column],
        errors="coerce"
    ).dropna()

    if len(parsed) < 2:
        return np.arange(len(frame), len(frame) + count)

    step = parsed.diff().dropna().median()

    if pd.isna(step) or step <= pd.Timedelta(0):
        return np.arange(len(frame), len(frame) + count)

    return pd.date_range(
        start=parsed.iloc[-1] + step,
        periods=count,
        freq=step
    )


def forecast_sensor(values, steps):
    values = np.asarray(values, dtype=float)

    window = min(len(values), 100)
    recent = values[-window:]

    x = np.arange(window, dtype=float)

    slope, intercept = np.polyfit(x, recent, 1)

    future_x = np.arange(
        window,
        window + steps,
        dtype=float
    )

    prediction = slope * future_x + intercept

    return prediction


def create_excel_report(results, forecast, model_info):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:

        results.to_excel(
            writer,
            sheet_name="Sensor_Analysis",
            index=False
        )

        forecast.to_excel(
            writer,
            sheet_name="Future_Forecast",
            index=False
        )

        model_info.to_excel(
            writer,
            sheet_name="Model_Summary",
            index=False
        )

    return output.getvalue()


def show_status(classification):
    if classification == "Anomaly":
        return "status-danger"

    if classification == "Normal pattern":
        return "status-normal"

    return "status-warning"


# ============================================================
# APPLICATION HEADER
# ============================================================

st.markdown("""
<div class="main-header">
    <div class="section-label">INDUSTRIAL ANALYTICS / AI SYSTEM</div>
    <h1>Sensor Intelligence Platform</h1>
    <p>
        Intelligent sensor monitoring, machine learning classification,
        anomaly detection and predictive trend analysis.
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## SENSOR ANALYTICS")

    st.caption("INDUSTRIAL MONITORING SYSTEM")

    st.divider()

    st.markdown("### Data source")

    uploaded = st.file_uploader(
        "Import sensor dataset",
        type=["csv", "xlsx", "xls"],
        help="Upload historical sensor readings."
    )

    st.divider()

    st.markdown("### Platform information")

    st.markdown("""
    **Analysis modules**

    - Sensor data processing
    - AI classification
    - Anomaly detection
    - Trend forecasting
    - Report generation
    """)

    st.divider()

    st.caption("Version 2.0.0")
    st.caption("Industrial Sensor Intelligence")


# ============================================================
# FILE UPLOAD
# ============================================================

if uploaded is None:

    st.markdown("## Welcome to Sensor Intelligence")

    st.markdown("""
    Upload your historical sensor dataset to begin intelligent
    analysis and predictive monitoring.
    """)

    st.markdown("### Supported data structure")

    st.code("""
Timestamp | Temperature | Gas | Light
--------------------------------------
10:00     | 28.5        | 120 | 450
10:05     | 29.2        | 135 | 430
10:10     | 30.1        | 180 | 410
10:15     | 31.5        | 350 | 390
    """)

    st.info(
        "Supported formats: CSV, XLSX and XLS. "
        "A minimum of 10 usable sensor observations is required."
    )

    st.stop()


# ============================================================
# READ DATASET
# ============================================================

try:

    file_bytes = uploaded.getvalue()

    if uploaded.name.lower().endswith(".csv"):

        df = pd.read_csv(io.BytesIO(file_bytes))

    else:

        excel = pd.ExcelFile(io.BytesIO(file_bytes))

        selected_sheet = st.sidebar.selectbox(
            "Select worksheet",
            excel.sheet_names
        )

        df = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name=selected_sheet
        )

except Exception as exc:

    st.error(f"Dataset reading error: {exc}")
    st.stop()


if df.empty:

    st.error("The uploaded dataset contains no records.")
    st.stop()


df.columns = [str(c).strip() for c in df.columns]

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

if not numeric_columns:

    st.error("No numeric sensor columns were identified.")
    st.stop()


# ============================================================
# SENSOR CONFIGURATION
# ============================================================

st.markdown('<div class="section-label">CONFIGURATION</div>',
            unsafe_allow_html=True)

st.header("Sensor Configuration")

st.caption(
    "Identify the sensor columns and configure the analysis parameters."
)

with st.expander("Configure dataset and sensor inputs", expanded=True):

    c1, c2, c3 = st.columns(3)

    with c1:

        temp_guess = guess_column(
            numeric_columns,
            ["temperature", "temp"]
        )

        temp_col = st.selectbox(
            "Temperature",
            ["(None)"] + numeric_columns,
            index=numeric_columns.index(temp_guess) + 1
            if temp_guess else 0
        )

    with c2:

        gas_guess = guess_column(
            numeric_columns,
            ["gas", "mq", "smoke"]
        )

        gas_col = st.selectbox(
            "Gas / Smoke",
            ["(None)"] + numeric_columns,
            index=numeric_columns.index(gas_guess) + 1
            if gas_guess else 0
        )

    with c3:

        light_guess = guess_column(
            numeric_columns,
            ["light", "ldr", "lux"]
        )

        light_col = st.selectbox(
            "Light intensity",
            ["(None)"] + numeric_columns,
            index=numeric_columns.index(light_guess) + 1
            if light_guess else 0
        )

    selected_sensors = [
        c for c in [temp_col, gas_col, light_col]
        if c != "(None)"
    ]

    other_numeric = [
        c for c in numeric_columns
        if c not in selected_sensors
    ]

    extra_cols = st.multiselect(
        "Additional sensor columns",
        other_numeric
    )

    sensor_cols = list(dict.fromkeys(
        selected_sensors + extra_cols
    ))

    time_options = ["(Row order)"] + [
        c for c in df.columns if c not in sensor_cols
    ]

    time_guess = next(
        (
            c for c in df.columns
            if any(k in c.lower() for k in ["time", "date"])
        ),
        "(Row order)"
    )

    if time_guess not in time_options:
        time_guess = "(Row order)"

    time_col = st.selectbox(
        "Timestamp",
        time_options,
        index=time_options.index(time_guess)
    )

    label_options = [
        "(No label — automatic detection)"
    ] + [
        c for c in df.columns if c not in sensor_cols
    ]

    label_col = st.selectbox(
        "Existing operating condition label",
        label_options
    )

    forecast_steps = st.slider(
        "Forecast horizon (number of future observations)",
        min_value=5,
        max_value=100,
        value=30,
        step=5
    )


if not sensor_cols:

    st.warning("Select at least one numeric sensor column.")
    st.stop()


# ============================================================
# DATA PREPROCESSING
# ============================================================

work = df.copy()

for col in sensor_cols:

    work[col] = pd.to_numeric(
        work[col],
        errors="coerce"
    )

work = work.replace([np.inf, -np.inf], np.nan)

work = work.dropna(
    subset=sensor_cols,
    how="all"
).reset_index(drop=True)

if len(work) < 10:

    st.error(
        "At least 10 usable sensor observations are required."
    )
    st.stop()

X = work[sensor_cols].copy()

imputer = SimpleImputer(strategy="median")

X_filled = pd.DataFrame(
    imputer.fit_transform(X),
    columns=sensor_cols,
    index=work.index
)

# ============================================================
# DATASET OVERVIEW
# ============================================================

st.markdown('<div class="section-label">DATA OVERVIEW</div>',
            unsafe_allow_html=True)

st.header("Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    metric_card(
        "Total observations",
        f"{len(work):,}",
        "Usable sensor records"
    )

with c2:
    metric_card(
        "Sensor variables",
        str(len(sensor_cols)),
        "Selected numeric inputs"
    )

with c3:
    metric_card(
        "Missing readings",
        f"{int(X.isna().sum().sum()):,}",
        "Missing sensor values"
    )

with c4:
    metric_card(
        "File format",
        uploaded.name.split(".")[-1].upper(),
        "Uploaded data source"
    )

st.write("")

with st.expander("View uploaded dataset", expanded=False):

    st.dataframe(
        work.head(100),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MACHINE LEARNING
# ============================================================

st.markdown('<div class="section-label">ARTIFICIAL INTELLIGENCE</div>',
            unsafe_allow_html=True)

st.header("AI Classification & Anomaly Detection")

accuracy = None
classification_text = None
confusion = None
safety_values = None

if label_col != "(No label — automatic detection)":

    labels = work[label_col].astype("string").fillna("Unknown")

    class_counts = labels.value_counts()

    if len(class_counts) < 2 or class_counts.min() < 2:

        st.error(
            "Classification requires at least two classes "
            "with at least two observations per class."
        )

        st.stop()

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ])

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        predictions = model.predict(X)

        probabilities = model.predict_proba(X)

        confidence = probabilities.max(axis=1) * 100

        predicted_labels = predictions.astype(str)

        confidence_values = confidence

        method = "Random Forest Classifier"

        safety_values = None

        classification_text = classification_report(
            y_test,
            y_pred,
            zero_division=0
        )

        confusion = confusion_matrix(
            y_test,
            y_pred,
            labels=model.classes_
        )

        st.success("Supervised AI model trained successfully.")

    except Exception as exc:

        st.error(f"Classification failed: {exc}")
        st.stop()

else:

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_filled)

    raw_predictions = model.predict(X_filled)

    decision = model.decision_function(X_filled)

    low, high = np.percentile(
        decision,
        [5, 95]
    )

    if high <= low:

        relative_score = np.where(
            raw_predictions == 1,
            75.0,
            20.0
        )

    else:

        relative_score = np.clip(
            (decision - low) / (high - low) * 100,
            0,
            100
        )

    predicted_labels = np.where(
        raw_predictions == 1,
        "Normal pattern",
        "Anomaly"
    )

    confidence_values = relative_score

    safety_values = relative_score

    method = "Isolation Forest"

    normal_count = int(
        np.sum(raw_predictions == 1)
    )

    anomaly_count = int(
        np.sum(raw_predictions == -1)
    )

    st.success("Anomaly detection completed successfully.")


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

if accuracy is not None:

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "Test accuracy",
            f"{accuracy * 100:.2f}%",
            "Held-out test observations"
        )

    with c2:
        metric_card(
            "Operating classes",
            str(labels.nunique()),
            "Unique labelled conditions"
        )

    with c3:
        metric_card(
            "Training records",
            f"{len(X_train):,}",
            "Records used for training"
        )

    st.caption(
        "Confidence represents the highest predicted class probability. "
        "It is not a physical safety probability."
    )

    with st.expander("Detailed classification report"):

        st.text(classification_text)

    with st.expander("Confusion matrix"):

        confusion_df = pd.DataFrame(
            confusion,
            index=model.classes_,
            columns=model.classes_
        )

        st.dataframe(
            confusion_df,
            use_container_width=True
        )

else:

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "Normal patterns",
            f"{normal_count:,}",
            "Records identified as typical"
        )

    with c2:
        metric_card(
            "Anomalies",
            f"{anomaly_count:,}",
            "Unusual sensor patterns"
        )

    with c3:
        metric_card(
            "Anomaly percentage",
            f"{anomaly_count / len(work) * 100:.2f}%",
            "Share of observations flagged"
        )

    st.caption(
        "Isolation Forest identifies unusual patterns relative to the "
        "uploaded dataset. It does not establish actual safe or dangerous "
        "operating conditions."
    )


# ============================================================
# ANALYSIS RESULTS
# ============================================================

results = work.copy()

results["AI Classification"] = predicted_labels

results["Model Score / Confidence (%)"] = np.round(
    confidence_values,
    2
)

if safety_values is not None:

    results["Relative Pattern Indicator (%)"] = np.round(
        safety_values,
        2
    )


# ============================================================
# DASHBOARD TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "OVERVIEW",
    "SENSOR ANALYTICS",
    "PREDICTIVE FORECAST",
    "REPORTS & EXPORT"
])


# ============================================================
# TAB 1: OVERVIEW
# ============================================================

with tab1:

    st.subheader("Operational Intelligence Overview")

    latest = results.iloc[-1]

    c1, c2, c3 = st.columns(3)

    with c1:

        metric_card(
            "Latest classification",
            str(latest["AI Classification"]),
            "Most recent uploaded observation"
        )

    with c2:

        metric_card(
            "Model score",
            f"{latest['Model Score / Confidence (%)']:.1f}%",
            "Model confidence or relative anomaly score"
        )

    with c3:

        metric_card(
            "Active model",
            method,
            "Selected machine learning algorithm"
        )

    st.write("")

    st.markdown("### Latest sensor readings")

    reading_cols = st.columns(min(len(sensor_cols), 4))

    for i, col in enumerate(sensor_cols):

        with reading_cols[i % len(reading_cols)]:

            metric_card(
                col,
                f"{latest[col]:.2f}",
                "Latest available reading"
            )

    st.write("")

    st.markdown("### Recent classification results")

    st.dataframe(
        results.tail(20),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Operating condition distribution")

    distribution = (
        results["AI Classification"]
        .value_counts()
        .rename_axis("Condition")
        .to_frame("Observations")
    )

    st.bar_chart(
        distribution,
        horizontal=True
    )


# ============================================================
# TAB 2: SENSOR ANALYTICS
# ============================================================

with tab2:

    st.subheader("Historical Sensor Analytics")

    selected_chart = st.selectbox(
        "Select sensor variable",
        sensor_cols
    )

    chart_data = pd.DataFrame({
        "Sensor reading": X_filled[selected_chart]
    })

    st.markdown(f"#### {selected_chart} — Historical Trend")

    st.line_chart(
        chart_data,
        height=350
    )

    st.markdown("### Statistical summary")

    stats = X_filled[sensor_cols].describe().T

    stats.columns = [
        "Count",
        "Mean",
        "Std",
        "Minimum",
        "25%",
        "Median",
        "75%",
        "Maximum"
    ]

    st.dataframe(
        stats.round(3),
        use_container_width=True
    )

    st.markdown("### Sensor correlation analysis")

    if len(sensor_cols) >= 2:

        correlation = X_filled.corr()

        st.dataframe(
            correlation.round(3),
            use_container_width=True
        )

    else:

        st.info(
            "Select at least two sensor variables "
            "to calculate correlation."
        )


# ============================================================
# TAB 3: FUTURE PREDICTION
# ============================================================

with tab3:

    st.subheader("Sensor Trend Forecasting")

    st.caption(
        "Linear trend extrapolation based on the most recent "
        "100 observations."
    )

    future_index = make_future_index(
        work,
        forecast_steps,
        time_col
    )

    future_data = pd.DataFrame({
        "Future step / time": future_index
    })

    for col in sensor_cols:

        values = X_filled[col].to_numpy(dtype=float)

        forecast = forecast_sensor(
            values,
            forecast_steps
        )

        future_data[f"Predicted {col}"] = np.round(
            forecast,
            3
        )

    st.markdown("### Forecast summary")

    forecast_metrics = st.columns(
        min(len(sensor_cols), 4)
    )

    for i, col in enumerate(sensor_cols):

        with forecast_metrics[i % len(forecast_metrics)]:

            metric_card(
                col,
                f"{future_data[f'Predicted {col}'].iloc[-1]:.2f}",
                f"Predicted value at step {forecast_steps}"
            )

    st.write("")

    st.markdown("### Future sensor predictions")

    st.dataframe(
        future_data,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Historical vs predicted trends")

    for col in sensor_cols:

        st.markdown(f"#### {col}")

        historical = X_filled[col].tail(100).reset_index(
            drop=True
        )

        predicted = future_data[
            f"Predicted {col}"
        ].reset_index(drop=True)

        chart = pd.DataFrame({
            "Historical": pd.Series(
                historical.to_numpy(),
                index=range(len(historical))
            ),
            "Predicted": pd.Series(
                predicted.to_numpy(),
                index=range(len(historical), len(historical) + forecast_steps)
            )
        })

        st.line_chart(
            chart,
            height=260
        )

    st.warning(
        "Forecasts are statistical extrapolations, not guaranteed future "
        "readings. They do not account for changing operating conditions, "
        "machine failures or external disturbances."
    )


# ============================================================
# TAB 4: REPORTS AND EXPORT
# ============================================================

with tab4:

    st.subheader("Analysis Reports & Data Export")

    st.markdown("### Classified sensor dataset")

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown("### Export options")

    c1, c2 = st.columns(2)

    with c1:

        csv_results = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download classified sensor data",
            data=csv_results,
            file_name="sensor_analysis_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    with c2:

        csv_forecast = future_data.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download future predictions",
            data=csv_forecast,
            file_name="sensor_forecast.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.divider()

    st.markdown("### Complete Excel report")

    model_info = pd.DataFrame({
        "Parameter": [
            "AI Algorithm",
            "Sensor Variables",
            "Total Observations",
            "Classification Accuracy",
            "Forecast Horizon",
            "Analysis Date"
        ],
        "Value": [
            method,
            ", ".join(sensor_cols),
            len(work),
            f"{accuracy * 100:.2f}%" if accuracy is not None else "Not applicable",
            forecast_steps,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]
    })

    excel_report = create_excel_report(
        results,
        future_data,
        model_info
    )

    st.download_button(
        label="Download complete analysis report (Excel)",
        data=excel_report,
        file_name="industrial_sensor_intelligence_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    INDUSTRIAL SENSOR INTELLIGENCE PLATFORM
    <br>
    Machine Learning | Anomaly Detection | Predictive Analytics
    <br><br>
    Analytical results are intended for decision support and
    are not a certified industrial safety system.
</div>
""", unsafe_allow_html=True)
