import streamlit as st

import pandas as pd
import numpy as np

import time
import traceback

from collections import deque

# =====================================================
# IMPORTS
# =====================================================

from stream.esp_serial_reader import (

    connect_serial,

    read_sensor_packet
)

from processing.signal_buffer import (
    SignalBuffer
)

from processing.filtering import (
    process_window
)

from processing.realtime_feature_extraction import (
    extract_features
)

from processing.realtime_predictor import (
    predict_bp
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(

    page_title="Realtime Biomedical Dashboard",

    layout="wide"
)

# =====================================================
# TITLE
# =====================================================

st.title(
    "Realtime Biomedical Dashboard"
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Settings")

model_choice = st.sidebar.selectbox(

    "Prediction Model",

    [

        "xgboost",

        "random_forest",

        "gradient_boosting"
    ]
)

serial_port = st.sidebar.text_input(

    "Serial Port",

    "COM7"
)

start_button = st.sidebar.button(
    "Start Monitoring"
)

# =====================================================
# METRICS
# =====================================================

metric_col1, metric_col2, metric_col3 = (
    st.columns(3)
)

metric_col4, metric_col5 = (
    st.columns(2)
)

hr_metric = metric_col1.empty()

spo2_metric = metric_col2.empty()

temp_metric = metric_col3.empty()

bp_metric = metric_col4.empty()

status_metric = metric_col5.empty()

# =====================================================
# ECG
# =====================================================

st.subheader("ECG Waveform")

ecg_chart_placeholder = st.empty()

# =====================================================
# PPG
# =====================================================

st.subheader("PPG Waveform")

ppg_chart_placeholder = st.empty()

# =====================================================
# MOTION
# =====================================================

motion_col1, motion_col2 = (
    st.columns(2)
)

motion_col1.subheader(
    "Accelerometer"
)

motion_col2.subheader(
    "Gyroscope"
)

accel_chart_placeholder = (
    motion_col1.empty()
)

gyro_chart_placeholder = (
    motion_col2.empty()
)

# =====================================================
# FEATURES
# =====================================================

st.subheader("Extracted Features")

feature_placeholder = st.empty()

# =====================================================
# START
# =====================================================

if start_button:

    try:

        # =============================================
        # SERIAL
        # =============================================

        ser = connect_serial(

            port=serial_port,

            baud_rate=115200
        )

        # =============================================
        # BUFFER
        # =============================================

        signal_buffer = SignalBuffer(

            window_seconds=8,

            sampling_rate=100
        )

        # =============================================
        # CHART BUFFERS
        # =============================================

        ecg_chart = deque(
            maxlen=1000
        )

        ppg_chart = deque(
            maxlen=1000
        )

        accel_chart = deque(
            maxlen=1000
        )

        gyro_chart = deque(
            maxlen=1000
        )

        # =============================================
        # LOOP
        # =============================================

        while True:

            try:

                # =====================================
                # READ SAMPLE
                # =====================================

                sample = read_sensor_packet(
                    ser
                )

                print("\n====================")
                print("RAW SAMPLE")
                print("====================")

                print(sample)

                if sample is None:

                    time.sleep(0.001)

                    continue

                # =====================================
                # BUFFER
                # =====================================

                signal_buffer.add_sample(
                    sample
                )

                # =====================================
                # CHART DATA
                # =====================================

                ecg_chart.append(
                    sample["ecg"]
                )

                ppg_chart.append(
                    sample["ir"]
                )

                accel_mag = np.sqrt(

                    sample["accX"]**2 +

                    sample["accY"]**2 +

                    sample["accZ"]**2
                )

                gyro_mag = np.sqrt(

                    sample["gyroX"]**2 +

                    sample["gyroY"]**2 +

                    sample["gyroZ"]**2
                )

                accel_chart.append(
                    accel_mag
                )

                gyro_chart.append(
                    gyro_mag
                )

                # =====================================
                # ALWAYS SHOW CHARTS
                # =====================================

                ecg_df = pd.DataFrame({

                    "ECG":
                        list(ecg_chart)
                })

                ecg_chart_placeholder.line_chart(
                    ecg_df
                )

                ppg_df = pd.DataFrame({

                    "PPG":
                        list(ppg_chart)
                })

                ppg_chart_placeholder.line_chart(
                    ppg_df
                )

                accel_df = pd.DataFrame({

                    "Accel":
                        list(accel_chart)
                })

                accel_chart_placeholder.line_chart(
                    accel_df
                )

                gyro_df = pd.DataFrame({

                    "Gyro":
                        list(gyro_chart)
                })

                gyro_chart_placeholder.line_chart(
                    gyro_df
                )

                # =====================================
                # RAW METRICS
                # =====================================

                temp_metric.metric(

                    "Temperature",

                    f"{sample['temp']:.1f} °C"
                )

                # =====================================
                # BUFFER STATUS
                # =====================================

                if not signal_buffer.is_ready():

                    hr_metric.metric(

                        "Heart Rate",

                        "-- BPM"
                    )

                    spo2_metric.metric(

                        "SpO2",

                        "-- %"
                    )

                    bp_metric.metric(

                        "Blood Pressure",

                        "-- / --"
                    )

                    status_metric.metric(

                        "Status",

                        "Buffer Filling"
                    )

                    time.sleep(0.01)

                    continue

                # =====================================
                # PROCESS WINDOW
                # =====================================

                window = signal_buffer.get_window()

                processed = process_window(
                    window
                )

                if processed is None:

                    status_metric.metric(

                        "Status",

                        "Processing Failed"
                    )

                    time.sleep(0.01)

                    continue

                # =====================================
                # FEATURES
                # =====================================

                features = extract_features(
                    processed
                )

                print("\nFEATURES:")

                print(features)

                # =====================================
                # IF FEATURES FAIL
                # =====================================

                if features is None:

                    hr_metric.metric(

                        "Heart Rate",

                        "-- BPM"
                    )

                    spo2_metric.metric(

                        "SpO2",

                        "-- %"
                    )

                    bp_metric.metric(

                        "Blood Pressure",

                        "-- / --"
                    )

                    status_metric.metric(

                        "Status",

                        "Signal Invalid"
                    )

                    time.sleep(0.01)

                    continue

                # =====================================
                # FEATURE TABLE
                # =====================================

                feature_df = pd.DataFrame(
                    [features]
                )

                feature_placeholder.dataframe(
                    feature_df
                )

                # =====================================
                # METRICS
                # =====================================

                hr_metric.metric(

                    "Heart Rate",

                    f"{features['heart_rate_bpm']:.1f} BPM"
                )

                spo2_metric.metric(

                    "SpO2",

                    f"{features['spo2']:.1f}%"
                )

                temp_metric.metric(

                    "Temperature",

                    f"{features['temperature']:.1f} °C"
                )

                # =====================================
                # PREDICT BP
                # =====================================

                prediction = predict_bp(

                    features,

                    model_type=model_choice
                )

                print("\nPREDICTION:")

                print(prediction)

                if prediction is None:

                    bp_metric.metric(

                        "Blood Pressure",

                        "-- / --"
                    )

                    status_metric.metric(

                        "Status",

                        "Prediction Failed"
                    )

                    time.sleep(0.01)

                    continue

                # =====================================
                # BP
                # =====================================

                bp_metric.metric(

                    "Blood Pressure",

                    f"{prediction['sbp']} / "
                    f"{prediction['dbp']} mmHg"
                )

                # =====================================
                # STATUS
                # =====================================

                status_metric.metric(

                    "Status",

                    prediction["status"]
                )

                # =====================================
                # LOOP SLEEP
                # =====================================

                time.sleep(0.01)

            except Exception:

                print("\n================")

                traceback.print_exc()

                print("================\n")

                status_metric.metric(

                    "Status",

                    "Runtime Error"
                )

                time.sleep(0.05)

    except Exception:

        traceback.print_exc()

        st.error(
            "failed to start monitoring"
        )