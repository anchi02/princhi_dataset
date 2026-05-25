import streamlit as st
import pandas as pd
import numpy as np
import time
from collections import deque

# =====================================================
# IMPORTS
# =====================================================

from stream.esp_serial_reader import (
    connect_serial,
    read_sensor_packet
)

from processing.signal_buffer import SignalBuffer
from processing.filtering import process_window
from processing.realtime_feature_extraction import extract_features
from processing.realtime_predictor import predict_bp

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Realtime BP Dashboard",
    layout="wide"
)

st.title("Realtime Biomedical Dashboard")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Settings")

model_choice = st.sidebar.selectbox(
    "Prediction Model",
    ["xgboost", "random_forest", "gradient_boosting"]
)

serial_port = st.sidebar.text_input(
    "Serial Port",
    "COM5"
)

start_button = st.sidebar.button("Start Monitoring")

# =====================================================
# METRICS
# =====================================================

col1, col2, col3 = st.columns(3)
col4, col5 = st.columns(2)

hr_metric = col1.empty()
spo2_metric = col2.empty()
temp_metric = col3.empty()
bp_metric = col4.empty()
status_metric = col5.empty()

# =====================================================
# CHARTS
# =====================================================

st.subheader("ECG Waveform")
ecg_chart_placeholder = st.empty()

st.subheader("PPG Waveform")
ppg_chart_placeholder = st.empty()

motion_col1, motion_col2 = st.columns(2)

motion_col1.subheader("Accelerometer")
motion_col2.subheader("Gyroscope")

accel_chart_placeholder = motion_col1.empty()
gyro_chart_placeholder = motion_col2.empty()

st.subheader("Extracted Features")
feature_placeholder = st.empty()

# =====================================================
# START STREAMING
# =====================================================

if start_button:

    # SERIAL CONNECTION
    ser = connect_serial(
        port=serial_port,
        baud_rate=115200
    )

    # BUFFER
    signal_buffer = SignalBuffer(
        window_seconds=8,
        sampling_rate=250
    )

    # CHART BUFFERS
    ecg_chart = deque(maxlen=1000)
    ppg_chart = deque(maxlen=1000)
    accel_chart = deque(maxlen=1000)
    gyro_chart = deque(maxlen=1000)

    # =================================================
    # STREAM LOOP (ONLY BUFFER FIX APPLIED)
    # =================================================

    while True:

        try:

            time.sleep(0.03)

            # READ SENSOR
            sample = read_sensor_packet(ser)

            if sample is None:
                continue

            # =================================================
            # FIXED BUFFER BLOCK (ONLY CHANGE)
            # =================================================

            try:
                clean_sample = {
                    "ecg": float(sample.get("ecg", 0)),
                    "ir": float(sample.get("ir", 0)),
                    "accX": float(sample.get("accX", 0)),
                    "accY": float(sample.get("accY", 0)),
                    "accZ": float(sample.get("accZ", 0)),
                    "gyroX": float(sample.get("gyroX", 0)),
                    "gyroY": float(sample.get("gyroY", 0)),
                    "gyroZ": float(sample.get("gyroZ", 0)),
                }

                signal_buffer.add_sample(clean_sample)

            except:
                continue

            # =================================================
            # CHART DATA
            # =================================================

            ecg_chart.append(sample["ecg"])
            ppg_chart.append(sample["ir"])

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

            accel_chart.append(accel_mag)
            gyro_chart.append(gyro_mag)

            # =================================================
            # WINDOW CHECK
            # =================================================

            if not signal_buffer.is_ready():
                continue

            window = signal_buffer.get_window()

            processed = process_window(window)
            if processed is None:
                continue

            features = extract_features(processed)
            if features is None:
                continue

            prediction = predict_bp(
                features,
                model_type=model_choice
            )

            if prediction is None:
                continue

            # =================================================
            # UI UPDATES (UNCHANGED)
            # =================================================

            hr_metric.metric(
                "Heart Rate",
                f"{features['heart_rate_bpm']:.1f} BPM"
            )

            spo2_metric.metric("SpO2", "98%")

            temp_metric.metric(
                "Temperature",
                f"{features['temperature']:.1f} °C"
            )

            bp_metric.metric(
                "Blood Pressure",
                f"{prediction['sbp']} / {prediction['dbp']} mmHg"
            )

            status_metric.metric(
                "Status",
                prediction["status"]
            )

            # =================================================
            # PLOTS
            # =================================================

            ecg_chart_placeholder.line_chart(
                pd.DataFrame({"ECG": list(ecg_chart)})
            )

            ppg_chart_placeholder.line_chart(
                pd.DataFrame({"PPG": list(ppg_chart)})
            )

            accel_chart_placeholder.line_chart(
                pd.DataFrame({"Accel": list(accel_chart)})
            )

            gyro_chart_placeholder.line_chart(
                pd.DataFrame({"Gyro": list(gyro_chart)})
            )

            feature_placeholder.dataframe(
                pd.DataFrame([features])
            )

        except Exception as e:
            st.error(f"runtime error: {e}")
            time.sleep(0.5)