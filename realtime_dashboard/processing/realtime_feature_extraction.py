import numpy as np

from scipy.signal import (
    find_peaks,
    savgol_filter
)

# =====================================================
# REALTIME FEATURE EXTRACTION
# =====================================================

def extract_features(

    processed_window,

    fs=100
):

    try:

        ecg = processed_window[
            "ecg_filtered"
        ]

        ppg = processed_window[
            "ppg_filtered"
        ]

        accel = processed_window[
            "accel_magnitude"
        ]

        gyro = processed_window[
            "gyro_magnitude"
        ]

        temp = processed_window[
            "temperature"
        ]

        # =============================================
        # SIGNAL QUALITY
        # =============================================

        ecg_std = np.std(ecg)

        ppg_std = np.std(ppg)

        print(
            "\nECG STD:",
            ecg_std
        )

        print(
            "PPG STD:",
            ppg_std
        )

        # =============================================
        # SMOOTHING
        # =============================================

        smooth_window = max(
            5,
            int(fs * 0.12)
        )

        if smooth_window % 2 == 0:
            smooth_window += 1

        ecg = savgol_filter(
            ecg,
            smooth_window,
            3
        )

        ppg = savgol_filter(
            ppg,
            smooth_window,
            3
        )

        # =============================================
        # ECG PEAKS
        # =============================================

        ecg_peaks, _ = find_peaks(

            ecg,

            distance=max(
                1,
                int(fs * 0.4)
            ),

            prominence=np.std(
                ecg
            ) * 0.4
        )

        print(
            "ecg peaks:",
            len(ecg_peaks)
        )

        if len(ecg_peaks) < 2:

            return None

        # =============================================
        # RR INTERVALS
        # =============================================

        rr_ms = (

            np.diff(ecg_peaks) / fs

        ) * 1000

        rr_ms = rr_ms[
            (
                rr_ms > 400
            ) &
            (
                rr_ms < 1500
            )
        ]

        if len(rr_ms) < 1:

            return None

        # =============================================
        # HEART RATE
        # =============================================

        heart_rate = (
            60000 /
            np.mean(rr_ms)
        )

        # =============================================
        # HRV
        # =============================================

        sdnn = np.std(
            rr_ms
        )

        if len(rr_ms) > 1:

            rmssd = np.sqrt(

                np.mean(
                    np.diff(rr_ms) ** 2
                )
            )

        else:

            rmssd = 0

        # =============================================
        # PPG FEATURES
        # =============================================

        ppg_amp = (
            np.max(ppg) -
            np.min(ppg)
        )

        pulse_width_ms = (
            np.mean(rr_ms) * 0.35
        )

        # =============================================
        # TEMPORARY VALUES
        # =============================================

        pat_mean = 180.0

        pat_std = 20.0

        spo2 = 98.0

        # =============================================
        # FEATURES
        # =============================================

        features = {

            "pat_mean_ms":
                float(pat_mean),

            "pat_std_ms":
                float(pat_std),

            "heart_rate_bpm":
                float(heart_rate),

            "spo2":
                float(spo2),

            "sdnn":
                float(sdnn),

            "rmssd":
                float(rmssd),

            "ppg_amplitude":
                float(ppg_amp),

            "pulse_width_ms":
                float(
                    pulse_width_ms
                ),

            "accel_motion":
                float(
                    np.mean(accel)
                ),

            "gyro_motion":
                float(
                    np.mean(gyro)
                ),

            "temperature":
                float(
                    np.mean(temp)
                )
        }

        print(
            "\nFEATURES:"
        )

        print(features)

        return features

    except Exception as e:

        print(
            "feature extraction error:",
            e
        )

        return None