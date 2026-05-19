import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

# ----------------------------------------
# load dataset
# ----------------------------------------

df = pd.read_csv("data/data_collect1.csv")

# ----------------------------------------
# sort by timestamp
# ----------------------------------------

df = df.sort_values("timestamp_us")

# ----------------------------------------
# remove duplicate timestamps
# ----------------------------------------

df = df.drop_duplicates(subset="timestamp_us")

# ----------------------------------------
# create time in seconds
# ----------------------------------------

df["time_sec"] = (
    df["timestamp_us"] - df["timestamp_us"].iloc[0]
) / 1_000_000

# ----------------------------------------
# estimate sampling frequency
# ----------------------------------------

sampling_intervals = np.diff(df["time_sec"])

sampling_intervals = sampling_intervals[
    sampling_intervals > 0
]

if len(sampling_intervals) == 0:
    raise ValueError(
        "cannot estimate sampling frequency"
    )

fs = 1 / np.mean(sampling_intervals)

print("sampling frequency:", fs)

# ----------------------------------------
# remove disconnected leads
# ----------------------------------------

df = df[df["leads"] == 1]

# ----------------------------------------
# remove invalid temperature values
# ----------------------------------------

df = df[
    (df["temp"] > 20) &
    (df["temp"] < 45)
]

# ----------------------------------------
# remove missing values
# ----------------------------------------

df = df.dropna()

# ----------------------------------------
# clip outliers instead of deleting rows
# ----------------------------------------

numeric_cols = [
    "ecg",
    "ir",
    "temp",
    "accX",
    "accY",
    "accZ",
    "gyroX",
    "gyroY",
    "gyroZ"
]

for col in numeric_cols:

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df[col] = df[col].clip(
        lower,
        upper
    )

# ----------------------------------------
# ensure enough samples
# ----------------------------------------

if len(df) < 20:
    raise ValueError(
        "too few samples after cleaning"
    )

# ----------------------------------------
# bandpass filter functions
# ----------------------------------------

def butter_bandpass(
    lowcut,
    highcut,
    fs,
    order=4
):

    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    if high >= 1:
        high = 0.99

    if low >= high:
        raise ValueError(
            "invalid filter frequencies"
        )

    b, a = butter(
        order,
        [low, high],
        btype="band"
    )

    return b, a


def apply_filter(
    signal,
    lowcut,
    highcut,
    fs
):

    b, a = butter_bandpass(
        lowcut,
        highcut,
        fs
    )

    return filtfilt(
        b,
        a,
        signal
    )

# ----------------------------------------
# filter ECG
# ----------------------------------------

df["ecg_filtered"] = apply_filter(
    df["ecg"],
    lowcut=0.5,
    highcut=40,
    fs=fs
)

# ----------------------------------------
# filter PPG
# ----------------------------------------

df["ppg_filtered"] = apply_filter(
    df["ir"],
    lowcut=0.5,
    highcut=8,
    fs=fs
)

# ----------------------------------------
# motion magnitude
# ----------------------------------------

df["accel_magnitude"] = np.sqrt(
    df["accX"]**2 +
    df["accY"]**2 +
    df["accZ"]**2
)

df["gyro_magnitude"] = np.sqrt(
    df["gyroX"]**2 +
    df["gyroY"]**2 +
    df["gyroZ"]**2
)

# ----------------------------------------
# normalize signals
# ----------------------------------------

def normalize(x):

    std = np.std(x)

    if std == 0:
        return np.zeros(len(x))

    return (
        (x - np.mean(x)) / std
    )

df["ecg_norm"] = normalize(
    df["ecg_filtered"]
)

df["ppg_norm"] = normalize(
    df["ppg_filtered"]
)

# ----------------------------------------
# save cleaned dataset
# ----------------------------------------

df.to_csv(
    "cleaned_output.csv",
    index=False
)

# ----------------------------------------
# done
# ----------------------------------------

print("cleaning complete")
print(df.head())
print("final rows:", len(df))