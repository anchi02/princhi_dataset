import serial
import json
import time

# =====================================================
# CONNECT
# =====================================================

def connect_serial(

    port="COM5",

    baud_rate=921600
):

    print(
        "\nconnecting to ESP32...\n"
    )

    ser = serial.Serial(

        port,

        baud_rate,

        timeout=1
    )

    time.sleep(2)

    print("connected\n")

    return ser

# =====================================================
# READ SENSOR PACKET
# =====================================================

def read_sensor_packet(ser):

    try:

        raw_line = ser.readline()

        if not raw_line:
            return None

        line = raw_line.decode(

            "utf-8",

            errors="ignore"

        ).strip()

        if len(line) == 0:
            return None

        data = json.loads(line)

        required_keys = [

            "timestamp",

            "ecg",

            "ir",

            "red",

            "temp",

            "accX",

            "accY",

            "accZ",

            "gyroX",

            "gyroY",

            "gyroZ"
        ]

        for key in required_keys:

            if key not in data:
                return None

        return data

    except json.JSONDecodeError:

        return None

    except Exception as e:

        print(
            "serial read error:",
            e
        )

        return None