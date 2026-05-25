import serial
import json
import time

# =====================================================
# CONNECT
# =====================================================

def connect_serial(

    port="COM7",

    baud_rate=115200
):

    print(
        "\nconnecting to ESP32...\n"
    )

    ser = serial.Serial(

        port,

        baud_rate,

        timeout=1
    )

    # =============================================
    # WAIT FOR ESP RESET
    # =============================================

    time.sleep(2)

    # =============================================
    # CLEAR BUFFER
    # =============================================

    ser.reset_input_buffer()

    print(
        "connected\n"
    )

    return ser

# =====================================================
# READ PACKET
# =====================================================

def read_sensor_packet(ser):

    try:

        # =========================================
        # NO DATA
        # =========================================

        if ser.in_waiting == 0:

            return None

        # =========================================
        # READ LINE
        # =========================================

        raw_line = ser.readline()

        if not raw_line:

            return None

        # =========================================
        # DECODE
        # =========================================

        line = raw_line.decode(

            "utf-8",

            errors="ignore"

        ).strip()

        # =========================================
        # EMPTY
        # =========================================

        if len(line) == 0:

            return None

        # =========================================
        # DEBUG RAW JSON
        # =========================================

        print("\nRAW LINE:")

        print(line)

        # =========================================
        # MUST LOOK LIKE JSON
        # =========================================

        if not line.startswith("{"):

            return None

        if not line.endswith("}"):

            return None

        # =========================================
        # JSON PARSE
        # =========================================

        data = json.loads(line)

        return data

    except json.JSONDecodeError:

        print(
            "json decode failed"
        )

        return None

    except Exception as e:

        print(
            "serial read error:",
            e
        )

        return None