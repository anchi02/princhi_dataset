import serial

try:
    ser = serial.Serial("COM5", 115200, timeout=1)
    print("COM5 is AVAILABLE")
    ser.close()

except Exception as e:
    print("COM5 is BUSY or NOT FOUND")
    print(e)