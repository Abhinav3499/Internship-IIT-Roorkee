import serial
import time
from collections import deque
from datetime import datetime

ser = serial.Serial('/dev/serial0', 115200, timeout=1)

print("Receiving and filtering ADC data from STM32...\n")

sample_count = 0
start_time = time.time()

T = 0.0005
tau = 1 / 25
alpha = T / (T + tau)
gain = 0.561

prev_filtered = 0
smoothing_window = deque(maxlen=10)

try:
    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
        except UnicodeDecodeError:
            continue

        if line.isdigit():
            adc_val = int(line) * gain
            filtered_val = alpha * adc_val + (1 - alpha) * prev_filtered
            prev_filtered = filtered_val

            print(f'Raw: {adc_val:.2f}, Filtered: {filtered_val:.2f}')

            sample_count += 1

        current_time = time.time()
        if current_time - start_time >= 1.0:
            print(f"Samples per second: {sample_count}")
            sample_count = 0
            start_time = current_time

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    ser.close()
