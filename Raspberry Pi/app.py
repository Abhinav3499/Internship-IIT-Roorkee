import threading
import time
from tkinter import *
from tkinter import ttk, messagebox
from collections import deque
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import serial

from inference import load_model_and_scaler, inference_worker_improved, MODEL_SCALER_MAP

WINDOW_SIZE     = 2000
SAMPLE_RATE     = 2000
UPDATE_INTERVAL = 50         
#GAIN            = 0.0245   # DC-DC Boost Converter Gain 
GAIN            = 0.561     # DC Motor Gain

T     = 1.0 / SAMPLE_RATE    
tau   = 1.0 / 25.0
alpha = T / (T + tau)        

smoothing_window = deque(maxlen=10)

try:
    ser = serial.Serial('/dev/serial0', 115200, timeout=1)
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    messagebox.showerror("Serial Error",
                         f"Could not open serial port /dev/serial0.\n{e}")
    exit()

try:
    first = ser.readline().decode('utf-8', errors='ignore').strip()
    if first.isdigit():
        prev_filtered = [int(first) * GAIN]
    else:
        prev_filtered = [0.0]
except Exception:
    prev_filtered = [0.0]

buffer = "" 

def reader_thread(shared_state, stop_event):
    global buffer
    while not stop_event.is_set():
        try:
            chunk = ser.read(ser.in_waiting or 1).decode('utf-8', errors='ignore')
            buffer += chunk

            if '\n' in buffer:
                lines = buffer.split('\n')
                buffer = lines.pop() 

                out_vals = []
                for line in lines:
                    line = line.strip()
                    if not line or not line.isdigit():
                        continue

                    raw = int(line)
                    voltage = raw * GAIN

                    filtered = alpha * voltage + (1 - alpha) * prev_filtered[0]
                    prev_filtered[0] = filtered

                    smoothing_window.append(filtered)
                    smoothed = sum(smoothing_window) / len(smoothing_window)

                    out_vals.append(smoothed)

                if out_vals:
                    with shared_state['lock']:
                        shared_state['queue'].extend(out_vals)

        except Exception as e:
            print(f"[Reader] {e}")
        time.sleep(0.001)

class RealTimeMonitor:
    def __init__(self, root, shared_state):
        self.root = root
        self.shared_state = shared_state
        self.setup_gui()
        self.update_gui()

    def setup_gui(self):
        self.root.title("Real-Time Voltage Monitor")
        self.root.geometry("900x600")
        self.root.configure(bg='#f0f2f5')

        cf = Frame(self.root, bg='#f0f2f5', pady=5)
        cf.pack(fill=X)

        Label(cf, text="Prediction:", font=("Arial",12), bg='#f0f2f5')\
            .pack(side=LEFT, padx=(10,5))
        self.pred_var = StringVar(self.root, "Initializing…")
        Label(cf, textvariable=self.pred_var,
              font=("Arial",12,"bold"), fg="#0056b3", bg="#e7f3ff",
              padx=10, pady=5, relief="groove")\
            .pack(side=LEFT)

        self.status_var = StringVar(self.root, "Running")
        Label(cf, textvariable=self.status_var,
              font=("Arial",10,"bold"), fg="green", bg='#f0f2f5')\
            .pack(side=RIGHT, padx=10)
        Button(cf, text="Quit", command=self.quit_app,
               bg="#dc3545", fg="white", font=("Arial",10))\
            .pack(side=RIGHT, padx=5)

        pf = Frame(self.root, bg='white')
        pf.pack(fill=BOTH, expand=True, padx=10, pady=10)

        self.fig = Figure(figsize=(8,4), dpi=100)
        self.ax  = self.fig.add_subplot(111)
        x = np.arange(WINDOW_SIZE)
        self.line, = self.ax.plot(x, np.zeros(WINDOW_SIZE), lw=1.5)

        self.ax.set_xlim(0, WINDOW_SIZE)
        self.ax.set_ylim(0, 400)

        self.ax.set_title("Voltage (V)")
        self.ax.set_xlabel("Sample")
        self.ax.set_ylabel("V")
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=pf)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    def update_gui(self):
        try:
            with self.shared_state['lock']:
                data = list(self.shared_state['queue'])
                pred = self.shared_state.get('prediction', "N/A")
                pc   = self.shared_state.get('prediction_class', 0)

            self.pred_var.set(pred)
            self.line.set_color(['blue', 'red'][pc])

            if len(data) == WINDOW_SIZE:
                self.line.set_ydata(data)

                y_min = min(data)
                y_max = max(data)
                margin = (y_max - y_min) * 0.1 if y_max != y_min else 10
                self.ax.set_ylim(y_min - margin, y_max + margin)

            self.canvas.draw_idle()
        except Exception as e:
            print(f"[GUI] {e}")

        self.root.after(UPDATE_INTERVAL, self.update_gui)


    def quit_app(self):
        self.status_var.set("Shutting down…")
        self.root.after(100, self.root.quit)

def main():
    sel_root = Tk()
    sel_root.title("Select Model")
    model_var = StringVar(sel_root, "ResNet-Boost")

    def on_start():
        sel_root.destroy()

    Label(sel_root, text="Select a model:", font=("Arial",12)).pack(pady=10)
    cb = ttk.Combobox(sel_root, textvariable=model_var,
                      values=list(MODEL_SCALER_MAP.keys()),
                      state="readonly")
    cb.pack(pady=5, padx=10)
    Button(sel_root, text="Start", command=on_start).pack(pady=10)
    sel_root.mainloop()

    model_name = model_var.get()
    model, scaler = load_model_and_scaler(model_name)
    if model is None or scaler is None:
        messagebox.showerror("Error", f"Could not load '{model_name}'")
        return

    shared_state = {
        "queue": deque([0.0]*WINDOW_SIZE, maxlen=WINDOW_SIZE),
        "lock": threading.Lock(),
        "prediction": "Initializing…",
        "prediction_class": 0
    }
    stop_evt = threading.Event()

    threading.Thread(target=reader_thread,
                     args=(shared_state, stop_evt),
                     daemon=True).start()
    threading.Thread(target=inference_worker_improved,
                     args=(model, scaler, shared_state, stop_evt),
                     daemon=True).start()

    root = Tk()
    app = RealTimeMonitor(root, shared_state)

    def on_close():
        stop_evt.set()
        time.sleep(0.1)
        ser.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

if __name__ == "__main__":
    main()
