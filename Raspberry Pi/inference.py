import torch
import numpy as np
import joblib
import time
import csv
from collections import deque

WINDOW_SIZE = 2000
INFERENCE_INTERVAL = 1  

# Change the path of the models as per the new directory structure
MODEL_SCALER_MAP = {
    "ResNet-Motor": ("/home/pi/project_with_plot_tkinter_uart/motor/model.pt", "/home/pi/project_with_plot_tkinter_uart/motor/scaler.pkl"),
    "ResNet-Boost": ("/home/pi/project_with_plot_tkinter/model_1/resnet40_50.pt", "/home/pi/project_with_plot_tkinter/model_1/standard_scaler_40_50.pkl"),
    "CNN-Boost": ("/home/pi/project_with_plot_tkinter/model_2/CNN_2D.pt", "/home/pi/project_with_plot_tkinter/model_2/standard_scaler_40_50.pkl"),
    "VGG-Boost": ("/home/pi/project_with_plot_tkinter/model_3/VGG.pt", "/home/pi/project_with_plot_tkinter/model_3/standard_scaler_40_50.pkl"),
    "U-Net-Boost": ("/home/pi/project_with_plot_tkinter/model_4/UNet2D_40x50.pt", "/home/pi/project_with_plot_tkinter/model_4/standard_scaler_40_50.pkl"),
    "MLP-Boost": ("/home/pi/project_with_plot_tkinter/model_5/MLP_2D.pt", "/home/pi/project_with_plot_tkinter/model_5/standard_scaler_40_50.pkl")
}

def load_model_and_scaler(model_choice):
    try:
        model_path, scaler_path = MODEL_SCALER_MAP[model_choice]
        model = torch.jit.load(model_path, map_location='cpu')
        model.eval()
        scaler = joblib.load(scaler_path)
        print(f"Loaded {model_choice} successfully.")
        return model, scaler
    except Exception as e:
        print(f"Error loading model or scaler for {model_choice}: {e}")
        return None, None

def inference_worker_improved(model, scaler, shared_state, stop_event):
    from datetime import datetime

    label_map = {0: "Normal", 1: "Abnormal"}
    history = deque(maxlen=2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    volt_log_file = open(f"reading_log_{timestamp}.csv", "w", newline='')
    pred_log_file = open(f"prediction_log_{timestamp}.csv", "w", newline='')

    volt_writer = csv.writer(volt_log_file)
    volt_writer.writerow(["Timestamp", "Voltage"])

    pred_writer = csv.writer(pred_log_file)
    pred_writer.writerow(["Timestamp", "Prediction"])

    while not stop_event.is_set():
        time.sleep(INFERENCE_INTERVAL)

        try:
            with shared_state['lock']:
                data = np.array(shared_state['queue'])

            now = datetime.now().isoformat()
            for v in data:
                volt_writer.writerow([now, round(v, 3)])
            volt_log_file.flush()

            scaled = scaler.transform(data.reshape(1, -1))
            input_tensor = torch.tensor(scaled, dtype=torch.float32).reshape(1, 1, 40, 50)

            with torch.no_grad():
                output = model(input_tensor)
                pred_idx = torch.argmax(output, dim=1).item()

            history.append(pred_idx)

            if list(history) == [1, 1]:
                final = 1
            else:
                final = 0

            prediction = label_map[final]

            with shared_state['lock']:
                shared_state['prediction'] = prediction
                shared_state['prediction_class'] = final

            pred_writer.writerow([now, prediction])
            pred_log_file.flush()

            print(f"🔍 Prediction: {prediction}")

        except Exception as e:
            print(f"Inference error: {e}")
            with shared_state['lock']:
                shared_state['prediction'] = "Error"
                shared_state['prediction_class'] = 1
