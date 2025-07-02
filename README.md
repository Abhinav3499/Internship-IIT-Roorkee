# Internship-IIT-Roorkee
Real Time Anomaly Detection in DC-DC Boost Converter Signal

## Project Overview

This project implements real-time anomaly detection for signals from a DC-DC Boost Converter (and DC Motor) using deep learning models. The workflow includes data collection from STM32, model training on Kaggle, deployment to Raspberry Pi, and real-time inference with data logging.

---

## 1. Data Collection from STM32

- **Hardware Used:** STM32 Nucleo-F303ZE
- **Signal Acquisition:**
  - **ADC Input:** PF4 (ADC1_IN5) is used to read the analog signal.
  - **UART Output:** PC4 (USART2_TX) and PC5 (USART2_RX) are used for UART communication with the Raspberry Pi.
- **Firmware:** The STM32 firmware reads ADC values, applies a digital filter, and transmits filtered values over UART at 115200 baud.

**Pin Connections:**
- STM32 PF4: Connect to the analog signal source (e.g., output of Boost Converter).
- STM32 PC4 (TX): Connect to Raspberry Pi RX (GPIO15, physical pin 10).
- STM32 PC5 (RX): Connect to Raspberry Pi TX (GPIO14, physical pin 8).
- Common GND between STM32 and Raspberry Pi.

---

## 2. Uploading Data to Kaggle

1. **Collect Data:**
   - Use the Raspberry Pi to receive UART data and log it as CSV files (see `Raspberry Pi/uart.py`).
   - Example files: `normal.csv`, `random.csv`, `load.csv` in `Dataset/DC Motor/` or `Dataset/Boost Converter/`.

2. **Upload to Kaggle:**
   - Go to [Kaggle Datasets](https://www.kaggle.com/datasets).
   - Click "New Dataset" and upload your CSV files.
   - Fill in the dataset details and make it public or private as needed.

---

## 3. Model Training on Kaggle

1. **Open the Notebook:**
   - Use `Kaggle/model-training.ipynb` as your starting point.

2. **Load the Dataset:**
   - Use Kaggle's data path, e.g.:
     ```python
     import pandas as pd
     normal = pd.read_csv('/kaggle/input/your-dataset/normal.csv')
     load = pd.read_csv('/kaggle/input/your-dataset/load.csv')
     random = pd.read_csv('/kaggle/input/your-dataset/random.csv')
     ```

3. **Preprocess and Generate Training Data:**
   - The notebook provides functions to segment, label, and augment the data.

4. **Activate GPU:**
   - In the Kaggle notebook, go to `Settings` (right sidebar) and set "Accelerator" to "GPU".

5. **Train the Model:**
   - The notebook includes code for training several models (ResNet, CNN, VGG, U-Net, MLP).
   - Training uses PyTorch and scikit-learn.

6. **Save the Model and Scaler:**
   - After training, save the model and scaler:
     ```python
     import torch
     import joblib
     traced_model = torch.jit.trace(model, dummy_input)
     traced_model.save('model.pt')
     joblib.dump(scaler, 'scaler.pkl')
     ```
   - Download these files from the Kaggle notebook output.

---

## 4. Deploying to Raspberry Pi

1. **Setup Raspberry Pi:**
   - Install dependencies:
     ```bash
     pip install -r Raspberry\ Pi/requirements.txt
     ```
   - Ensure `pyserial`, `torch`, `joblib`, `matplotlib`, `tk`, etc. are installed.

2. **Copy Model Files:**
   - Create a new folder in `Raspberry Pi/` (e.g., `model_1/`).
   - Add your `model.pt` and `scaler.pkl` to this folder.

3. **Update Model Paths:**
   - Edit `Raspberry Pi/inference.py` and update `MODEL_SCALER_MAP` with your new model folder and file names.

---

## 5. Running Real-Time Inference

- Run the main application:
  ```bash
  python Raspberry\ Pi/app.py
  ```
- The app:
  - Reads UART data from STM32.
  - Applies the trained model for anomaly detection.
  - Displays real-time plots and predictions.
  - **Records all incoming data and predictions to CSV logs.**
- **Retraining:** If new data is collected, you can upload it to Kaggle and repeat the training process.

---

## 6. Notes on Data Logging and Retraining

- Every run of the inference app logs both the raw voltage and the model's predictions.
- These logs can be used to further improve or retrain your models.

---

## 7. Hardware Pin Summary

| Function         | STM32 Pin | STM32 Peripheral | Raspberry Pi Pin |
|------------------|-----------|------------------|------------------|
| ADC Input        | PF4       | ADC1_IN5         | Analog Signal    |
| UART TX (to Pi)  | PC4       | USART2_TX        | Pin:8 (RX)       |
| UART RX (from Pi)| PC5       | USART2_RX        | Pin:10 (TX)      |
| GND              | GND       | -                | GND              |

---

## 8. Troubleshooting

- **Serial Port:** Ensure `/dev/serial0` is enabled on Raspberry Pi (`raspi-config` > Interface Options > Serial).
- **Baud Rate:** Both STM32 and Pi must use 115200 baud.
- **Permissions:** You may need to add your user to the `dialout` group on the Pi for serial access.

---

## 9. References

- STM32 Nucleo-F303ZE [Datasheet](https://www.st.com/en/evaluation-tools/nucleo-f303ze.html)
- [Kaggle Documentation](https://www.kaggle.com/docs)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
