import serial
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import time

# 1. Data Collection: Read raw sensor data from serial port
def read_sensor_data(sensor_port='/dev/ttyUSB0', baud_rate=9600, signal_length=1000):

    # Read sensor data from a rail defect detection sensor connected via serial port.

    try:
        with serial.Serial(sensor_port, baud_rate, timeout=1) as ser:
            sensor_data = []

            # Collect the data based on signal length (how many samples you need)
            for _ in range(signal_length):
                line = ser.readline()  # Read one line of data
                if line:
                    try:
                        data_point = float(line.decode('utf-8').strip())
                        sensor_data.append(data_point)
                    except ValueError:
                        continue  # Skip if data is not a valid number

            sensor_data = np.array(sensor_data)
            return sensor_data
        
    except serial.SerialException as e:
        print(f"Error reading from serial port: {e}")
        return None

# 2. Data Preprocessing: Filtering the sensor data
def preprocess_signal(signal_data, lowcut=20, highcut=200, sample_rate=1000):

    # Apply bandpass filtering to the raw sensor signal.

    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, signal_data)
    return filtered_signal

# 3. Feature Extraction: Extract features from the raw signal
def extract_features(signal_data):

    # Extract features from the signal

    # Statistical features
    mean = np.mean(signal_data)
    std = np.std(signal_data)
    skew = np.mean((signal_data - mean)*3) / (std*3) if std != 0 else 0
    kurtosis = np.mean((signal_data - mean)*4) / (std*4) if std != 0 else 0
    
    # Frequency domain features (using FFT)
    freqs, psd = signal.welch(signal_data, fs=1000, nperseg=1024)
    peak_freq = freqs[np.argmax(psd)]
    
    # Combine the features into a single vector
    features = np.array([mean, std, skew, kurtosis, peak_freq])
    return features

# 4. Model Training: Train a Random Forest model using labeled data
def train_model(X_train, y_train):

    #Train a Random Forest Classifier on the training data.

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# 5. Defect Detection: Use the trained model to classify new sensor data
def detect_defect(model, sensor_data):

    # Use the trained model to predict if a defect is present in the new sensor data.

    # Preprocess the sensor data
    processed_data = preprocess_signal(sensor_data)
    
    # Extract features from the processed data
    features = extract_features(processed_data)
    
    # Reshape the feature vector for prediction
    features = features.reshape(1, -1)
    
    # Make the prediction
    prediction = model.predict(features)
    return prediction

def main():
    # Step 1: Data Collection (from the sensor)
    sensor_port = '/dev/ttyUSB0'  
    baud_rate = 9600  
    signal_length = 1000  # Number of samples to collect from sensor

    # Collect sensor data 
    sensor_data = read_sensor_data(sensor_port=sensor_port, baud_rate=baud_rate, signal_length=signal_length)
    if sensor_data is None:
        print("Failed to collect sensor data.")
        return

    # Step 2: Preprocessing the data 
    sample_rate = 1000  # Replace with actual sample rate of sensor
    filtered_data = preprocess_signal(sensor_data, sample_rate=sample_rate)
    
    # Step 3: Feature Extraction 
    features = extract_features(filtered_data)
    print("Extracted features: ", features)

    # Step 4: Training the model (this would be done beforehand with labeled data)

    # Load or prepare your training data
    # X_train = extracted from sensor signals
    # y_train = Labels (0 for no defect, 1 for defect)

    # For now, let's simulate some data for training
    # In reality, you would use your historical labeled data

    X_train = 1#--> TO BE FILLED
    y_train = 1#--> TO BE FILLED

    # Train the model
    model = train_model(X_train, y_train)

    # Step 5: Use the trained model to detect defects in the new sensor data
    prediction = detect_defect(model, sensor_data)
    if prediction == 1:
        print("Defect detected!")
    else:
        print("No defect detected.")

if _name_ == "_main_":
    main()