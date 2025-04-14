import numpy as np
import scipy.signal as signal
import time
import winsound  # Windows-specific library to play sounds

# Simulated real-time data stream (you would replace this with actual sensor data reading in a real system)
def read_sensor_data(sample_rate=1000, signal_length=1000):
    """
    Simulate reading acoustic data from a rail sensor.
    """
    noise = np.random.randn(signal_length)  # White noise
    defect = np.zeros(signal_length)
    has_defect = False  # Default to no defect
    
    # Add defect with some probability
    if np.random.rand() < 0.2:  # 20% chance to have a defect
        defect_frequency = np.random.randint(10, 50)  # Random frequency for defect
        defect = np.sin(2 * np.pi * defect_frequency * np.arange(signal_length) / sample_rate)
        has_defect = True
    
    signal_with_defect = noise + defect
    return signal_with_defect, has_defect

# Step 1: Signal Preprocessing (e.g., Bandpass Filtering to remove noise)
def preprocess_signal(raw_signal, lowcut=20, highcut=200, sample_rate=1000):
    """
    Apply bandpass filter to remove frequencies outside the range of interest.
    """
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, raw_signal)
    return filtered_signal

# Step 2: Feature Extraction (e.g., FFT and statistical features)
def extract_features(signal):
    """
    Extract features from the signal. This could be in the time domain (e.g., mean, std) and frequency domain (e.g., FFT).
    """
    # Frequency domain features: FFT
    fft_signal = np.abs(np.fft.fft(signal))
    freq_domain_features = [
        np.mean(fft_signal),
        np.std(fft_signal),
        np.max(fft_signal),
        np.min(fft_signal),
        np.argmax(fft_signal)
    ]
    
    # Time-domain features: basic statistics
    time_domain_features = [
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.min(signal),
        np.median(signal)
    ]
    
    # Combine features
    return np.concatenate([freq_domain_features, time_domain_features])

# Step 3: Model Training
def train_model(X_train, y_train):
    """
    Train a Random Forest classifier to classify defects.
    """
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 4: Real-time Inference (Detect defects from streaming data)
def detect_defect(model, scaler, signal, sample_rate=1000):
    """
    Process a new signal and make a prediction on whether a defect is present.
    """
    # Step 1: Preprocess the signal (filtering)
    filtered_signal = preprocess_signal(signal, sample_rate=sample_rate)
    
    # Step 2: Extract features from the filtered signal
    features = extract_features(filtered_signal).reshape(1, -1)
    
    # Step 3: Standardize features using the same scaler that was used during training
    features_scaled = scaler.transform(features)
    
    # Step 4: Make a prediction using the trained model
    prediction = model.predict(features_scaled)
    
    # Step 5: Return prediction (1 = defect, 0 = no defect)
    return prediction[0]

# Main pipeline for real-time detection
def main():
    # Simulate a dataset for training the model (this would be real sensor data in practice)
    n_samples = 1000
    signal_length = 1000
    sample_rate = 1000  # Hz
    accuracy=100
    # Generate synthetic data for training
    X = []
    y = []
    for _ in range(n_samples):
        signal, has_defect = read_sensor_data(sample_rate=sample_rate, signal_length=signal_length)
        features = extract_features(signal)
        X.append(features)
        y.append(1 if has_defect else 0)  # Use the defect status directly
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data for training and testing
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 1: Standardize the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 2: Train the model
    model = train_model(X_train_scaled, y_train)
    
    # Simulate real-time defect detection from a sensor stream
    print("Starting real-time defect detection...")
    for _ in range(100):  
        signal, _ = read_sensor_data(sample_rate=sample_rate, signal_length=signal_length)
        prediction = detect_defect(model, scaler, signal, sample_rate)
        

        if prediction == 0:  
            print("No defect detected!")
        else:
            print("Defect detected!")
            winsound.Beep(1000, 1000)  
        
        time.sleep(0.5)

if __name__ == "__main__":
    main()
