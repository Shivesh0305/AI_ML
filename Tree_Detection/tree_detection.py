import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import kagglehub

# Download latest version
path = kagglehub.dataset_download("mcii34/urbantree-subset-public")

print("Path to dataset files:", path)


# Prepare data
train_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2)
train_data = train_gen.flow_from_directory(path, target_size=(128, 128), class_mode='binary')

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=10)

# Save the model in TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('tree_planting_model.tflite', 'wb') as f:
    f.write(tflite_model)
