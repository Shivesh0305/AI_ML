import os, cv2, numpy as np, matplotlib.pyplot as plt, tensorflow as tf

# importing datasets and dividing into testing and training data
dataset=tf.keras.datasets.mnist
(x_train,y_train), (x_test,y_test)=dataset.load_data()

# Normalizing the data to [-1,1]
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

# Forming the Neural Network
model=tf.keras.models.Sequential()

# Adding Layers
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

# Compiles the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Trains the model
model.fit(x_train,y_train, epochs=3)

# Save the model
model.save('handwritten.keras')