import cv2
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow import keras

# Load CSV data containing tree info
tree_data = pd.read_csv('tree_data.csv')  # Your CSV file

# Load a pretrained object detection model (e.g., COCO model in TensorFlow)
model = tf.saved_model.load('ssd_mobilenet_v2_coco/saved_model')  # Change path based on your model

def load_image(image_path):
    """Loads an image and converts it to a tensor."""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (300, 300))  # Resize for the model
    return np.expand_dims(img_resized, axis=0)

def detect_tree(image_tensor):
    """Uses the object detection model to find trees in the image."""
    # Model's inference step
    detections = model(image_tensor)
    
    # Post-processing to extract detected objects
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    
    # Define a threshold for detecting trees
    tree_class_id = 19  # For example, 'tree' class in COCO dataset
    threshold = 0.5
    
    tree_boxes = []
    for i in range(len(scores)):
        if scores[i] > threshold and classes[i] == tree_class_id:
            box = boxes[i]  # [ymin, xmin, ymax, xmax]
            tree_boxes.append(box)
    
    return tree_boxes

def match_tree_with_data(tree_boxes):
    """Matches detected tree boxes with height and width from the CSV file."""
    if len(tree_boxes) > 0:
        # Assuming you have tree IDs to match (you may need a more complex logic here)
        # For simplicity, we'll return the first match based on the tree ID
        tree_id = 1  # Example, you may detect tree IDs or classify based on features
        tree_info = tree_data[tree_data['tree_id'] == tree_id]
        height = tree_info['height'].values[0]
        width = tree_info['width'].values[0]
        return height, width
    else:
        return None, None

def visualize_detection(image_path, tree_boxes):
    """Visualizes the tree detection results on the image."""
    img = cv2.imread(image_path)
    for box in tree_boxes:
        ymin, xmin, ymax, xmax = box
        img = cv2.rectangle(img, (int(xmin*img.shape[1]), int(ymin*img.shape[0])),
                            (int(xmax*img.shape[1]), int(ymax*img.shape[0])), (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

# Testing the system
image_path = 'test_image.jpg'  # Path to the test image
image_tensor = load_image(image_path)
tree_boxes = detect_tree(image_tensor)
height, width = match_tree_with_data(tree_boxes)

if height and width:
    print(f"Detected tree's height: {height} meters, width: {width} meters.")
else:
    print("No tree detected in the image.")
