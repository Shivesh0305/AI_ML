# import os
# import cv2
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# # Step 1: Load a Pre-trained Model (MobileNet or SSD)
# def load_pretrained_model():
#     """Loads a pre-trained MobileNet model for object detection."""
#     model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=True)
#     return model

# model = load_pretrained_model()

# # Step 2: Process Images and Detect Trees
# def detect_trees(image, model):
#     """
#     Detect trees in the image using the pre-trained model.
#     Returns bounding boxes and classification scores (dummy logic for simplicity).
#     """
#     # Resize image for the model
#     resized_image = cv2.resize(image, (224, 224))
#     input_image = preprocess_input(np.expand_dims(resized_image, axis=0))
    
#     # Predict (Dummy object detection using MobileNet)
#     preds = model.predict(input_image)
#     label = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)
#     if 'tree' in label[0][0][1]:
#         # Returning a simulated detection
#         height, width = image.shape[:2]
#         return [(50, 50, width - 50, height - 50)]  # Example bounding box
#     return []

# def calculate_average_height_diameter(image, boxes):
#     """
#     Simulate average height and diameter calculation based on bounding boxes.
#     """
#     heights = []
#     diameters = []
    
#     for box in boxes:
#         x_min, y_min, x_max, y_max = box
#         box_height = y_max - y_min
#         box_diameter = x_max - x_min
#         heights.append(box_height * 0.1)  # Convert pixels to meters (dummy factor)
#         diameters.append(box_diameter * 0.1)  # Convert pixels to meters (dummy factor)
    
#     return np.mean(heights), np.mean(diameters)

# # Step 3: Generate CSV from Images
# def process_images(image_dir, output_csv):
#     """
#     Process all images in the directory and generate a CSV with tree details.
#     """
#     data = []
#     image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

#     for image_file in image_files:
#         image_path = os.path.join(image_dir, image_file)
#         image = cv2.imread(image_path)
        
#         # Detect trees
#         boxes = detect_trees(image, model)
#         if boxes:
#             avg_height, avg_diameter = calculate_average_height_diameter(image, boxes)
#             data.append({
#                 'image_name': image_file,
#                 'average_height': avg_height,
#                 'average_diameter': avg_diameter
#             })
#         else:
#             print(f"No trees detected in {image_file}")

#     # Save to CSV
#     df = pd.DataFrame(data)
#     df.to_csv(output_csv, index=False)
#     print(f"CSV file saved to {output_csv}")

# # Step 4: Run the Script
# image_directory = "C:\\Users\\shive\\Coding\\AI\\tree_images" 
# output_csv_file = "tree_measurements.csv"
# process_images(image_directory, output_csv_file)
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Step 1: Load Pre-trained Model for Tree Detection
def load_model():
    """Loads a pre-trained MobileNetV2 model for object detection."""
    model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=True)
    return model

model = load_model()

# Step 2: Detect Trees in the Image
def detect_tree(image, model):
    """
    Detect trees in the image using the pre-trained model.
    Simulates bounding box detection for simplicity.
    """
    # Resize image for the model
    resized_image = cv2.resize(image, (224, 224))
    input_image = preprocess_input(np.expand_dims(resized_image, axis=0))
    
    # Predict
    preds = model.predict(input_image)
    label = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)
    if 'tree' in label[0][0][1]:
        # Example bounding box (simulated)
        height, width = image.shape[:2]
        return [(50, 50, width - 50, height - 50)]  # Example bounding box
    return []

# Step 3: Estimate Height and Diameter
def calculate_dimensions(image, boxes, scale_factor=0.1):
    """
    Estimate height and diameter of trees based on bounding boxes.
    Scale factor converts pixel dimensions to real-world units (meters).
    """
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        pixel_height = y_max - y_min
        pixel_diameter = x_max - x_min

        # Convert pixels to real-world dimensions
        height = pixel_height * scale_factor
        diameter = pixel_diameter * scale_factor
        return height, diameter
    return None, None

# Step 4: Process an Image
def process_image(image_path, model, scale_factor=0.1):
    """
    Process a single image to detect trees and calculate dimensions.
    """
    image = cv2.imread(image_path)
    boxes = detect_tree(image, model)
    
    if boxes:
        height, diameter = calculate_dimensions(image, boxes, scale_factor)
        print(f"Tree detected in {image_path}:")
        print(f"  - Estimated Height: {height:.2f} meters")
        print(f"  - Estimated Diameter: {diameter:.2f} meters")
    else:
        print(f"No tree detected in {image_path}")

# Step 5: Test the Code
image_directory = "C:\\Users\\shive\\Coding\\AI\\tree_images"  
scale_factor = 0.05  
image_files = [f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.png'))]

for image_file in image_files:
    process_image(os.path.join(image_directory, image_file), model, scale_factor)
