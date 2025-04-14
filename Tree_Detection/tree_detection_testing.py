import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="C:\\Users\\shive\\Coding\\AI\\tree_planting_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_path, input_shape):
    """Preprocess the image for the model"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(image_resized, axis=0) / 255.0  # Normalize to [0,1]
    return image, input_data

def draw_bounding_boxes(image, boxes, classes, scores, threshold=0.5, labels=None):
    """Draw bounding boxes on the image"""
    height, width, _ = image.shape
    for i in range(len(scores)):
        if scores[i] >= threshold:
            # Get box coordinates
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)

            # Draw the rectangle
            color = (0, 255, 0)  # Green for bounding boxes
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

            # Add label
            label = f"Tree: {scores[i]*100:.2f}%" if labels is None else f"{labels[int(classes[i])]}: {scores[i]*100:.2f}%"
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def preprocess_image(image_path, input_shape):
    """Preprocess the image for the model"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(image_resized, axis=0).astype(np.float32) / 255.0  # Normalize to [0,1] and ensure FLOAT32
    return image, input_data

def detect_tree(image_path):
    """Run inference to classify if the image contains a tree"""
    # Preprocess the image
    image, input_data = preprocess_image(image_path, input_details[0]['shape'])

    # Set the model input
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output (classification result)
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    # print("Model Output:", output)

    # Interpret the output
    # if image_path=="C:\\Users\\shive\\Coding\\AI\\human.jpg":
    #     print("No tree detected.")

    if output[0]<0.7:  
        print("Tree detected!")
    
    else:
        print("No tree detected.")


# Run the detection
# detect_tree("C:\\Users\\shive\\Coding\\AI\\human.jpg")
detect_tree("C:\\Users\\shive\\OneDrive\\Desktop\\images.jpg")

