import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="tree_planting_model.tflite")
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image, input_shape):
    """Preprocess the captured image to feed into the model."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (input_shape[2], input_shape[1]))  # Resize to model input size
    input_data = np.expand_dims(image_resized / 255.0, axis=0).astype(np.float32)  # Normalize and expand dims
    return image, input_data

def capture_photo():
    """Capture an image using the webcam."""
    cap = cv2.VideoCapture(0)  # Open the default camera (webcam)
    if not cap.isOpened():
        raise Exception("Could not open the camera")

    print("Press 'SPACE' to capture the image.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Retrying...")
            continue

        # Display the live video feed
        cv2.imshow("Camera", frame)

        # Wait for the user to press 'SPACE' or 'ESC'
        key = cv2.waitKey(1)
        if key == 32:  # SPACE key
            print("Image captured!")
            break
        elif key == 27:  # ESC key
            print("Capture cancelled.")
            frame = None
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame

def detect_tree_from_camera():
    """Capture an image and detect if it contains a tree."""
    # Capture photo from the camera
    image = capture_photo()
    if image is None:
        print("No image captured. Exiting.")
        return

    # Preprocess the image for the model
    preprocessed_image, input_data = preprocess_image(image, input_details[0]['shape'])

    # Set the model input
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output (classification result)
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    print("Model Output:", output)

    # Interpret the result
    if output[0] > 0.7:  # Adjust threshold as needed
        print("Tree detected!")
        label = "Tree detected!"
    else:
        print("No tree detected.")
        label = "No tree detected."

    # Display the result with a bounding box (optional)
    height, width, _ = preprocessed_image.shape
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function to capture photo and detect tree
detect_tree_from_camera()
