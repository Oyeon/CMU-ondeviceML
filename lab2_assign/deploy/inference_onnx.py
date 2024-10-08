# import onnx
# import onnxruntime
# import numpy as np
# import torch
# import time  # Import time module for latency measurement
# from config import CONFIG

# def load_onnx_model(onnx_model_path):
#     onnx_model = onnx.load(onnx_model_path)
#     onnx.checker.check_model(onnx_model)
#     print(f"ONNX model {onnx_model_path} is loaded and checked.")
#     ort_session = onnxruntime.InferenceSession(onnx_model_path)
#     return ort_session

# def infer(ort_session, input_data):
#     ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    
#     # Measure inference latency
#     start_time = time.time()
#     ort_outs = ort_session.run(None, ort_inputs)
#     end_time = time.time()
    
#     # Calculate latency
#     latency = end_time - start_time
#     print(f"Inference Latency: {latency:.6f} seconds")
    
#     probabilities = softmax(ort_outs[0])
#     return probabilities

# def softmax(x):
#     e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
#     return e_x / e_x.sum(axis=1, keepdims=True)

# def main():
#     onnx_model_path = 'model_checkpoints/model_20241005_204235_epoch_2.onnx'  # Update path
#     ort_session = load_onnx_model(onnx_model_path)
    
#     # Example input data
#     dummy_tensor = torch.randn(1, 49)  # Ensure this matches the model's expected input size
#     input_data = dummy_tensor.numpy().astype(np.float32)
    
#     # Perform inference and measure latency
#     probabilities = infer(ort_session, input_data)
#     predicted_class = np.argmax(probabilities, axis=1)
#     print(f"Predicted class: {predicted_class}")
#     print(f"Probabilities: {probabilities}")

# if __name__ == "__main__":
#     main()

import onnx
import onnxruntime
import numpy as np
import torch
import time

def load_onnx_model(onnx_model_path):
    """
    Loads and checks the ONNX model, then creates an inference session.

    Args:
        onnx_model_path (str): Path to the ONNX model file.

    Returns:
        onnxruntime.InferenceSession: The inference session for the model.
    """
    try:
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX model '{onnx_model_path}' is loaded and checked.")
        ort_session = onnxruntime.InferenceSession(onnx_model_path)
        return ort_session
    except onnx.onnx_cpp2py_export.checker.ValidationError as e:
        print(f"Model validation failed: {e}")
        raise
    except Exception as e:
        print(f"Failed to load ONNX model: {e}")
        raise

def infer(ort_session, input_data):
    """
    Performs inference using the ONNX runtime session.

    Args:
        ort_session (onnxruntime.InferenceSession): The ONNX runtime session.
        input_data (numpy.ndarray): The input data for inference.

    Returns:
        numpy.ndarray: The output probabilities from the model.
    """
    try:
        ort_inputs = {ort_session.get_inputs()[0].name: input_data}
        start_time = time.time()
        ort_outs = ort_session.run(None, ort_inputs)
        end_time = time.time()
        latency = end_time - start_time
        print(f"Inference Latency: {latency:.6f} seconds")
        probabilities = softmax(ort_outs[0])
        return probabilities
    except Exception as e:
        print(f"Inference failed: {e}")
        raise

def softmax(x):
    """
    Applies the softmax function to the input array.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Softmax probabilities.
    """
    try:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
    except Exception as e:
        print(f"Softmax computation failed: {e}")
        raise

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocesses the image for model inference:
    - Loads the image.
    - Resizes it to the target size.
    - Converts it to a NumPy array with appropriate dimensions.

    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Desired image size (height, width).

    Returns:
        numpy.ndarray: Preprocessed image ready for inference.
    """
    try:
        from PIL import Image
        # Load image
        image = Image.open(image_path).convert('RGB')
        # Resize image
        image = image.resize(target_size)
        # Convert to NumPy array
        image_np = np.array(image).astype(np.float32)
        # Normalize the image (optional, based on model requirements)
        image_np /= 255.0
        # Transpose to channel-first format if required
        image_np = np.transpose(image_np, (2, 0, 1))
        # Add batch dimension
        image_np = np.expand_dims(image_np, axis=0)
        return image_np
    except Exception as e:
        print(f"Image preprocessing failed: {e}")
        raise

def main():
    onnx_model_path = 'yolov8m.onnx'
    ort_session = load_onnx_model(onnx_model_path)
    
    # Example input data processing
    # If using an image as input:
    # Replace 'path_to_image.jpg' with your actual image path
    # image_path = 'path_to_image.jpg'
    # input_data = preprocess_image(image_path)
    
    # If using dummy data with shape (1, 3, 224, 224):
    input_shape = (1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 image
    dummy_tensor = torch.randn(*input_shape).float()
    input_data = dummy_tensor.numpy().astype(np.float32)
    
    # Perform inference and measure latency
    probabilities = infer(ort_session, input_data)
    predicted_class = np.argmax(probabilities, axis=1)
    print(f"Predicted class: {predicted_class}")
    print(f"Probabilities: {probabilities}")

if __name__ == "__main__":
    main()