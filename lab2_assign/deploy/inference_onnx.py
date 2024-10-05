import onnx
import onnxruntime
import numpy as np
import torch
import time  # Import time module for latency measurement
from config import CONFIG

def load_onnx_model(onnx_model_path):
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model {onnx_model_path} is loaded and checked.")
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    return ort_session

def infer(ort_session, input_data):
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    
    # Measure inference latency
    start_time = time.time()
    ort_outs = ort_session.run(None, ort_inputs)
    end_time = time.time()
    
    # Calculate latency
    latency = end_time - start_time
    print(f"Inference Latency: {latency:.6f} seconds")
    
    probabilities = softmax(ort_outs[0])
    return probabilities

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def main():
    onnx_model_path = 'model_checkpoints/model_20241005_204235_epoch_2.onnx'  # Update path
    ort_session = load_onnx_model(onnx_model_path)
    
    # Example input data
    dummy_tensor = torch.randn(1, 49)  # Ensure this matches the model's expected input size
    input_data = dummy_tensor.numpy().astype(np.float32)
    
    # Perform inference and measure latency
    probabilities = infer(ort_session, input_data)
    predicted_class = np.argmax(probabilities, axis=1)
    print(f"Predicted class: {predicted_class}")
    print(f"Probabilities: {probabilities}")

if __name__ == "__main__":
    main()

##########################################################################################
# import onnx
# import onnxruntime
# import numpy as np
# import torch
# import time
# # from config import CONFIG  # Remove if not used

# def load_onnx_model(onnx_model_path: str) -> onnxruntime.InferenceSession:
#     onnx_model = onnx.load(onnx_model_path)
#     onnx.checker.check_model(onnx_model)
#     print(f"ONNX model {onnx_model_path} is loaded and checked.")
#     return onnxruntime.InferenceSession(onnx_model_path)

# def infer(ort_session: onnxruntime.InferenceSession, input_data: np.ndarray) -> np.ndarray:
#     ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    
#     # Measure inference latency using a context manager
#     with time_measurement("Inference Latency"):
#         ort_outs = ort_session.run(None, ort_inputs)
    
#     probabilities = softmax(ort_outs[0])
#     return probabilities

# def softmax(x: np.ndarray) -> np.ndarray:
#     e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
#     return e_x / e_x.sum(axis=1, keepdims=True)

# class time_measurement:
#     def __init__(self, label: str):
#         self.label = label

#     def __enter__(self):
#         self.start_time = time.time()

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         end_time = time.time()
#         latency = end_time - self.start_time
#         print(f"{self.label}: {latency:.6f} seconds")

# def main():
#     onnx_model_path = 'model_checkpoints/model_20241005_204235_epoch_2.onnx'
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