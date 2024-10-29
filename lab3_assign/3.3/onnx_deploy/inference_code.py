import onnxruntime
import numpy as np
import time
from PIL import Image

models = ['dynamic_quantized_best.onnx', 'FP16_best.onnx', 'default.onnx']

# input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)
image_path = 'datasets/val/images/002115.png'
image = Image.open(image_path)
image = image.resize((640, 640))  # Resize to match model input size
input_data = np.array(image).astype(np.float32)
input_data = np.transpose(input_data, (2, 0, 1))  # Change to (C, H, W) format
input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension


for model_path in models:
    # Set session options (if needed)
    session_options = onnxruntime.SessionOptions()
    
    # Load the model
    session = onnxruntime.InferenceSession(model_path, sess_options=session_options)
    
    input_name = session.get_inputs()[0].name
    input_type = session.get_inputs()[0].type

    # Check and convert input data type
    if 'float16' in input_type:
        model_input = input_data.astype(np.float16)
    elif 'float' in input_type or 'float32' in input_type:
        model_input = input_data.astype(np.float32)
    elif 'quantized' in model_path.lower() or 'int8' in input_type:
        # For quantized models, convert to uint8 or int8
        model_input = (input_data * 127).astype(np.int8)  # Example scaling
    else:
        model_input = input_data  # Default to float32

    # Measure inference time
    times = []
    for _ in range(10):
        start_time = time.time()
        outputs = session.run(None, {input_name: model_input})
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    print(f"Model {model_path} average inference time: {avg_time:.6f} seconds")
