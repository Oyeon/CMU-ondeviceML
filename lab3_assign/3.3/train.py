import torch
from ultralytics import YOLO
import os
from datetime import datetime
import onnx
from onnxruntime.quantization import (
    quantize_static,
    quantize_dynamic,
    CalibrationDataReader,
    QuantType
)
import json
import time
import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms
import yaml
import glob

def compute_mean(metric):
    """
    Computes the mean of a metric if it's an array-like object (list or NumPy array),
    handling empty lists or arrays. Otherwise, returns the scalar value.

    Args:
        metric (array-like or scalar): The metric to compute the mean for.

    Returns:
        float: The mean value of the metric.
    """
    if isinstance(metric, (list, np.ndarray)):
        if len(metric) == 0:
            print("Warning: Received an empty list or array for metric. Returning 0.0.")
            return 0.0  # Or np.nan, based on your preference
        else:
            return float(np.mean(metric))
    else:
        return float(metric)


# class YOLOCalibrationDataReader(CalibrationDataReader):
#     def __init__(self, data_yaml_path, img_size, batch_size=1):
#         self.data_yaml_path = data_yaml_path
#         self.img_size = img_size
#         self.batch_size = batch_size
#         self.image_paths = self._gather_image_paths()
#         self.dataset_iterator = iter(self._batch_data())
#         self.input_name = None  

#     def _batch_data(self):
#         transform = transforms.Compose([
#             transforms.Resize((self.img_size, self.img_size)),
#             transforms.ToTensor(),
#         ])
#         for img_path in self.image_paths:
#             img = Image.open(img_path).convert('RGB')
#             img = transform(img).unsqueeze(0).numpy().astype(np.float32)
#             yield img  # Yield one image at a time

#     def get_next(self):
#         batch = next(self.dataset_iterator, None)
#         if batch is not None:
#             return {self.input_name: batch}
#         else:
#             return None

#     def _gather_image_paths(self):
#         """
#         Gathers image paths from the validation dataset specified in data.yaml.

#         Returns:
#             list: List of image file paths.
#         """
#         # Load the data.yaml file
#         with open(self.data_yaml_path, 'r') as f:
#             data_config = yaml.safe_load(f)

#         # Get the validation images path or list
#         val_data = data_config.get('val')
#         if not val_data:
#             raise ValueError("The 'val' key is missing in data.yaml.")

#         image_paths = []
#         if isinstance(val_data, str):
#             if os.path.isdir(val_data):
#                 # It's a directory
#                 image_dir = val_data
#                 image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
#                 for ext in image_extensions:
#                     image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
#             elif os.path.isfile(val_data):
#                 # It's a file containing image paths
#                 with open(val_data, 'r') as f:
#                     image_paths = [line.strip() for line in f.readlines()]
#             else:
#                 raise FileNotFoundError(f"Validation data path not found: {val_data}")
#         elif isinstance(val_data, list):
#             # It's a list of image paths
#             image_paths = val_data
#         else:
#             raise TypeError("Unsupported type for 'val' in data.yaml.")

#         if not image_paths:
#             raise FileNotFoundError(f"No images found for validation in: {val_data}")

#         print(f"Found {len(image_paths)} validation images.")
#         return image_paths            

class YOLOCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_yaml_path, img_size, batch_size=1):
        self.data_yaml_path = data_yaml_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.image_paths = self._gather_image_paths()
        self.dataset_iterator = iter(self._batch_data())
        self.input_name = None  
        print(f"Initialized YOLOCalibrationDataReader with {len(self.image_paths)} images.")

    def _batch_data(self):
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])
        batch = []
        for img_path in self.image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img = transform(img).unsqueeze(0).numpy().astype(np.float32)
                batch.append(img)
                if len(batch) == self.batch_size:
                    batch_np = np.concatenate(batch, axis=0)
                    # print(f"Yielding a batch of size: {batch_np.shape}")
                    yield batch_np
                    batch = []
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
        if batch:
            batch_np = np.concatenate(batch, axis=0)
            # print(f"Yielding the final batch of size: {batch_np.shape}")
            yield batch_np

    def get_next(self):
        try:
            batch = next(self.dataset_iterator)
            # print(f"Retrieved a batch from iterator: {batch.shape}")
            return {self.input_name: batch}
        except StopIteration:
            print("No more data to read in CalibrationDataReader.")
            return None

    def rewind(self):
        self.dataset_iterator = iter(self._batch_data())
        print("Rewound the CalibrationDataReader.")
        
    def _gather_image_paths(self):
        """
        Gathers image paths from the validation dataset specified in data.yaml.

        Returns:
            list: List of image file paths.
        """
        # Load the data.yaml file
        with open(self.data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)

        # Get the validation images path or list
        val_data = data_config.get('val')
        if not val_data:
            raise ValueError("The 'val' key is missing in data.yaml.")

        image_paths = []
        if isinstance(val_data, str):
            if os.path.isdir(val_data):
                # It's a directory
                image_dir = val_data
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
                for ext in image_extensions:
                    found = glob.glob(os.path.join(image_dir, ext))
                    print(f"Searching for {ext} files: Found {len(found)} files.")
                    image_paths.extend(found)
            elif os.path.isfile(val_data):
                # It's a file containing image paths
                with open(val_data, 'r') as f:
                    image_paths = [line.strip() for line in f.readlines()]
            else:
                raise FileNotFoundError(f"Validation data path not found: {val_data}")
        elif isinstance(val_data, list):
            # It's a list of image paths
            image_paths = val_data
        else:
            raise TypeError("Unsupported type for 'val' in data.yaml.")

        if not image_paths:
            raise FileNotFoundError(f"No images found for validation in: {val_data}")

        print(f"Total validation images found: {len(image_paths)}")
        return image_paths

def evaluate_accuracy(metrics):
    """
    Extracts mAP@0.5 from the evaluation metrics.

    Returns:
        float: mAP at IoU 0.5 as a percentage.
    """
    try:
        mAP_0_5 = metrics['mAP50']  # mAP@0.5
        print(f"mAP@0.5: {mAP_0_5:.4f}")
        return mAP_0_5 * 100  # Convert to percentage
    except KeyError as e:
        print(f"Error accessing mAP@0.5: {e}")
        return None

def evaluate_precision(metrics):
    """
    Extracts mean precision from the evaluation metrics.

    Returns:
        float: Precision as a percentage.
    """
    try:
        precision = metrics['precision']  # Mean precision
        print(f"Precision: {precision:.4f}")
        return precision * 100  # Convert to percentage
    except KeyError as e:
        print(f"Error accessing precision: {e}")
        return None


def quantize_yolo_model_static(model_path, quantized_onnx_model_path, config, data_yaml_path, img_size, quantized_pt_model_path):
    """
    Applies static quantization to the YOLOv8 model exported in ONNX format and saves it in .pt format.
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # Export the model to ONNX format with dynamic input shapes (if supported)
    export_success = model.export(format="onnx", dynamic=True)
    onnx_export_path = os.path.splitext(model_path)[0] + '.onnx'

    if not os.path.exists(onnx_export_path):
        print(f"ONNX export failed or ONNX file not found at {onnx_export_path}")
        return
    print(f"YOLOv8 model exported to ONNX at {onnx_export_path}")

    # Save the model in .pt format
    torch.save(model.model.state_dict(), quantized_pt_model_path)
    print(f"Model saved in .pt format at {quantized_pt_model_path}")

    # Apply static quantization using ONNX Runtime
    calibration_data_reader = YOLOCalibrationDataReader(data_yaml_path, img_size, batch_size=1)
    session = ort.InferenceSession(onnx_export_path)
    calibration_data_reader.input_name = session.get_inputs()[0].name

    quantize_static(
        onnx_export_path,
        quantized_onnx_model_path,
        calibration_data_reader,
        weight_type=QuantType.QInt8
    )
    print(f"Static quantization completed. Quantized model saved at {quantized_onnx_model_path}")

    return quantized_onnx_model_path


def quantize_yolo_model_dynamic(model_path, quantized_onnx_model_path, quantized_pt_model_path):
    """
    Applies dynamic quantization to the YOLOv8 model exported in ONNX format and saves it in .pt format.
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # Export the model to ONNX format
    export_success = model.export(format="onnx")
    onnx_export_path = os.path.splitext(model_path)[0] + '.onnx'

    if not os.path.exists(onnx_export_path):
        print(f"ONNX export failed or ONNX file not found at {onnx_export_path}")
        return
    print(f"YOLOv8 model exported to ONNX at {onnx_export_path}")

    # Save the model in .pt format
    torch.save(model.model.state_dict(), quantized_pt_model_path)
    print(f"Model saved in .pt format at {quantized_pt_model_path}")

    # Apply dynamic quantization using ONNX Runtime
    quantize_dynamic(
        onnx_export_path,
        quantized_onnx_model_path,
        weight_type=QuantType.QInt8
    )
    print(f"Dynamic quantization completed. Quantized model saved at {quantized_onnx_model_path}")

    return quantized_onnx_model_path


def print_size_of_model(model_path, label=""):
    """
    Prints the size of the model file.

    Args:
        model_path (str): Path to the model file.
        label (str): Label for the model (e.g., 'Original', 'Quantized').
    """
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / 1e6
    size_gb = size_bytes / 1e9
    if size_gb >= 1:
        print(f"Model Size ({label}): {size_gb:.2f} GB")
    else:
        print(f"Model Size ({label}): {size_mb:.2f} MB")

def calculate_inference_latency(model_path, device='cpu', img_size=640, batch_size=1, num_runs=100):
    """
    Calculates the average inference latency of the model.

    Args:
        model_path (str): Path to the ONNX model.
        device (str): Device to run inference on ('cpu' or 'gpu').
        img_size (int): Image size for inference.
        batch_size (int): Batch size for inference.
        num_runs (int): Number of runs to average the latency.

    Returns:
        float: Average inference latency in milliseconds.
    """
    # Choose the appropriate providers based on the device and installed packages
    if device == 'gpu':
        providers = ['CUDAExecutionProvider']
    else:
        # Use DNNL or OpenVINO if available, else fall back to CPUExecutionProvider
        available_providers = ort.get_available_providers()
        if 'DnnlExecutionProvider' in available_providers:
            providers = ['CPUExecutionProvider']
        elif 'OpenVINOExecutionProvider' in available_providers:
            providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

    # providers = ['DnnlExecutionProvider', 'CPUExecutionProvider']

    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name

    # Create a dummy image
    dummy_image = np.random.rand(batch_size, 3, img_size, img_size).astype(np.float32)

    # Warm-up
    for _ in range(10):
        session.run(None, {input_name: dummy_image})

    # Measure inference latency
    start_time = time.time()
    for _ in range(num_runs):
        session.run(None, {input_name: dummy_image})
    end_time = time.time()

    total_time = end_time - start_time
    avg_latency_ms = (total_time / num_runs) * 1000  # Convert to milliseconds
    return avg_latency_ms

def evaluate_yolo_model(home_dir, model_path, data_yaml_path, img_size, batch_size, project_name):
    """
    Evaluates the YOLOv8 model on the validation dataset.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    model = YOLO(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Evaluating YOLOv8 model on device: {device}")

    metrics = model.val(
        data=data_yaml_path,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        verbose=True,
        save=True,
        project=os.path.join(home_dir, 'runs', 'detect', project_name, 'val'),
        save_txt=True,
        save_conf=True
    )

    print("Validation completed.")

    # Debugging: Print types and values of metrics
    print(f"Type of metrics.box.p: {type(metrics.box.p)}")
    print(f"metrics.box.p: {metrics.box.p}")
    print(f"Type of metrics.box.r: {type(metrics.box.r)}")
    print(f"metrics.box.r: {metrics.box.r}")
    print(f"Type of metrics.box.map50: {type(metrics.box.map50)}")
    print(f"metrics.box.map50: {metrics.box.map50}")
    print(f"Type of metrics.box.map: {type(metrics.box.map)}")
    print(f"metrics.box.map: {metrics.box.map}")

    # Extract necessary metrics using the helper function
    results = {
        'precision': compute_mean(metrics.box.p),       # Mean precision
        'recall': compute_mean(metrics.box.r),          # Mean recall
        'mAP50': compute_mean(metrics.box.map50),        # mAP@0.5
        'mAP50-95': compute_mean(metrics.box.map),       # mAP@0.5:0.95
    }

    return results

def evaluate_quantized_model(model_path, data_yaml_path, img_size, batch_size):
    """
    Evaluates the quantized ONNX model on the validation dataset.
    
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    # Load the ONNX model
    session = load_onnx_model(model_path)
    
    # Prepare the validation dataset
    val_dataset = load_validation_dataset(data_yaml_path, img_size)
    
    # Initialize metrics
    all_predictions = []
    all_ground_truths = []
    
    # Loop over the validation dataset
    for images, targets in val_dataset:
        # Preprocess images if necessary
        # Perform inference
        outputs = perform_inference(session, images)
        
        # Post-process outputs to get predictions
        predictions = post_process_outputs(outputs)
        
        # Collect predictions and ground truths
        all_predictions.extend(predictions)
        all_ground_truths.extend(targets)
    
    # Compute evaluation metrics
    metrics = compute_metrics(all_predictions, all_ground_truths)
    return metrics



def train_yolo_model(config, home_dir, data_yaml_path, model_version='yolov8m.pt', epochs=5, img_size=640, batch_size=16, project_name='medium-baseline', mixed_precision=False):
    """
    Fine-tunes a YOLOv8 model on a custom dataset and applies quantization based on the configuration.
    """
    model = YOLO(model_version)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_path = os.path.join(home_dir, 'runs', 'detect', project_name)

    print(f"Training YOLOv8 model ({model_version}) on device: {device}")
    print(f"Project path: {project_path}")

    # Train the model
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        save=True,
        save_period=-1,  # Save only the best model
        project=project_path,
        verbose=True,
        plots=True,
        name='train',  # Specify the run name to control the save directory
        amp=mixed_precision  # Enable mixed precision training
    )

    # Get the actual save directory
    best_model_path = os.path.join(model.trainer.save_dir, 'weights', 'best.pt')
    print(f"Best model saved at: {best_model_path}")

    # Check if the file exists
    if not os.path.exists(best_model_path):
        print("Error: Best model file not found. Please check the training process.")
        return

    # Quantize the trained YOLOv8 model based on configuration
    quantization_type = config.get('quantization', 'none').lower()
    quantized_model_path = os.path.join(model.trainer.save_dir, 'weights', 'best_quantized.onnx')
    quantized_pt_model_path = os.path.join(model.trainer.save_dir, 'weights', 'best_quantized.pt')

    if quantization_type == 'static':
        quantization_type_display = 'Static'
        quantized_model_path = os.path.join(project_path, 'quantized_model_static.onnx')
        quantized_model_path = quantize_yolo_model_static(
            best_model_path,
            quantized_model_path,
            config,
            data_yaml_path,
            img_size,
            quantized_pt_model_path
        )
    elif quantization_type == 'dynamic':
        quantization_type_display = 'Dynamic'
        quantized_model_path = os.path.join(project_path, 'quantized_model_dynamic.onnx')
        quantized_model_path = quantize_yolo_model_dynamic(
            best_model_path,
            quantized_model_path,
            quantized_pt_model_path
        )
    else:
        print("Quantization not enabled or unrecognized option.")
        quantization_type_display = 'None'

    # Evaluate YOLOv8 Model
    yolo_metrics = evaluate_yolo_model(
        home_dir=home_dir,
        model_path=best_model_path,
        data_yaml_path=data_yaml_path,
        img_size=img_size,
        batch_size=batch_size,
        project_name=project_name
    )

    # Extract Accuracy and Precision
    accuracy = evaluate_accuracy(yolo_metrics)
    precision = evaluate_precision(yolo_metrics)

    # Calculate Model Sizes
    original_model_path = best_model_path
    print_size_of_model(original_model_path, label="Original")

    if quantized_model_path and os.path.exists(quantized_model_path):
        print_size_of_model(quantized_model_path, label=quantization_type_display)
    else:
        print("Quantized model not available.")

        
    # Evaluate the quantized model if available
    if quantized_model_path and os.path.exists(quantized_model_path):
        # Load the quantized model with Ultralytics YOLO
        quantized_model = YOLO(quantized_model_path)
        quantized_metrics = quantized_model.val(
            data=data_yaml_path,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            verbose=True,
            save=True,
            project=os.path.join(home_dir, 'runs', 'detect', project_name, 'val_quantized'),
            save_txt=True,
            save_conf=True
        )

        # Extract metrics
        quantized_yolo_metrics = {
            'precision': compute_mean(quantized_metrics.box.p),
            'recall': compute_mean(quantized_metrics.box.r),
            'mAP50': compute_mean(quantized_metrics.box.map50),
            'mAP50-95': compute_mean(quantized_metrics.box.map),
        }

        # Extract Accuracy and Precision
        quantized_accuracy = evaluate_accuracy(quantized_yolo_metrics)
        quantized_precision = evaluate_precision(quantized_yolo_metrics)
    else:
        quantized_yolo_metrics = None
        quantized_accuracy = None
        quantized_precision = None
        print("Quantized model evaluation skipped due to missing model.")
    


    print('##########################################################')
    # Measure Inference Latency for Original Model
    original_onnx_model_path = os.path.splitext(original_model_path)[0] + '.onnx'
    if os.path.exists(original_onnx_model_path):
        original_inference_latency = calculate_inference_latency(
            original_onnx_model_path,
            device='cpu',
            img_size=img_size,
            batch_size=1,
            num_runs=100
        )
        print(f"Inference Latency (Original): {original_inference_latency:.2f} ms")
    else:
        print(f"Original ONNX model not found at {original_onnx_model_path}. Exporting model to ONNX.")
        model = YOLO(original_model_path)

        export_dir = os.path.dirname(original_onnx_model_path)
        model_name = os.path.basename(original_onnx_model_path)
        model.export(format='onnx', save_dir=export_dir, name=model_name)


        original_inference_latency = calculate_inference_latency(
            original_onnx_model_path,
            device='cpu',
            img_size=img_size,
            batch_size=1,
            num_runs=100
        )
        print(f"Inference Latency (Original): {original_inference_latency:.2f} ms")

    # Measure Inference Latency for Quantized Model
    if quantized_model_path and os.path.exists(quantized_model_path):
        quantized_inference_latency = calculate_inference_latency(
            quantized_model_path,
            device='cpu',
            img_size=img_size,
            batch_size=1,
            num_runs=100
        )
        print(f"Inference Latency ({quantization_type_display} Quantized): {quantized_inference_latency:.2f} ms")
    else:
        quantized_inference_latency = None
        print("Quantized model not found. Skipping latency calculation.")

    # Save Evaluation Metrics and Calculations
    results = {
        'YOLOv8': {
            'metrics': yolo_metrics,
            'accuracy_percentage': accuracy,
            'precision_percentage': precision,
            'model_size_MB': os.path.getsize(original_model_path) / 1e6,
            'quantized_model_size_MB': os.path.getsize(quantized_model_path) / 1e6 if quantized_model_path and os.path.exists(quantized_model_path) else None,
            'inference_latency_original_ms': original_inference_latency,
            'inference_latency_quantized_ms': quantized_inference_latency,
            'quantized_model_path': quantized_model_path,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'quantization_type': quantization_type_display
        }
    }

    # Save results to JSON
    results_path = os.path.join(home_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")

def main_training_pipeline(config):
    """
    Orchestrates the training, quantization, and evaluation pipeline.

    Args:
        config (dict): Configuration dictionary.
    """
    home_dir = os.getcwd()
    data_yaml_path = os.path.join(home_dir, 'data.yaml')

    yolo_project_name = 'medium-baseline'

    # Train YOLOv8 Model
    train_yolo_model(
        config=config,
        home_dir=home_dir,
        data_yaml_path=data_yaml_path,
        model_version=config.get('yolo_model_version', 'yolov8m.pt'),
        epochs=config.get('yolo_epochs', 1),
        img_size=config.get('yolo_img_size', 32),
        batch_size=config.get('yolo_batch_size', 1024),
        project_name=yolo_project_name,
        mixed_precision=False  # Enable mixed precision
    )

if __name__ == "__main__":
    # Example configuration dictionary
    CONFIG = {
        'yolo_model_version': 'yolov8m.pt',  # Replace with desired YOLOv8 model
        'yolo_epochs': 5,  # Reduced for quick testing
        'yolo_img_size': 256,
        'yolo_batch_size': 32,
        'quantization': 'dynamic',  # Options: 'dynamic', 'static',  None
    }

    print(f"Using configuration: {CONFIG}")

    main_training_pipeline(CONFIG)
