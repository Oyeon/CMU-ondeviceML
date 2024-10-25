CONFIG = {
    # YOLOv8 Training Configurations
    'enable_yolo': True,  # Set to True to enable YOLOv8 training and evaluation
    'yolo_model_version': 'yolov8m.pt',  # Pre-trained YOLOv8 model to use (e.g., 'yolov8n.pt', 'yolov8s.pt', etc.)
    'yolo_epochs': 1,  # Number of training epochs
    'yolo_img_size': 640,  # Image size for training and evaluation
    'yolo_batch_size': 32,  # Batch size for training and evaluation

    # Quantization Settings
    # Currently using dynamic quantization via ONNX Runtime
    # For static or QAT, additional implementation is required
}
