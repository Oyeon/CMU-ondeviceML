# Use NVIDIA's Jetson-compatible base image with Python 3 and PyTorch
FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3

# Set the working directory
WORKDIR /app

# Copy only necessary files
COPY inference_onnx.py model_20241005_204235_epoch_2.onnx /app/

# Install runtime dependencies with pip and clean up APT cache to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev && \
    pip install --no-cache-dir onnx onnxruntime numpy && \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables for NVIDIA runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Define the entrypoint to run the inference script
ENTRYPOINT ["python3", "inference_onnx.py"]