# Stage 1: Base Image with PyTorch and CUDA support
FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3 AS base

# Stage 2: Final Image
FROM base

# Set the working directory
WORKDIR /app

# Copy only necessary files
COPY inference_onnx.py jetson_model.onnx /app/

# Install runtime dependencies and cleanup in a single RUN statement
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && pip install --no-cache-dir onnx onnxruntime numpy --ignore-installed \
    && apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip

# Set environment variables for NVIDIA runtime
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Define the entrypoint to run the inference script
ENTRYPOINT ["python3", "inference_onnx.py"]


################################################################################


# # Stage 1: Base Image with PyTorch and CUDA support
# FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3 AS base

# # Stage 2: Final Image
# FROM base

# # Set the working directory
# WORKDIR /app

# # Copy only necessary files
# COPY inference_onnx.py jetson_model.onnx /app/

# # Install runtime dependencies and cleanup in a single RUN statement
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         libglib2.0-0 \
#         libsm6 \
#         libxext6 \
#         libxrender-dev \
#     && pip install --no-cache-dir onnx onnxruntime numpy \
#     && apt-get purge -y --auto-remove \
#     && rm -rf /var/lib/apt/lists/* /root/.cache/pip

# # Set environment variables for NVIDIA runtime
# ENV NVIDIA_VISIBLE_DEVICES=all \
#     NVIDIA_DRIVER_CAPABILITIES=compute,utility

# # Define the entrypoint to run the inference script
# ENTRYPOINT ["python3", "inference_onnx.py"]
