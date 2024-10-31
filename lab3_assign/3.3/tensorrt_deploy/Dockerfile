# Stage 1: Base Image with PyTorch and CUDA support
FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3 AS base

# Stage 2: Final Image
FROM base

# Set the working directory
WORKDIR /workspace

# Install runtime dependencies and cleanup in a single RUN statement
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && pip install --no-cache-dir onnx onnxruntime numpy --ignore-installed \
    && apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip

# Set environment variables for CUDA
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda

# Copy models, code, and sample data to the container
COPY inference_code.py /workspace/
COPY datasets /workspace/datasets
COPY best.pt /workspace/
COPY convert2tensorrt.py /workspace/

# Set NVIDIA runtime capabilities
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Command to run inference script
CMD ["python3", "convert2tensorrt.py"]




# # Base image (Jetson Nano L4T base image)
# FROM nvcr.io/nvidia/l4t-base:r32.5.0

# # Set working directory
# WORKDIR /workspace

# # Install necessary certificates and gnupg
# RUN apt-get update && apt-get install -y \
#     gnupg \
#     ca-certificates

# # Add NVIDIA's package repositories and keys
# RUN apt-key adv --fetch-keys https://repo.download.nvidia.com/jetson/jetson-ota-public.asc && \
#     echo "deb https://repo.download.nvidia.com/jetson/common r32.5 main" > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list && \
#     echo "deb https://repo.download.nvidia.com/jetson/t210 r32.5 main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list

# # Continue with the remaining installation steps
# RUN apt-get update && apt-get install -y \
#     python3.8 \
#     python3.8-dev \
#     python3-pip \
#     python3-setuptools \
#     build-essential \
#     git \
#     wget \
#     cmake \
#     libprotobuf-dev \
#     protobuf-compiler \
#     libssl-dev \
#     libcurl4-openssl-dev \
#     python3-libnvinfer-dev \
#     python3-libnvinfer \
#     && rm -rf /var/lib/apt/lists/*

# # Update alternatives to use python3.8 as default
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# # Upgrade pip
# RUN python3 -m pip install --upgrade pip

# # Install Python packages
# RUN pip3 install \
#     numpy \
#     pillow \
#     onnx==1.9.0 \
#     pycuda \
#     ultralytics


# # Set environment variables
# ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

# COPY inference_code.py /workspace/
# COPY datasets /workspace/datasets

# COPY best.pt /workspace/
# COPY convert2tensorrt.py /workspace/

# # Set the NVIDIA runtime
# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# # Set command to run inference script
# # RUN python3 convert2tensorrt.py

# # CMD ["python3", "inference_code.py"]
# CMD ["python3", "convert2tensorrt.py"]

