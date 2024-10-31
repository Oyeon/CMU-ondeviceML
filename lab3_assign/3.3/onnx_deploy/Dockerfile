# Base image (Jetson Nano L4T base image)
FROM nvcr.io/nvidia/l4t-base:r32.5.0

# Set working directory
WORKDIR /workspace

# Install required packages
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3-setuptools \
    build-essential \
    git \
    wget \
    cmake \
    libprotobuf-dev \
    protobuf-compiler \
    libssl-dev \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Update alternatives to use python3.8 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Python packages
RUN pip3 install \
    numpy \
    pillow \
    onnx==1.9.0

# Install ONNX Runtime for Jetson
RUN pip3 install onnxruntime \
    --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46

# Set environment variables
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

# Copy models, code, and sample data
COPY dynamic_quantized_best.onnx /workspace/
COPY FP16_best.onnx /workspace/
COPY default.onnx /workspace/
COPY inference_code.py /workspace/
COPY datasets /workspace/datasets

# Set command to run inference script
CMD ["python3", "inference_code.py"]





















