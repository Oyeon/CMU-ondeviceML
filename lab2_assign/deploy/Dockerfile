# Use NVIDIA's optimized base image for Jetson with PyTorch and CUDA support
FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Install runtime dependencies including OpenCV via apt and Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        vim \  # Install vim
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip is installed for Python 3.10
RUN python3.10 -m ensurepip --upgrade

# Upgrade pip to the latest version using Python 3.10's pip
RUN python3.10 -m pip install --upgrade pip

# Configure update-alternatives for python3 and pip3 without creating symlinks
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3.10 1

# Install ultralytics without its dependencies to avoid conflicts with system-installed OpenCV
RUN pip3 install --no-cache-dir ultralytics --no-deps

# Manually install ultralytics' dependencies except opencv-python to prevent conflicts
RUN pip3 install --no-cache-dir \
        numpy \
        torch \
        torchvision \
        tqdm \
        PyYAML \
        ultralytics

# Copy the inference script, model, and configuration files
COPY inference_jetson.py data.yaml best.pt datasets ./

# Set environment variables for NVIDIA runtime
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Define the entrypoint to run the inference script
ENTRYPOINT ["python3", "inference_jetson.py"]
