# Use an official PyTorch base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /workspace

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

COPY pytorch3dunet /workspace/pytorch3dunet
COPY setup.py  /workspace/
# Install pytorch3dunet in editable mode
WORKDIR /workspace/
RUN pip install -e .

# Copy the source code into the container
COPY . /workspace/

# Set environment variables
ENV PYTHONPATH=/workspace

# Expose the port (if needed for web-based visualization)
EXPOSE 5000

# Command to run the predictor
CMD ["python3", "main.py"]
