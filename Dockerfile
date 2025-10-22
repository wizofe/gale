# Use an official CUDA runtime as a parent image
FROM nvidia/cuda:12.3.0-runtime-ubuntu20.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and required packages
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv git && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Upgrade pip and install dependencies
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Expose port (if required, e.g. for notebooks)
EXPOSE 8888

# Define default command
CMD ["python3", "src/experiments/run_experiments.py"]
