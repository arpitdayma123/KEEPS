# Start with a Python base image.
# Using a specific version like 3.10 is good practice.
# Consider a slim variant for smaller image size.
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install ffmpeg and other system dependencies
# -y assumes yes to all prompts, useful for Docker builds
RUN apt-get update && apt-get install -y ffmpeg libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces image size
# Consider upgrading pip first
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Command to run the RunPod worker
# -u for unbuffered Python output, useful for logging
# --rp_fast_init is a RunPod option that might be beneficial
# Ensure handler.py is executable or called via python -m
CMD ["python", "-u", "-m", "runpod.serverless.worker", "handler.py", "--rp_fast_init"]
