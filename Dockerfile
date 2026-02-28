# Dockerfile

# Base image - Python 3.11 on slim Debian Linux
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required by PyMuPDF and FAISS
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first - Docker caches this layer
# If requirements.txt doesn't change, Docker won't reinstall packages
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create data directories
RUN mkdir -p data/raw data/index

# Expose the port FastAPI runs on
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]