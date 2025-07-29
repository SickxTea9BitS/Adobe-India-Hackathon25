FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# CRITICAL: Download ALL EasyOCR models during build (single line approach)
RUN python -c "import easyocr; import numpy as np; import time; from pathlib import Path; print('=== DOWNLOADING ALL EASYOCR MODELS ==='); model_dir = Path.home() / '.EasyOCR'; model_dir.mkdir(parents=True, exist_ok=True); reader = easyocr.Reader(['en'], gpu=False, verbose=True, download_enabled=True); test_img = np.random.randint(0, 255, (300, 600, 3), dtype=np.uint8); result = reader.readtext(test_img, min_size=10, text_threshold=0.7); print(f'All EasyOCR models downloaded and verified! Found {len(result)} text regions.')"

# Create target model directory
RUN mkdir -p /model/easyocr

# Copy ALL EasyOCR models to container location (single line)
RUN python -c "import shutil; from pathlib import Path; source_dir = Path.home() / '.EasyOCR'; target_dir = Path('/model/easyocr'); print(f'Copying models from {source_dir} to {target_dir}'); shutil.copytree(str(source_dir), str(target_dir), dirs_exist_ok=True) if source_dir.exists() else print('Source directory not found'); copied_files = list(target_dir.rglob('*')) if target_dir.exists() else []; print(f'Successfully copied {len(copied_files)} files to {target_dir}')"

# Copy your YOLO model
COPY app/model/yolov11x_best.pt /model/yolov11x_best.pt

# Copy the main application script
COPY app/extract_outline_docker.py /app/extract_outline_docker.py

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Final verification of all models
RUN echo "=== FINAL MODEL VERIFICATION ===" && \
    echo "YOLO model:" && ls -lh /model/yolov11x_best.pt && \
    echo "EasyOCR models:" && find /model/easyocr -name "*.pth" -exec ls -lh {} \; | head -5 && \
    echo "Total EasyOCR files:" && find /model/easyocr -type f | wc -l && \
    echo "Model verification complete"

# Set environment variables for offline operation
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=""
ENV YOLO_CONFIG_DIR=/tmp
ENV ULTRALYTICS_OFFLINE=1

# Run the application
CMD ["python", "/app/extract_outline_docker.py"]
