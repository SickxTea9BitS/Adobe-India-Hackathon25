# Dockerfile

# 1. Use an official Python runtime as a parent image
FROM python:3.11-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Create mount points for input and output
#    (Docker creates these automatically on mount, but this is good practice)
RUN mkdir -p /app/input /app/output

# 4. Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code into the container
COPY . .

# 6. Set the entrypoint to run your main script
#    This allows you to pass command-line arguments directly to the script
ENTRYPOINT ["python", "main.py"]