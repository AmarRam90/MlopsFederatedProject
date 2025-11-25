FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directories for data and models if they don't exist
RUN mkdir -p data models

# Default command (can be overridden)
CMD ["python", "server.py"]
