# Step-by-Step Guide: Docker, Orchestration, and CI/CD

This guide explains how to containerize, orchestrate, and automate the Federated Learning project.

## 1. Dockerization (Containerization)

**Goal**: Package the application and its dependencies into a standard unit (container) that runs consistently everywhere.

**Step 1: Create a `Dockerfile`**
The `Dockerfile` is a recipe for building your image.
```dockerfile
# Use a lightweight Python base image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create necessary directories
RUN mkdir -p data models

# Default command (can be overridden)
CMD ["python", "server.py"]
```

**Step 2: Build the Image**
```bash
docker build -t fl-project .
```

## 2. Orchestration (Docker Compose)

**Goal**: Manage multiple containers (Server, Nodes, Monitoring) as a single system.

**Step 1: Create `docker-compose.yml`**
This file defines the services, networks, and volumes.

*   **Services**:
    *   `server`: Runs `server.py`. Exposes UI on port 7860.
    *   `node-1, 2, 3`: Run `node.py`.
    *   `prometheus`: Collects metrics.
    *   `grafana`: Visualizes metrics.
*   **Volumes**:
    *   `./data:/app/data`: Maps the local `data` folder to the container's `/app/data`. This allows all containers to share files (simulating a network file system).

**Step 2: Run the System**
```bash
docker-compose up --build
```
*   `--build`: Rebuilds images before starting.
*   `-d`: Runs in detached mode (background).

## 3. CI/CD (GitHub Actions)

**Goal**: Automate testing and building whenever code changes.

**Step 1: Create Workflow File**
Create `.github/workflows/ci.yml`.

**Step 2: Define Steps**
*   **Trigger**: `on: push` to `main`.
*   **Jobs**:
    *   `Checkout`: Get the code.
    *   `Setup Python`: Install Python.
    *   `Install Dependencies`: `pip install -r requirements.txt`.
    *   `Build Docker`: Verify the `Dockerfile` builds successfully.

## 4. Monitoring (Prometheus & Grafana)

**Goal**: Track system performance.

*   **Prometheus**: Scrapes metrics from `http://<service>:8000`. Configured in `prometheus.yml`.
*   **Grafana**: Connects to Prometheus to display graphs. Access at `http://localhost:3000`.

### How to use Grafana
1.  Login: `admin` / `admin`.
2.  Add Data Source: Select **Prometheus**. URL: `http://prometheus:9090`.
3.  Create Dashboard: Add panels querying metrics like `fl_global_accuracy` or `fl_node_training_loss`.
