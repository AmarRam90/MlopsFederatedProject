# Kubernetes Deployment Guide

This directory contains the Kubernetes manifests to deploy the Federated Learning application.

## Prerequisites
- **Docker**: To build the application image.
- **Kubernetes Cluster**: A local cluster like Docker Desktop (with Kubernetes enabled) or Minikube.
- **kubectl**: Command-line tool for interacting with the cluster.

## Deployment Steps

### 1. Build the Docker Image
First, build the Docker image locally so that Kubernetes can use it.
```bash
docker build -t fl-app:latest .
```

### 2. Apply the Manifests
Apply the configuration files in the following order:

1.  **Storage (PV & PVC)**:
    This sets up the shared storage required for the server and nodes to communicate.
    ```bash
    kubectl apply -f k8s/pv-pvc.yaml
    ```
    *Note: If you are not using Docker Desktop on Windows, you may need to adjust the `hostPath` in `k8s/pv-pvc.yaml` to a valid path on your machine.*

2.  **Server**:
    Deploy the central server.
    ```bash
    kubectl apply -f k8s/server-deployment.yaml
    ```

3.  **Nodes**:
    Deploy the three federated learning nodes.
    ```bash
    kubectl apply -f k8s/nodes.yaml
    ```

4.  **Service**:
    Expose the server so you can access the UI.
    ```bash
    kubectl apply -f k8s/service.yaml
    ```

### 3. Access the Application
The application is exposed via a `NodePort` service.

- **Gradio UI**: `http://localhost:30000`
- **Prometheus Metrics**: `http://localhost:30001`

### 4. Verify Deployment
Check the status of your pods:
```bash
kubectl get pods
```
You should see 1 server pod and 3 node pods in the `Running` state.

### 5. Cleanup
To remove the deployment:
```bash
kubectl delete -f k8s/
```
