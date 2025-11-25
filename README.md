# Federated Learning for Mood Prediction

A robust Federated Learning (FL) system designed to predict user mood based on health metrics (sleep, steps, heart rate, etc.) while preserving data privacy. This project demonstrates a complete FL pipeline with a central aggregator, multiple training nodes, and an advanced analytics dashboard.

## 🚀 Key Features

*   **Federated Averaging**: Implements the FedAvg algorithm to aggregate model updates from multiple nodes without sharing raw data.
*   **Selective Aggregation**: Intelligent aggregator that evaluates node models on a **Global Evaluation Dataset** and only includes high-performing models (Accuracy > 0.3) in the global update.
*   **Real-time Analytics Dashboard**: Built with **Gradio**, offering:
    *   **Live Metrics**: Global Accuracy, Loss, F1 Score, Precision, and Recall.
    *   **Visualizations**: Time-series plots for all metrics and node performance.
    *   **Confusion Matrix**: Heatmap to visualize classification performance.
    *   **Data Injection**: Interface to simulate new data arriving at specific nodes.
    *   **Inference**: Real-time mood prediction using the global model.
*   **Data Privacy**: Raw data remains on local nodes (`node_X.csv`); only model weights are shared.
*   **Automated Orchestration**: Background scheduler runs FL cycles automatically every 15 seconds.

## 📂 Project Structure

```
federated-ops-project/
├── data/                   # Data storage
│   ├── global_eval.csv     # Global dataset for validation
│   ├── node_1.csv          # Local data for Node 1
│   ├── node_2.csv          # Local data for Node 2
│   └── node_3.csv          # Local data for Node 3
├── models/                 # Model checkpoints
│   ├── global_model.pth    # Aggregated global model
│   └── node_*.pth          # Local node models
├── src/                    # Source code
│   ├── aggregator.py       # Federated Averaging logic
│   ├── data_manager.py     # Data loading & preprocessing
│   ├── model.py            # PyTorch Neural Network architecture
│   └── node_trainer.py     # Local training loop
├── main.py                 # Entry point: Gradio UI & FL Orchestrator
├── resplit_data_v2.py      # Script to initialize/reset data splits
└── requirements.txt        # Python dependencies
```

## 🛠️ Setup & Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd federated-ops-project
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Initialize Data**:
    If `data/` is empty, run the split script to generate node datasets from raw data:
    ```bash
    python resplit_data_v2.py
    ```
    *(Note: This requires `raw_data_upload.csv` to be present in the root)*

## 🏃‍♂️ Usage

1.  **Start the Application**:
    ```bash
    python main.py
    ```

2.  **Access the Dashboard**:
    Open your browser and navigate to: `http://localhost:7860`

3.  **Monitor Training**:
    *   The system will automatically start training cycles in the background.
    *   Watch the **Dashboard** tab for real-time graphs of Global Accuracy, F1 Score, and Node Performance.
    *   Check the **Confusion Matrix** to see how well the model distinguishes between 'Neutral', 'Happy', and 'Sad'.

4.  **Interact**:
    *   **Data Injection**: Go to the "Data Injection" tab to add new synthetic health data to a specific node and observe how it impacts the next training cycle.
    *   **Prediction**: Use the "Prediction" tab to input health metrics and get a mood prediction from the current Global Model.

## 🧠 Technical Details

*   **Model**: Simple Feed-Forward Neural Network (PyTorch).
*   **Input Features**: Sleep Time, Wakeup Time, Breakfast Type, Meditation, RHR, Steps, HRV, Temperature, Humidity.
*   **Target**: Mood (Neutral, Happy, Sad).
*   **Aggregation Strategy**: Weighted average of model weights from eligible nodes (Accuracy > Threshold).

## 📊 Metrics Explained

*   **Global Accuracy**: Overall correctness of the global model on the held-out evaluation set.
*   **F1 Score**: Harmonic mean of precision and recall, useful for imbalanced classes.
*   **Node Accuracy**: How well each individual node's model performs on the global evaluation set (used for contribution quality check).
