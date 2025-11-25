import time
import schedule
import os
import sys
from prometheus_client import start_http_server, Gauge
from src.node_trainer import NodeTrainer

# Prometheus Metrics
FL_NODE_TRAINING_LOSS = Gauge('fl_node_training_loss', 'Last training loss', ['node_id'])
FL_NODE_SAMPLES_COUNT = Gauge('fl_node_samples_count', 'Total samples trained on', ['node_id'])

# Start Prometheus server
start_http_server(8000)

# Configuration
NODE_ID = int(os.environ.get('NODE_ID', 1))
DATA_DIR = 'data'
MODELS_DIR = 'models'
NODE_MODEL_PATH = os.path.join(MODELS_DIR, f'node_{NODE_ID}.pth')
DATA_PATH = os.path.join(DATA_DIR, f'node_{NODE_ID}.csv')

print(f"Starting Node {NODE_ID}...")
print(f"Data Path: {DATA_PATH}")
print(f"Model Path: {NODE_MODEL_PATH}")

trainer = NodeTrainer(NODE_ID, DATA_PATH, NODE_MODEL_PATH)

def train_job():
    print(f"[Node {NODE_ID}] Checking for new data...")
    res = trainer.train()
    if res:
        print(f"[Node {NODE_ID}] Training completed and model updated.")
        # Note: In a real scenario, we'd get the loss from the trainer. 
        # For now, we'll just increment a sample counter as a proxy for activity.
        FL_NODE_SAMPLES_COUNT.labels(node_id=NODE_ID).inc(10) # Assuming batch size or similar
    else:
        print(f"[Node {NODE_ID}] No sufficient new data to train.")

# Schedule training every 10 seconds (check for new data)
schedule.every(10).seconds.do(train_job)

if __name__ == "__main__":
    # Run once immediately
    train_job()
    
    while True:
        schedule.run_pending()
        time.sleep(1)
