import gradio as gr
import threading
import time
import schedule
import pandas as pd
import torch
import os
from src.aggregator import Aggregator
from src.data_manager import DataManager
from src.model import SimpleNN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from prometheus_client import start_http_server, Gauge

# Prometheus Metrics
FL_GLOBAL_ACCURACY = Gauge('fl_global_accuracy', 'Current accuracy of the global model')
FL_GLOBAL_LOSS = Gauge('fl_global_loss', 'Current loss of the global model')
FL_ROUND_COUNT = Gauge('fl_round_count', 'Number of aggregation rounds completed')

# Start Prometheus server
start_http_server(8000)

# Configuration
NODES = [1, 2, 3]
DATA_DIR = 'data'
MODELS_DIR = 'models'
GLOBAL_MODEL_PATH = os.path.join(MODELS_DIR, 'global_model.pth')
NODE_MODEL_PATHS = [os.path.join(MODELS_DIR, f'node_{i}.pth') for i in NODES]

# State
history = {
    'timestamp': [], 
    'accuracy': [], 
    'loss': [],
    'f1': [],
    'precision': [],
    'recall': [],
    'confusion_matrix': [],
    'node_local_acc': {1: [], 2: [], 3: []}, # Node model on Local Test
    'node_global_acc': {1: [], 2: [], 3: []}, # Node model on Global Test
    'global_on_local_acc': {1: [], 2: [], 3: []} # Global model on Local Test
}

aggregator = Aggregator(GLOBAL_MODEL_PATH, NODE_MODEL_PATHS)

# Initialize global model if not exists
aggregator.initialize_global_model()

def evaluate_model(model, data_path=None, dm=None):
    if dm is None:
        if data_path is None:
            return 0, 0
        dm = DataManager(data_path=data_path)
        df = dm.load_data()
        features, labels = dm.preprocess(df)
    else:
        # Use existing DM (e.g. for test set)
        dataset = dm.get_test_dataset()
        if dataset is None:
            return 0, 0
        features = dataset.features
        labels = dataset.labels

    if features is None or len(features) == 0:
        return 0, 0

    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        outputs = model(features)
        loss = criterion(outputs, labels).item()
        _, predicted = torch.max(outputs.data, 1)
        
        y_true = labels.numpy()
        y_pred = predicted.numpy()
        
        accuracy = (y_pred == y_true).mean()
        
    return accuracy, loss

def evaluate_global_model(model=None):
    eval_path = os.path.join(DATA_DIR, 'global_eval.csv')
    
    if model is None:
        model = SimpleNN()
        if os.path.exists(GLOBAL_MODEL_PATH):
            model.load_state_dict(torch.load(GLOBAL_MODEL_PATH))
    
    acc, loss = evaluate_model(model, data_path=eval_path)
    
    # Calculate advanced metrics for global model only
    # Re-running inference to get preds (could optimize but keeping simple)
    dm = DataManager(data_path=eval_path)
    df = dm.load_data()
    features, labels = dm.preprocess(df)
    
    if features is None:
        return acc, loss, 0, 0, 0, None
        
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        label_idx = predicted.item() if len(predicted) == 1 else 0 # Handle single item
        
        y_true = labels.numpy()
        y_pred = predicted.numpy()
        
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        
    return acc, loss, f1, precision, recall, cm

def fl_job():
    print("Running FL Cycle (Aggregation)...")
    
    # In decoupled mode, we don't trigger training directly.
    # We check if nodes have updated their models.
    # For simplicity in this "file-based" decoupled approach, we will just attempt aggregation.
    # The nodes run their own loops and update node_X.pth.
    
    # 2. Evaluate Node Models on Global Eval (for Selective Aggregation & Logging)
    node_global_accs = {}
    for i in NODES:
        node_id = i
        # Load node model
        node_model = SimpleNN()
        node_model_path = NODE_MODEL_PATHS[i-1]
        try:
            if os.path.exists(node_model_path):
                node_model.load_state_dict(torch.load(node_model_path))
                acc, _ = evaluate_model(node_model, data_path=os.path.join(DATA_DIR, 'global_eval.csv'))
                node_global_accs[node_id] = acc
            else:
                 node_global_accs[node_id] = 0.0
        except Exception as e:
            print(f"Error evaluating node {node_id}: {e}")
            node_global_accs[node_id] = 0.0

    # 3. Aggregate with Selective Logic (Eval Node Models on Global Test)
    def eval_func(model):
        return evaluate_global_model(model)[0:2] # Return acc, loss
        
    success, _ = aggregator.aggregate(eval_func)
    
    if success:
        # 4. Eval Global Model on Global Test
        acc, loss, f1, prec, rec, cm = evaluate_global_model()
        
        # Update History
        timestamp = time.strftime("%H:%M:%S")
        history['timestamp'].append(timestamp)
        history['accuracy'].append(acc)
        history['loss'].append(loss)
        
        # Update Prometheus Metrics
        FL_GLOBAL_ACCURACY.set(acc)
        FL_GLOBAL_LOSS.set(loss)
        FL_ROUND_COUNT.inc()
        history['f1'].append(f1)
        history['precision'].append(prec)
        history['recall'].append(rec)
        history['confusion_matrix'] = cm
        
        for i in NODES:
            # For now, "local acc" is same as "global acc" since we only use global eval
            history['node_local_acc'][i].append(node_global_accs.get(i, 0)) 
            history['node_global_acc'][i].append(node_global_accs.get(i, 0))
            history['global_on_local_acc'][i].append(0) # Deprecated
        
        print(f"Cycle Complete. Global Acc: {acc:.4f}")
    else:
        print("Aggregation skipped (no improvements or no models).")

def run_scheduler():
    schedule.every(15).seconds.do(fl_job)
    while True:
        schedule.run_pending()
        time.sleep(1)

# Start background thread
thread = threading.Thread(target=run_scheduler, daemon=True)
thread.start()

# --- Gradio UI ---

def get_metrics_table():
    if not history['timestamp']:
        return pd.DataFrame()
    
    # Create DataFrame from history
    data = {
        'Timestamp': history['timestamp'],
        'Global Accuracy': history['accuracy'],
        'Loss': history['loss'],
        'F1 Score': history['f1'],
        'Precision': history['precision'],
        'Recall': history['recall']
    }
    return pd.DataFrame(data)

def get_metrics_plot():
    if not history['timestamp']:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Accuracy & Loss
    ax = axes[0, 0]
    ax.plot(history['timestamp'], history['accuracy'], marker='o', label='Accuracy', color='blue')
    ax.plot(history['timestamp'], history['loss'], marker='x', label='Loss', color='red', linestyle='--')
    ax.set_title("Global Accuracy & Loss")
    ax.legend()
    ax.tick_params(axis='x', rotation=45)

    # Plot 2: F1 Score
    ax = axes[0, 1]
    ax.plot(history['timestamp'], history['f1'], marker='s', label='F1 Score', color='green')
    ax.set_title("Global F1 Score")
    ax.legend()
    ax.tick_params(axis='x', rotation=45)

    # Plot 3: Precision
    ax = axes[1, 0]
    ax.plot(history['timestamp'], history['precision'], marker='^', label='Precision', color='purple')
    ax.set_title("Global Precision")
    ax.legend()
    ax.tick_params(axis='x', rotation=45)

    # Plot 4: Recall
    ax = axes[1, 1]
    ax.plot(history['timestamp'], history['recall'], marker='v', label='Recall', color='orange')
    ax.set_title("Global Recall")
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    
    fig.tight_layout()
    return fig

def get_node_metrics_plot():
    if not history['timestamp']:
        return None
        
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in NODES:
        ax.plot(history['timestamp'], history['node_global_acc'][i], label=f'Node {i}')
    ax.axhline(y=0.3, color='r', linestyle='--', label='Threshold (0.3)')
    ax.set_title("Node Model Accuracy (on Global Eval)")
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    return fig

def get_confusion_matrix_plot():
    if history['confusion_matrix'] is None or len(history['confusion_matrix']) == 0:
        return None
        
    cm = history['confusion_matrix']
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Neutral', 'Happy', 'Sad'], 
                yticklabels=['Neutral', 'Happy', 'Sad'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"Confusion Matrix (Last Update)")
    return fig

def get_latest_metrics():
    if not history['timestamp']:
        return "No data yet."
    
    last_ts = history['timestamp'][-1]
    
    # Format detailed metrics
    report = f"### Last Update: {last_ts}\n\n"
    report += f"**Global Model (Global Test)**:\n"
    report += f"- Accuracy: {history['accuracy'][-1]:.4f}\n"
    report += f"- F1 Score: {history['f1'][-1]:.4f}\n"
    report += f"- Precision: {history['precision'][-1]:.4f}\n"
    report += f"- Recall: {history['recall'][-1]:.4f}\n"
    
    report += "\n**Node Contribution (Global Test)**:\n"
    for i in NODES:
        report += f"- Node {i}: {history['node_global_acc'][i][-1]:.4f}\n"
        
    return report

def inject_data(node_id, sleep, wakeup, breakfast, meditation, rhr, steps, hrv, temp, hum, mood):
    # Create dict
    data = {
        'Sleep_Time': sleep,
        'Wakeup_Time': wakeup,
        'Breakfast_Type': breakfast,
        'Meditation_Exercise_Minutes': meditation,
        'Resting_Heart_Rate': rhr,
        'Step_Count': steps,
        'HRV': hrv,
        'Temperature_Celsius': temp,
        'Humidity_Percent': hum,
        'Mood': mood
    }
    
    # Append to TRAIN data
    path = os.path.join(DATA_DIR, f'node_{node_id}.csv')
    dm = DataManager(node_id=node_id, data_path=path)
    dm.append_data(data)
    return f"Added data to Node {node_id}"

def predict_mood(sleep, wakeup, breakfast, meditation, rhr, steps, hrv, temp, hum):
    # Create dummy df for preprocessing
    data = {
        'Sleep_Time': [sleep],
        'Wakeup_Time': [wakeup],
        'Breakfast_Type': [breakfast],
        'Meditation_Exercise_Minutes': [meditation],
        'Resting_Heart_Rate': [rhr],
        'Step_Count': [steps],
        'HRV': [hrv],
        'Temperature_Celsius': [temp],
        'Humidity_Percent': [hum],
        'Mood': ['Neutral'] # Dummy
    }
    df = pd.DataFrame(data)
    dm = DataManager()
    features, _ = dm.preprocess(df)
    
    # Load model
    model = SimpleNN()
    if os.path.exists(GLOBAL_MODEL_PATH):
        model.load_state_dict(torch.load(GLOBAL_MODEL_PATH))
    model.eval()
    
    with torch.no_grad():
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        label_idx = predicted.item() if len(predicted) == 1 else 0 # Handle single item
        
    return dm.mood_cats[label_idx]

with gr.Blocks(title="Federated Learning Dashboard") as demo:
    gr.Markdown("# Federated Learning Operations Dashboard (Advanced)")
    
    with gr.Tab("Dashboard"):
        gr.Markdown("### Real-time Analytics")
        with gr.Row():
            with gr.Column(scale=1):
                metrics_text = gr.Markdown("Waiting for first update...")
                refresh_btn = gr.Button("Refresh Metrics")
            with gr.Column(scale=1):
                cm_plot = gr.Plot(label="Confusion Matrix")
        
        gr.Markdown("### Historical Metrics (Tabular)")
        metrics_table = gr.Dataframe(label="Metrics History", headers=['Timestamp', 'Global Accuracy', 'Loss', 'F1 Score', 'Precision', 'Recall'])

        gr.Markdown("### Performance Visualization")
        metrics_plot = gr.Plot(label="Global Metrics Analysis")
        node_plot = gr.Plot(label="Node Performance Analysis")
        
        refresh_btn.click(get_latest_metrics, outputs=metrics_text)
        refresh_btn.click(get_metrics_table, outputs=metrics_table)
        refresh_btn.click(get_metrics_plot, outputs=metrics_plot)
        refresh_btn.click(get_node_metrics_plot, outputs=node_plot)
        refresh_btn.click(get_confusion_matrix_plot, outputs=cm_plot)
        
    with gr.Tab("Data Injection"):
        # ... (Same as before)
        gr.Markdown("### Add New Data Example")
        with gr.Row():
            node_input = gr.Dropdown(choices=NODES, label="Node ID", value=1)
            mood_input = gr.Dropdown(choices=['Neutral', 'Happy', 'Sad'], label="Mood (Label)", value='Neutral')
        
        with gr.Row():
            sleep_input = gr.Number(label="Sleep Time", value=7.0)
            wakeup_input = gr.Number(label="Wakeup Time", value=7.0)
            breakfast_input = gr.Dropdown(choices=['Protein-rich', 'Heavy', 'Carb-rich', 'Skipped', 'Light'], label="Breakfast", value='Light')
            meditation_input = gr.Number(label="Meditation (min)", value=10)
            
        with gr.Row():
            rhr_input = gr.Number(label="Resting Heart Rate", value=70)
            steps_input = gr.Number(label="Step Count", value=5000)
            hrv_input = gr.Number(label="HRV", value=50)
            temp_input = gr.Number(label="Temp (C)", value=20)
            hum_input = gr.Number(label="Humidity (%)", value=50)
            
        add_btn = gr.Button("Inject Data")
        add_output = gr.Textbox(label="Status")
        
        add_btn.click(inject_data, 
                      inputs=[node_input, sleep_input, wakeup_input, breakfast_input, meditation_input, rhr_input, steps_input, hrv_input, temp_input, hum_input, mood_input],
                      outputs=add_output)

    with gr.Tab("Prediction"):
        # ... (Same as before)
        gr.Markdown("### Predict Mood using Global Model")
        with gr.Row():
            p_sleep = gr.Number(label="Sleep Time", value=7.0)
            p_wakeup = gr.Number(label="Wakeup Time", value=7.0)
            p_breakfast = gr.Dropdown(choices=['Protein-rich', 'Heavy', 'Carb-rich', 'Skipped', 'Light'], label="Breakfast", value='Light')
            p_meditation = gr.Number(label="Meditation (min)", value=10)
            
        with gr.Row():
            p_rhr = gr.Number(label="Resting Heart Rate", value=70)
            p_steps = gr.Number(label="Step Count", value=5000)
            p_hrv = gr.Number(label="HRV", value=50)
            p_temp = gr.Number(label="Temp (C)", value=20)
            p_hum = gr.Number(label="Humidity (%)", value=50)
            
        pred_btn = gr.Button("Predict")
        pred_output = gr.Textbox(label="Predicted Mood")
        
        pred_btn.click(predict_mood,
                       inputs=[p_sleep, p_wakeup, p_breakfast, p_meditation, p_rhr, p_steps, p_hrv, p_temp, p_hum],
                       outputs=pred_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
