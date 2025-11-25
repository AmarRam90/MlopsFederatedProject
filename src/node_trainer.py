import torch
import torch.nn as nn
import torch.optim as optim
from src.model import SimpleNN
from src.data_manager import DataManager
import os

class NodeTrainer:
    def __init__(self, node_id, data_path, model_path):
        self.node_id = node_id
        self.data_manager = DataManager(node_id, data_path)
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, epochs=50, lr=0.005):
        # Get next chunk of data
        dataset = self.data_manager.get_next_chunk()
        if dataset is None:
            print(f"Node {self.node_id}: No new data chunk (need 50 new examples). Skipping training.")
            return False

        print(f"Node {self.node_id}: Starting training on new chunk of size {len(dataset)}")

        # Load model
        model = SimpleNN().to(self.device)
        if os.path.exists(self.model_path):
            try:
                model.load_state_dict(torch.load(self.model_path))
            except RuntimeError:
                print(f"Node {self.node_id}: Model architecture mismatch (likely due to update). Starting fresh/using global.")
                pass
        
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for features, labels in loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
        # Save updated model
        if not os.path.exists(self.model_path):
             os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(model.state_dict(), self.model_path)
        
        print(f"Node {self.node_id}: Training complete. Model saved.")
        return True
