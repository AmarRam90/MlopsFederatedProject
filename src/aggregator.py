import torch
import os
import copy
from src.model import SimpleNN

class Aggregator:
    def __init__(self, global_model_path, node_model_paths):
        self.global_model_path = global_model_path
        self.node_model_paths = node_model_paths
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_global_model(self):
        if not os.path.exists(self.global_model_path):
            os.makedirs(os.path.dirname(self.global_model_path), exist_ok=True)
            model = SimpleNN().to(self.device)
            torch.save(model.state_dict(), self.global_model_path)
            print("Initialized global model.")
            # Distribute to nodes
            self.distribute_global_model()

    def aggregate(self, eval_func=None):
        print("Aggregating models...")
        global_state = None
        count = 0
        node_accuracies = {}

        for i, path in enumerate(self.node_model_paths):
            if not os.path.exists(path):
                continue
            
            # Load node model state
            state = torch.load(path)
            
            # Evaluate if function provided
            if eval_func:
                # Create temp model to eval
                model = SimpleNN().to(self.device)
                model.load_state_dict(state)
                acc, _ = eval_func(model)
                node_accuracies[i+1] = acc
                print(f"Node {i+1} Global Test Acc: {acc:.4f}")
                
                if acc <= 0.3:
                    print(f"Node {i+1} skipped (Acc {acc:.4f} <= 0.3)")
                    continue
            
            if global_state is None:
                global_state = copy.deepcopy(state)
            else:
                for key in global_state:
                    global_state[key] += state[key]
            count += 1

        if global_state is None or count == 0:
            print("No node models qualified for aggregation.")
            return False, node_accuracies

        # Average
        for key in global_state:
            global_state[key] = global_state[key] / count

        # Save global model
        torch.save(global_state, self.global_model_path)
        print(f"Global model updated (averaged {count} nodes).")
        
        # Distribute back to nodes
        self.distribute_global_model()
        return True, node_accuracies

    def distribute_global_model(self):
        if not os.path.exists(self.global_model_path):
            return
        
        global_state = torch.load(self.global_model_path)
        for path in self.node_model_paths:
            torch.save(global_state, path)
        print("Distributed global weights to all nodes.")
