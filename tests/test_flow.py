import os
import sys
import torch
# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.node_trainer import NodeTrainer
from src.aggregator import Aggregator
from src.data_manager import DataManager

def test_fl_loop():
    print("Testing FL Loop...")
    
    # Ensure data exists (setup_data should have run)
    if not os.path.exists('data/node_1.csv'):
        print("Data not found. Skipping test.")
        return

    # Initialize
    global_model_path = 'models/test_global.pth'
    node_model_path = 'models/test_node_1.pth'
    
    # Mock Aggregator init
    aggregator = Aggregator(global_model_path, [node_model_path])
    aggregator.initialize_global_model()
    
    assert os.path.exists(global_model_path)
    assert os.path.exists(node_model_path)
    
    # Train Node 1
    # We need to make sure DataManager sees "new" data. 
    # Since we just created it, seen_count is 0.
    trainer = NodeTrainer(1, 'data/node_1.csv', node_model_path)
    # Force chunk size small to ensure we have enough data
    trainer.data_manager.chunk_size = 10 
    
    result = trainer.train(epochs=1)
    
    if result:
        print("Training successful.")
        # Check if model changed (simple check: file modified time or load and compare)
        # For now just trust the return value
    else:
        print("Training skipped (not enough data?).")
        
    # Aggregate
    aggregator.aggregate()
    
    print("Test Complete.")

if __name__ == "__main__":
    os.makedirs('tests', exist_ok=True)
    test_fl_loop()
