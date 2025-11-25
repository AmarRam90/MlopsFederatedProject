import pandas as pd
import os
import numpy as np

def resplit_data_v2():
    raw_path = 'raw_data_upload.csv'
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found.")
        return

    df = pd.read_csv(raw_path)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    total_rows = len(df)
    print(f"Total rows: {total_rows}")
    
    # We need 250 rows per node total (Train + Test)
    # 15% of 250 is 37.5 -> 38 rows for test, 212 for train.
    
    node_size = 250
    test_size = int(node_size * 0.15)
    train_size = node_size - test_size
    
    print(f"Per Node: {node_size} total. Train: {train_size}, Test: {test_size}")
    
    start = 0
    os.makedirs('data', exist_ok=True)
    
    for i in range(1, 4):
        node_subset = df.iloc[start : start + node_size]
        start += node_size
        
        # Split into train/test
        node_test = node_subset.iloc[:test_size]
        node_train = node_subset.iloc[test_size:]
        
        node_train.to_csv(f'data/node_{i}_train.csv', index=False)
        node_test.to_csv(f'data/node_{i}_test.csv', index=False)
        print(f"Node {i}: Saved train ({len(node_train)}) and test ({len(node_test)})")
        
    # Rest for global eval
    global_eval = df.iloc[start:]
    global_eval.to_csv('data/global_eval.csv', index=False)
    print(f"Saved data/global_eval.csv ({len(global_eval)} rows)")

if __name__ == "__main__":
    resplit_data_v2()
