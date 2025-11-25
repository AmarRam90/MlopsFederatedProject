import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

class HealthDataset(Dataset):
    def __init__(self, data):
        self.data = data
        # Assuming last column is target or 'Mood' is target
        # We need to preprocess here or assume preprocessed
        # For simplicity, let's do on-the-fly preprocessing in DataManager or here.
        # Let's assume DataManager passes preprocessed tensors or we process here.
        # To keep it simple, we will process in DataManager and pass tensors.
        self.features = data['features']
        self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DataManager:
    def __init__(self, node_id=None, data_path=None):
        self.node_id = node_id
        # Expect data_path to be the prefix or we handle the logic here.
        # If data_path ends with .csv, we assume it's the train path for backward compatibility 
        # or we split it.
        # Let's assume data_path passed is the TRAIN path.
        self.data_path = data_path 
        self.test_path = data_path.replace('_train.csv', '_test.csv') if data_path and '_train.csv' in data_path else None
        
        self.chunk_size = 50
        self.seen_data_count = 0
        
        # Encoders (should be global ideally, but we fit on load for simplicity or load saved ones)
        # For a real FL system, these should be consistent across nodes.
        # We will hardcode categories for consistency.
        self.breakfast_cats = ['Protein-rich', 'Heavy', 'Carb-rich', 'Skipped', 'Light']
        self.mood_cats = ['Neutral', 'Happy', 'Sad']
        
        self.scaler = StandardScaler() # In real FL, this is tricky. We'll just fit on local data for now.

    def load_data(self):
        if not os.path.exists(self.data_path):
            return pd.DataFrame()
        return pd.read_csv(self.data_path)

    def load_test_data(self):
        if not self.test_path or not os.path.exists(self.test_path):
            return pd.DataFrame()
        return pd.read_csv(self.test_path)

    def preprocess(self, df):
        if df.empty:
            return None, None
            
        # Feature Engineering
        df = df.copy()
        # One-hot encode Breakfast_Type
        for cat in self.breakfast_cats:
            df[f'Breakfast_{cat}'] = (df['Breakfast_Type'] == cat).astype(float)
            
        # Label Encode Mood
        le = LabelEncoder()
        le.classes_ = self.mood_cats
        # Handle unseen labels if any (though we defined fixed cats)
        # df['Mood'] = df['Mood'].apply(lambda x: x if x in self.mood_cats else 'Neutral')
        labels = df['Mood'].apply(lambda x: self.mood_cats.index(x) if x in self.mood_cats else 0).values
        
        # Select features
        feature_cols = ['Sleep_Time', 'Wakeup_Time', 'Meditation_Exercise_Minutes', 
                        'Resting_Heart_Rate', 'Step_Count', 'HRV', 'Temperature_Celsius', 'Humidity_Percent'] + \
                       [f'Breakfast_{cat}' for cat in self.breakfast_cats]
                       
        features = df[feature_cols].values
        
        # Normalize (simple fit_transform on current batch - in real world use global stats)
        features = self.scaler.fit_transform(features)
        
        return torch.FloatTensor(features), torch.LongTensor(labels)

    def get_next_chunk(self):
        df = self.load_data()
        total_rows = len(df)
        
        if total_rows <= self.seen_data_count:
            return None # No new data
            
        # Get next chunk
        end_idx = min(self.seen_data_count + self.chunk_size, total_rows)
        
        if total_rows - self.seen_data_count < self.chunk_size:
            return None
            
        chunk_df = df.iloc[self.seen_data_count : self.seen_data_count + self.chunk_size]
        self.seen_data_count += self.chunk_size
        
        features, labels = self.preprocess(chunk_df)
        return HealthDataset({'features': features, 'labels': labels})
    
    def get_test_dataset(self):
        df = self.load_test_data()
        features, labels = self.preprocess(df)
        if features is None:
            return None
        return HealthDataset({'features': features, 'labels': labels})

    def append_data(self, new_data_dict):
        # new_data_dict is a dict of one row
        df = self.load_data()
        new_row = pd.DataFrame([new_data_dict])
        df = pd.concat([df, new_row], ignore_index=True)
        if not os.path.exists(os.path.dirname(self.data_path)):
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        df.to_csv(self.data_path, index=False)
