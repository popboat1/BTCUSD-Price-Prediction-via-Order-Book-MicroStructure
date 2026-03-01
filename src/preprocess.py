import pandas as pd
import numpy as np
import os

def load_and_preprocess(filepath: str, future_window: int = 50, train_split: float = 0.8):
    """
    Loads raw order book data, engineers the future target variable, 
    splits chronologically, and standardizes the features from scratch.
    
    future_window = 50 means predicting 5 seconds into the future (since data is 100ms)
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    df['future_price'] = df['mid_price'].shift(-future_window)
    
    df['target_return_bps'] = ((df['future_price'] - df['mid_price']) / df['mid_price']) * 10000
    
    df = df.dropna()
    
    feature_cols = ['spread', 'total_vol', 'wobi']
    X = df[feature_cols].values
    y = df['target_return_bps'].values.reshape(-1, 1)
    
    split_index = int(len(X) * train_split)
    
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    
    train_std[train_std == 0] = 1e-8 
    
    X_train_scaled = (X_train - train_mean) / train_std
    X_test_scaled = (X_test - train_mean) / train_std
    
    print(f"Data Processing Complete!")
    print(f"Training Samples: {len(X_train)} | Testing Samples: {len(X_test)}")
    print(f"Features used: {feature_cols}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    csv_path = 'AI/Linear Regression Model From Scratch/data/raw/btc_orderbook_10lvl.csv'
    if os.path.exists(csv_path):
        X_tr, X_te, y_tr, y_te = load_and_preprocess(csv_path)
    else:
        print(f"Could not find {csv_path}. Make sure you run data_collection.py first!")