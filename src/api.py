from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import uvicorn
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_and_preprocess
from model import LinearRegression

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'btc_orderbook_10lvl.csv')

class OrderBookFeatures(BaseModel):
    current_price: float
    spread: float
    total_vol: float
    wobi: float

model = None
scaler_params = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads data, scales it, and trains the model when the server boots up."""
    global model, scaler_params
    print("\n--- Starting API Server ---")
    
    if not os.path.exists(CSV_FILE):
        raise RuntimeError(f"Data file not found at: {CSV_FILE}")
    
    X_train, _, y_train, _ = load_and_preprocess(CSV_FILE, future_window=50)
    
    df = pd.read_csv(CSV_FILE).dropna()
    raw_features = df[['spread', 'total_vol', 'wobi']].values
    train_split_idx = int(len(raw_features) * 0.8)
    
    scaler_params['mean'] = np.mean(raw_features[:train_split_idx], axis=0)
    scaler_params['std'] = np.std(raw_features[:train_split_idx], axis=0)
    scaler_params['std'][scaler_params['std'] == 0] = 1e-8
    
    print("Training model in memory...")
    model = LinearRegression(
        solver='mini_batch', 
        batch_size=512, 
        learning_rate=0.001, 
        n_iterations=3000, 
        penalty='l2', 
        alpha=0.5
    )
    model.fit(X_train, y_train)
    print("Model successfully loaded and ready for predictions!\n")
    
    yield 
    
    print("\n--- Shutting Down API Server ---")

app = FastAPI(
    title="BTC Quant Model API", 
    description="Predicts future BTC prices based on order book imbalance.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict_price(features: OrderBookFeatures):
    if model is None or model.weights is None:
        raise HTTPException(status_code=500, detail="Model is not trained yet.")
    
    raw_input = np.array([features.spread, features.total_vol, features.wobi])
    scaled_input = (raw_input - scaler_params['mean']) / scaler_params['std']
    scaled_input = scaled_input.reshape(1, -1) 
    
    prediction_bps = float(model.predict(scaled_input)[0][0])
    projected_price = features.current_price * (1 + (prediction_bps / 10000))
    
    return {
        "current_price": features.current_price,
        "predicted_return_bps": round(prediction_bps, 4),
        "predicted_future_price": round(projected_price, 2),
        "signal": "BULLISH" if prediction_bps > 0 else "BEARISH"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)