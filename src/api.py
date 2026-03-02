import asyncio
import json
import websockets
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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

model = None
scaler_params = {}

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()
binance_task = None

# --- Background Binance Listener & Prediction Engine ---
async def binance_listener():
    """Connects to Binance, calculates WOBI, predicts, and broadcasts every 100ms."""
    uri = "wss://stream.binance.com:9443/ws/btcusdt@depth10@100ms"
    
    while True:
        try:
            async with websockets.connect(uri) as ws:
                print("\n[System] Connected to Binance Live Stream")
                async for msg in ws:
                    # Skip heavy math if no users are looking at the dashboard
                    if not manager.active_connections:
                        continue
                        
                    data = json.loads(msg)
                    bids = data['bids']
                    asks = data['asks']
                    
                    best_bid = float(bids[0][0])
                    best_ask = float(asks[0][0])
                    mid_price = (best_ask + best_bid) / 2.0
                    spread = best_ask - best_bid
                    
                    # 1. Feature Engineering: Calculate WOBI
                    weighted_bid_vol = 0
                    weighted_ask_vol = 0
                    total_vol = 0
                    
                    for i in range(10):
                        bid_vol = float(bids[i][1])
                        ask_vol = float(asks[i][1])
                        total_vol += (bid_vol + ask_vol)
                        
                        weight = 1.0 - (i * 0.1)
                        weighted_bid_vol += (bid_vol * weight)
                        weighted_ask_vol += (ask_vol * weight)
                        
                    wobi = 0.0
                    if (weighted_bid_vol + weighted_ask_vol) > 0:
                        wobi = (weighted_bid_vol - weighted_ask_vol) / (weighted_bid_vol + weighted_ask_vol)
                        
                    # 2. Model Inference
                    raw_input = np.array([spread, total_vol, wobi])
                    scaled_input = (raw_input - scaler_params['mean']) / scaler_params['std']
                    scaled_input = scaled_input.reshape(1, -1) 
                    
                    prediction_bps = float(model.predict(scaled_input)[0][0])
                    projected_price = mid_price * (1 + (prediction_bps / 10000))
                    
                    # 3. Construct Unified Payload
                    payload = {
                        "bids": bids,
                        "asks": asks,
                        "mid_price": mid_price,
                        "spread": round(spread, 2),
                        "wobi": round(wobi, 4),
                        "predicted_future_price": round(projected_price, 2),
                        "signal": "BULLISH" if prediction_bps > 0 else "BEARISH"
                    }
                    
                    # 4. Broadcast to all connected Next.js clients
                    await manager.broadcast(json.dumps(payload))
                    
        except Exception as e:
            print(f"[Error] Binance WS Disconnected: {e}. Reconnecting in 3 seconds...")
            await asyncio.sleep(3)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler_params, binance_task
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
        solver='mini_batch', batch_size=512, learning_rate=0.001, 
        n_iterations=3000, penalty='l2', alpha=0.5
    )
    model.fit(X_train, y_train)
    print("Model successfully loaded!")
    
    # Launch the background Binance worker
    binance_task = asyncio.create_task(binance_listener())
    
    yield 
    
    print("\n--- Shutting Down API Server ---")
    if binance_task:
        binance_task.cancel()

app = FastAPI(title="BTC Quant Model API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, 
    allow_methods=["*"], allow_headers=["*"],
)

# --- NEW: WebSocket Endpoint for the Frontend ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)