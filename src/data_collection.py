import asyncio
import websockets
import json
import csv
import os
from datetime import datetime

os.makedirs('AI/Linear Regression Model From Scratch/data/raw', exist_ok=True)
CSV_FILE = 'AI/Linear Regression Model From Scratch/data/raw/btc_orderbook_10lvl.csv'

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'mid_price', 'spread', 'total_vol', 'wobi'])

async def stream_order_book(symbol="btcusdt"):
    url = f"wss://stream.binance.com:9443/ws/{symbol}@depth10@100ms"
    print(f"Connecting to Binance {symbol.upper()} 10-Level Stream...")
    
    async with websockets.connect(url) as ws:
        print(f"Connected! Calculating WOBI and logging to {CSV_FILE}...\n")
        
        try:
            while True:
                response = await ws.recv()
                data = json.loads(response)
                
                bids = data['bids']
                asks = data['asks']
                
                best_bid_price = float(bids[0][0])
                best_ask_price = float(asks[0][0])
                mid_price = round((best_ask_price + best_bid_price) / 2, 2)
                spread = round(best_ask_price - best_bid_price, 2)
                
                weighted_bid_vol = 0
                weighted_ask_vol = 0
                total_raw_vol = 0
                
                for i in range(10):
                    bid_vol = float(bids[i][1])
                    ask_vol = float(asks[i][1])
                    total_raw_vol += (bid_vol + ask_vol)
                    
                    weight = 1.0 - (i * 0.1) 
                    
                    weighted_bid_vol += (bid_vol * weight)
                    weighted_ask_vol += (ask_vol * weight)
                
                if (weighted_bid_vol + weighted_ask_vol) > 0:
                    wobi = (weighted_bid_vol - weighted_ask_vol) / (weighted_bid_vol + weighted_ask_vol)
                else:
                    wobi = 0.0
                
                wobi = round(wobi, 4)
                total_raw_vol = round(total_raw_vol, 4)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                print(f"[{timestamp}] Mid: ${mid_price} | Spread: ${spread} | WOBI: {wobi}")
                
                with open(CSV_FILE, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, mid_price, spread, total_raw_vol, wobi])
                    
        except KeyboardInterrupt:
            print(f"\nData stream stopped. Data successfully saved to {CSV_FILE}.")

if __name__ == "__main__":
    asyncio.run(stream_order_book())