---
title: BTC Microstructure Predictor
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "3.50.2"
app_file: src/api.py
app_port: 8000
pinned: false
---

# High-Frequency Bitcoin Price Prediction via Order Book Microstructure

## Project Overview
In this project, I built a linear regression model from scratch using NumPy to predict short-term Bitcoin price movements. Unlike typical trading bots that rely on lagging indicators like RSI or MACD, I designed this system to utilize high-frequency Order Book data. By analyzing the immediate supply and demand imbalance, the model forecasts price changes on a five-second horizon, providing a more "real-time" look at market sentiment.

## Technical Architecture
The system is architected into three distinct layers to handle high-frequency data:

### 1. Predictive Engine (Python)
- Core Model: I implemented a custom Linear Regression class from the ground up, using Mini-Batch Gradient Descent for optimization rather than relying on high-level libraries like Scikit-Learn.
- Feature Engineering: I developed a Weighted Order Book Imbalance (WOBI) feature. This mathematically calculates the ratio of buy and sell pressure across ten levels of depth, applying a linear decay weight so that orders closest to the mid-price have the highest impact on the prediction.
- Regularization: To handle the extreme noise of the crypto market, I integrated L2 (Ridge) regularization into the cost function, which prevents the model weights from becoming too sensitive to volatile "spoof" orders.

### 2. Backend API (FastAPI)
- I used FastAPI to create a RESTful interface that hosts the trained model in-memory.
- The backend includes a preprocessing pipeline that standardizes live input using the specific mean and standard deviation from my training set to maintain mathematical integrity during live inference.

### 3. Frontend Dashboard (Next.js & TypeScript)
- For the UI, I used Next.js and the TradingView Lightweight Charts library to build a professional-grade dashboard.
- I implemented a hybrid data strategy: the chart preloads historical 1-second k-lines via REST API and then seamlessly "stitches" live microstructure updates from a WebSocket for a smooth, flicker-free experience.

## Mathematics of the Model
The engine minimizes the Mean Squared Error (MSE) augmented by an L2 penalty term:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \frac{\alpha}{2} \sum_{j=1}^{n} \theta_j^2$$

I chose Mini-Batch Gradient Descent to allow the model to update its parameters efficiently across the 100ms data firehose while maintaining a stable convergence curve.

## Deployment and Setup
1. Backend: Initialize the virtual environment and run `src/api.py` to boot the FastAPI server.
2. Frontend: Navigate to the `/frontend` directory, run `npm install`, and then `npm run dev`.
3. Interaction: The dashboard connects to the local inference engine and visualizes the 5-second price projection against live market data.
