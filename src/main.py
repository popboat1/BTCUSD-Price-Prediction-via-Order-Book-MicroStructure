import matplotlib.pyplot as plt
import numpy as np
from preprocess import load_and_preprocess
from model import LinearRegression

# 1. Configuration
CSV_FILE = 'AI/Linear Regression Model From Scratch/data/raw/btc_orderbook_10lvl.csv'
LEARNING_RATE = 0.001
ITERATIONS = 3000
BATCH_SIZE = 512
PENALTY = 'l2'
ALPHA = 0.5

def evaluate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess(CSV_FILE)
    
    print("\nInitializing Model...")
    model = LinearRegression(
        solver='mini_batch',
        learning_rate=LEARNING_RATE,
        n_iterations=ITERATIONS,
        batch_size=BATCH_SIZE,
        penalty=PENALTY,
        alpha=ALPHA
    )
    
    print("Training...")
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    test_mse = evaluate_mse(y_test, predictions)
    
    print("\n" + "="*40)
    print("Training Complete")
    print("="*40)
    print(f"Test Set Mean Squared Error: {test_mse:.4f}")
    print(f"Final Bias (Intercept): {model.bias:.4f}")
    
    print("\nLearned Feature Weights:")
    features = ['Spread', 'Total Vol', 'WOBI']
    for feature, weight in zip(features, model.weights.flatten()):
        print(f" - {feature}: {weight:.4f}")
        
    # 7. Visualize the Convergence
    plt.figure(figsize=(10, 6))
    plt.plot(model.loss_history, color='purple', alpha=0.8, linewidth=1.5)
    plt.title("Gradient Descent Convergence on Live Binance Data")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()