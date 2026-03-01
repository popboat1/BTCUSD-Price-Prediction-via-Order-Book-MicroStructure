import matplotlib.pyplot as plt
import numpy as np
from preprocess import load_and_preprocess
from model import LinearRegression

# 1. Configuration
CSV_FILE = 'AI/Linear Regression Model From Scratch/data/raw/btc_orderbook_10lvl.csv'
LEARNING_RATE = 0.001  # Lowered: Take smaller, more careful steps
ITERATIONS = 1200      # Increased: Give it more time to learn
BATCH_SIZE = 4096       # Increased: Average out the noise of 512 rows per step
PENALTY = 'l2'
ALPHA = 0.6

def evaluate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def main():
    # 2. Load and Preprocess the Data
    X_train, X_test, y_train, y_test = load_and_preprocess(CSV_FILE)
    
    # 3. Initialize Your Custom Model
    print("\nInitializing Custom Mini-Batch Gradient Descent Model...")
    model = LinearRegression(
        solver='mini_batch',
        learning_rate=LEARNING_RATE,
        n_iterations=ITERATIONS,
        batch_size=BATCH_SIZE,
        penalty=PENALTY,
        alpha=ALPHA
    )
    
    # 4. Train the Model
    print("Training in progress... (This proves your math works on real data!)")
    model.fit(X_train, y_train)
    
    # 5. Make Predictions on the Unseen Test Data
    predictions = model.predict(X_test)
    test_mse = evaluate_mse(y_test, predictions)
    
    # 6. Output the Results
    print("\n" + "="*40)
    print("🚀 TRAINING COMPLETE")
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