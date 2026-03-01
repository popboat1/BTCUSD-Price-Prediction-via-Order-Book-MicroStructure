import numpy as np

class LinearRegression:
    def __init__(
        self, 
        solver: str = 'batch', # Options: 'normal', 'batch', 'mini_batch', 'sgd'
        learning_rate: float = 0.01, 
        n_iterations: int = 1000,
        batch_size: int = 32,
        penalty: str = None,
        alpha: float = 0.0
    ):
        self.solver = solver
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.penalty = penalty
        self.alpha = alpha
        
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Appends a column of ones to X to account for the bias term"""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trains the model based on the selected solver."""
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        if self.solver == 'normal':
            self._normal_equation(X, y)
        elif self.solver in ['batch', 'mini_batch', 'sgd', 'gradient_descent']:
            self._gradient_descent(X, y)
        else:
            raise ValueError("Solver must be 'normal', 'batch', 'mini_batch', or 'sgd'")
    
    def _normal_equation(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Solves for weights mathematically.
        If L2 regularization is active, the formula becomes: 
        theta = (X^T * X + alpha * I)^-1 * X^T * y
        """
        X_b = self._add_intercept(X)
        
        if self.penalty == 'l1':
            raise ValueError("L1 regularization does not have a closed-form normal equation solution.")
        
        elif self.penalty == None:
            theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
            
        elif self.penalty == 'l2':
            I = np.eye(X_b.shape[1])
            I[0, 0] = 0
            
            theta = np.linalg.inv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y
        else:
            raise ValueError('Penalty for normal calculation must be "l2" or None.')
        
        self.bias = theta[0][0]
        self.weights = theta[1:].reshape(-1, 1)

    def _gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Iteratively updates weights to minimize Mean Squared Error.
        """
        n_samples, n_features = X.shape
        
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        
        if self.solver in ['batch', 'gradient_descent']:
            current_batch_size = n_samples
        elif self.solver == 'sgd':
            current_batch_size = 1
        else: # mini_batch
            current_batch_size = self.batch_size
            
        for i in range(self.n_iterations):
            indices = np.random.choice(n_samples, current_batch_size, replace=False)
            X_slice = X[indices]
            y_slice = y[indices]
            
            predictions = np.dot(X_slice, self.weights) + self.bias
            
            errors = predictions - y_slice
            
            dw = (2/current_batch_size) * np.dot(X_slice.T, errors)
            db = (2/current_batch_size) * np.sum(errors)
            
            if self.penalty == 'l2':
                dw += self.alpha * self.weights
            elif self.penalty == 'l1':
                dw += self.alpha * np.sign(self.weights)
                
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            mse = np.mean(errors ** 2)
            self.loss_history.append(mse)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the target variable based on learned weights."""
        if self.weights is None:
            raise Exception("Model has not been trained yet. Call fit() first.")
        
        return np.dot(X, self.weights) + self.bias