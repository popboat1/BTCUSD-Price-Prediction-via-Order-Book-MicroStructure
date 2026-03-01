import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.linear_model import Ridge as SklearnRidge

from src.model import LinearRegression as CustomLR

@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.rand(100, 3)
    
    y = 2 * X[:, 0] - 3 * X[:, 1] + 1.5 * X[:, 2] + 5 + np.random.randn(100) * 0.1
    y = y.reshape(-1, 1)
    
    return X, y

def test_normal_equation_no_penalty(sample_data):
    X, y = sample_data
    
    custom_model = CustomLR(solver='normal', penalty=None)
    custom_model.fit(X, y)
    
    sk_model = SklearnLR()
    sk_model.fit(X, y)
    
    np.testing.assert_almost_equal(custom_model.bias, sk_model.intercept_[0], decimal=5)
    np.testing.assert_almost_equal(custom_model.weights.flatten(), sk_model.coef_.flatten(), decimal=5)

def test_normal_equation_l2_penalty(sample_data):
    X, y = sample_data
    alpha_val = 1.0
    
    custom_model = CustomLR(solver='normal', penalty='l2', alpha=alpha_val)
    custom_model.fit(X, y)
    
    sk_model = SklearnRidge(alpha=alpha_val, solver='cholesky') 
    sk_model.fit(X, y)
    
    np.testing.assert_almost_equal(custom_model.bias, sk_model.intercept_[0], decimal=5)
    np.testing.assert_almost_equal(custom_model.weights.flatten(), sk_model.coef_.flatten(), decimal=5)

def test_batch_gradient_descent(sample_data):
    X, y = sample_data
    
    custom_model = CustomLR(solver='batch', learning_rate=0.01, n_iterations=5000)
    custom_model.fit(X, y)
    
    sk_model = SklearnLR()
    sk_model.fit(X, y)
    
    np.testing.assert_almost_equal(custom_model.bias, sk_model.intercept_[0], decimal=1)
    np.testing.assert_almost_equal(custom_model.weights.flatten(), sk_model.coef_.flatten(), decimal=1)

def test_mini_batch_loss_decreases(sample_data):
    X, y = sample_data
    
    custom_model = CustomLR(solver='mini_batch', batch_size=16, learning_rate=0.01, n_iterations=1000)
    custom_model.fit(X, y)
    
    initial_loss = np.mean(custom_model.loss_history[:10])
    final_loss = np.mean(custom_model.loss_history[-10:])
    
    assert final_loss < initial_loss, f"Loss did not decrease! Initial: {initial_loss}, Final: {final_loss}"

def test_sgd_with_l2_penalty(sample_data):
    X, y = sample_data
    
    unpenalized_model = CustomLR(solver='sgd', learning_rate=0.01, n_iterations=1000, penalty=None)
    unpenalized_model.fit(X, y)
    
    penalized_model = CustomLR(solver='sgd', learning_rate=0.01, n_iterations=1000, penalty='l2', alpha=10.0)
    penalized_model.fit(X, y)
    
    unpenalized_magnitude = np.sum(np.abs(unpenalized_model.weights))
    penalized_magnitude = np.sum(np.abs(penalized_model.weights))
    
    assert penalized_magnitude < unpenalized_magnitude, "L2 penalty failed to shrink weights"