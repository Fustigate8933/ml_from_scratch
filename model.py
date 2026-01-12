import numpy as np

class LinearRegression:
    def __init__(self, num_features: int, reg_lambda: float = 0.0):
        self.weights = np.random.randn(1, num_features) * 0.01
        self.bias = np.random.randn(1)
        self.grad_w = np.zeros_like(self.weights)
        self.grad_b = 0.0
        self.reg_lambda = reg_lambda  # L2 regularization strength

    def forward(self, X: np.ndarray):
        """
        Compute predictions.
        """
        return X @ self.weights.T + self.bias

    def predict(self, X: np.ndarray):
        """
        same as forward
        """
        return self.forward(X)

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Mean Squared Error loss.
        """
        mse = np.mean((y_pred - y_true) ** 2)
        l2_penalty = self.reg_lambda * np.sum(self.weights ** 2)
        return mse + l2_penalty

    def backward(self, X: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """
        Compute gradients for weights and bias.
        we want to minimize the loss, which is the mean squared error
        updates internal gradients grad_w and grad_b
        """
        n = X.shape[0]
        error = y_pred - y_true
        grad_w = (2/n) * (error.T @ X).reshape(1, -1) # derivative of MSE wrt weights
        grad_w += 2 * self.weights * self.reg_lambda  # L2 regularization gradient
        grad_b = (2/n) * np.sum(error)
        self.grad_w = grad_w
        self.grad_b = grad_b

    def step(self, lr: float) -> None:
        """
        Update parameters using gradients.
        """
        self.weights -= lr * self.grad_w
        self.bias -= lr * self.grad_b
