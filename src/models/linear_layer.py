import numpy as np

class Linear:
    def __init__(self, input_dim, output_dim, learning_rate=0.01):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = learning_rate

        # Initialize weights and bias
        # a linear layer has output_dim number of neurons
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.random.randn(1, output_dim) * 0.01

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.input = X  # store for backpropagation
        output = X @ self.weights + self.bias
        return output

    def compute_gradients(self, dL_dout: np.ndarray) -> tuple:
        dL_dW = self.input.T @ dL_dout  # shape: (input_dim, output_dim)

        dL_db = np.sum(dL_dout, axis=0, keepdims=True)  # shape: (1, output_dim)

        dL_dinput = dL_dout @ self.weights.T  # shape: (batch_size, input_dim)

        return dL_dW, dL_db, dL_dinput

    def step(self, dL_dW: np.ndarray, dL_db: np.ndarray):
        self.weights -= self.lr * dL_dW
        self.bias -= self.lr * dL_db
