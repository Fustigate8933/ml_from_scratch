import numpy as np
from src.models.base import BaseModel
from src.models.linear_layer import Linear

class LogisticRegression1Hidden(BaseModel):
    def __init__(self, num_features: int, hidden_dim: int = 64, learning_rate: float = 0.01, reg_lambda: float = 0.0):
        super().__init__(num_features, reg_lambda)
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        
        # Initialize two linear layers
        self.layer1 = Linear(num_features, hidden_dim, learning_rate)
        self.n_classes = 4
        self.layer2 = Linear(hidden_dim, self.n_classes, learning_rate)

    def forward(self, X: np.ndarray):
        """
        Compute predictions through the network.
        X -> Linear -> ReLU -> Linear -> Softmax
        """
        # First layer + ReLU activation
        self.z1 = self.layer1.forward(X)  # (n_samples, hidden_dim)
        self.a1 = self.relu(self.z1)  # (n_samples, hidden_dim)
        # Second layer + Softmax activation
        self.z2 = self.layer2.forward(self.a1)  # (n_samples, n_classes)
        y_pred = self.softmax(self.z2)  # (n_samples, n_classes)
        return y_pred

    def predict(self, X: np.ndarray):
        """
        Returns class predictions (argmax of softmax output)
        """
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Categorical cross-entropy loss with L2 regularization.
        y_true: (n_samples,) integer class labels (0,1,2,3)
        y_pred: (n_samples, n_classes) softmax probabilities
        """
        n = y_true.shape[0]
        # One-hot encode y_true
        y_true_onehot = np.zeros_like(y_pred)
        y_true_onehot[np.arange(n), y_true.astype(int)] = 1
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        # Categorical cross-entropy loss
        ce_loss = - (1/n) * np.sum(y_true_onehot * np.log(y_pred_clipped))
        # L2 regularization on both layers
        l2_reg = self.reg_lambda * (np.sum(self.layer1.weights ** 2) + np.sum(self.layer2.weights ** 2))
        total_loss = ce_loss + l2_reg
        return total_loss

    def backward(self, X: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """
        Compute gradients using backpropagation for softmax + cross-entropy.
        y_true: (n_samples,) integer class labels (0,1,2,3)
        y_pred: (n_samples, n_classes) softmax probabilities
        """
        n = y_true.shape[0]
        # One-hot encode y_true
        y_true_onehot = np.zeros_like(y_pred)
        y_true_onehot[np.arange(n), y_true.astype(int)] = 1
        # Gradient of loss w.r.t logits (softmax + cross-entropy)
        dL_dz2 = (y_pred - y_true_onehot) / n  # (n_samples, n_classes)
        # Backprop through layer2
        dL_dW2, dL_db2, dL_da1 = self.layer2.compute_gradients(dL_dz2)
        # Add L2 regularization gradient
        dL_dW2 += 2 * self.reg_lambda * self.layer2.weights
        # Backprop through ReLU
        dL_dz1 = dL_da1 * self.relu_derivative(self.z1)  # (n_samples, hidden_dim)
        # Backprop through layer1
        dL_dW1, dL_db1, _ = self.layer1.compute_gradients(dL_dz1)
        # Add L2 regularization gradient
        dL_dW1 += 2 * self.reg_lambda * self.layer1.weights
        # Store gradients
        self.dL_dW1 = dL_dW1
        self.dL_db1 = dL_db1
        self.dL_dW2 = dL_dW2
        self.dL_db2 = dL_db2

    def step(self, lr: float = None) -> None:
        """
        Update parameters using gradients.
        """
        if lr is None:
            lr = self.lr
            
        # Update both layers
        self.layer1.step(self.dL_dW1, self.dL_db1)
        self.layer2.step(self.dL_dW2, self.dL_db2)

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def relu(self, z: np.ndarray) -> np.ndarray:
        """
        ReLU activation function.
        """
        return np.maximum(0, z)
    
    def relu_derivative(self, z: np.ndarray) -> np.ndarray:
        """
        Derivative of ReLU activation function.
        """
        return (z > 0).astype(float)
    
    def softmax(self, z: np.ndarray) -> np.ndarray:
        """
        softmax activation
        z: (n_samples, n_classes)
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
