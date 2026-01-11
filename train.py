from model import LinearRegression
import numpy as np


def train_one_epoch(model: LinearRegression, X: np.ndarray, y: np.ndarray, lr: float):
    """
    Run one epoch of training.
    """
    y_pred = model.forward(X).reshape(-1)
    loss = model.compute_loss(y_pred, y)
    model.backward(X, y_pred, y)
    model.step(lr)
    return loss


def validate(model: LinearRegression, X: np.ndarray, y: np.ndarray):
    """
    Evaluate model on validation set.
    """
    y_pred = model.predict(X).reshape(-1)
    loss = model.compute_loss(y_pred, y)
    return loss


def train(
    model: LinearRegression,
    X_train,
    y_train,
    X_val,
    y_val,
    num_epochs: int,
    lr: float,
    log_interval: int = 1000,
):
    """
    Full training loop.
    """
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, X_train, y_train, lr)
        val_loss = validate(model, X_val, y_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % log_interval == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    return train_losses, val_losses
