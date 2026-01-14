import numpy as np

def mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray):
    """
    Compute Mean Squared Error between predictions and true values.
    """
    return np.mean((y_pred - y_true) ** 2)


def accuracy(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    """
    Compute classification accuracy.
    
    Args:
        y_pred: Predicted probabilities (0-1)
        y_true: True binary labels (0 or 1)
        threshold: Classification threshold (default 0.5)
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    return np.mean(y_pred_binary == y_true)


def precision(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    """
    Compute precision: TP / (TP + FP)
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    true_positives = np.sum((y_pred_binary == 1) & (y_true == 1))
    false_positives = np.sum((y_pred_binary == 1) & (y_true == 0))
    
    if true_positives + false_positives == 0:
        return 0.0
    return true_positives / (true_positives + false_positives)


def recall(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    """
    Compute recall (sensitivity): TP / (TP + FN)
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    true_positives = np.sum((y_pred_binary == 1) & (y_true == 1))
    false_negatives = np.sum((y_pred_binary == 0) & (y_true == 1))
    
    if true_positives + false_negatives == 0:
        return 0.0
    return true_positives / (true_positives + false_negatives)


def f1_score(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    """
    Compute F1 score: 2 * (precision * recall) / (precision + recall)
    """
    prec = precision(y_pred, y_true, threshold)
    rec = recall(y_pred, y_true, threshold)
    
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    """
    Compute confusion matrix.
    
    Returns:
        dict with keys: 'TP', 'TN', 'FP', 'FN'
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    tp = np.sum((y_pred_binary == 1) & (y_true == 1))
    tn = np.sum((y_pred_binary == 0) & (y_true == 0))
    fp = np.sum((y_pred_binary == 1) & (y_true == 0))
    fn = np.sum((y_pred_binary == 0) & (y_true == 1))
    
    return {'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)}


def evaluate_classification(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    """
    Compute all classification metrics at once.
    
    Returns:
        dict with all metrics
    """
    cm = confusion_matrix(y_pred, y_true, threshold)
    
    return {
        'accuracy': accuracy(y_pred, y_true, threshold),
        'precision': precision(y_pred, y_true, threshold),
        'recall': recall(y_pred, y_true, threshold),
        'f1_score': f1_score(y_pred, y_true, threshold),
        'confusion_matrix': cm
    }

