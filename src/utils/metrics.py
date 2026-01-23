import numpy as np

def mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray):
    """
    Compute Mean Squared Error between predictions and true values.
    """
    return np.mean((y_pred - y_true) ** 2)


def accuracy(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    """
    Compute classification accuracy.
    For multiclass, y_pred should be (n_samples, n_classes) and y_true integer labels.
    For binary, y_pred can be probabilities.
    """
    if y_pred.ndim == 2:
        y_pred_labels = np.argmax(y_pred, axis=1)
        return np.mean(y_pred_labels == y_true)
    else:
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
    For multiclass, returns a matrix. For binary, returns dict.
    """
    if y_pred.ndim == 2:
        y_pred_labels = np.argmax(y_pred, axis=1)
        n_classes = np.max(np.concatenate([y_pred_labels, y_true])) + 1
        matrix = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y_true, y_pred_labels):
            matrix[int(t), int(p)] += 1
        return matrix
    else:
        y_pred_binary = (y_pred >= threshold).astype(int)
        tp = np.sum((y_pred_binary == 1) & (y_true == 1))
        tn = np.sum((y_pred_binary == 0) & (y_true == 0))
        fp = np.sum((y_pred_binary == 1) & (y_true == 0))
        fn = np.sum((y_pred_binary == 0) & (y_true == 1))
        return {'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)}


def evaluate_classification(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    """
    Compute all classification metrics at once.
    For multiclass, only accuracy and confusion matrix are returned.
    """
    if y_pred.ndim == 2:
        return {
            'accuracy': accuracy(y_pred, y_true),
            'confusion_matrix': confusion_matrix(y_pred, y_true)
        }
    else:
        cm = confusion_matrix(y_pred, y_true, threshold)
        return {
            'accuracy': accuracy(y_pred, y_true, threshold),
            'precision': precision(y_pred, y_true, threshold),
            'recall': recall(y_pred, y_true, threshold),
            'f1_score': f1_score(y_pred, y_true, threshold),
            'confusion_matrix': cm
        }

