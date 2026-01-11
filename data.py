import numpy as np
import pandas as pd
from config import Config


config = Config()


def load_data(path: str) -> pd.DataFrame:
    """
    Load and preprocess car data
    """
    car_data = pd.read_csv(path)

    car_data["Age"] = car_data["Year"].max() - car_data["Year"]
    features = ["Age", "Engine HP", "Engine Cylinders", "Number of Doors", "highway MPG", "city mpg", "Popularity"]
    
    for v in ['AUTOMATIC', 'MANUAL', 'AUTOMATED_MANUAL']:
        feature = f"is_transmission_{v}"
        car_data[feature] = (car_data['Transmission Type'] == v).astype(int)
        features.append(feature)

    data = car_data[features + ["MSRP"]].dropna().reset_index(drop=True)

    np.random.seed(config.RANDOM_SEED)
    n = len(data)
    idx = np.arange(n)
    np.random.shuffle(idx)
    data_shuffled = data.iloc[idx]

    return data_shuffled


def train_val_split(data: pd.DataFrame, frac: float = 0.2) -> tuple[np.ndarray, pd.Series, np.ndarray, pd.Series, np.ndarray, pd.Series]:
    """
    Split data into training, validation, and test sets
    """
    n = len(data)
    n_test = int(n * frac)
    n_val = int(n * frac)
    n_train = n - n_val - n_test
    
    data_train = data[n_train:].reset_index(drop=True)
    data_val = data[n_train: n_train + n_val].reset_index(drop=True)
    data_test = data[n_train + n_val:].reset_index(drop=True)

    X_train = data_train.drop(columns=["MSRP"]).values
    y_train = data_train["MSRP"]
    X_val = data_val.drop(columns=["MSRP"]).values
    y_val = data_val["MSRP"]
    X_test = data_test.drop(columns=["MSRP"]).values
    y_test = data_test["MSRP"]

    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize_features(X: np.ndarray):
    """
    Normalize features (mean/std).
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_norm = (X - mean) / std
    return X_norm
