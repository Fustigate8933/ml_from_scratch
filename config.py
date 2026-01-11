from dataclasses import dataclass


@dataclass
class Config:
    """
    Configuration parameters for linear regression model
    """
    DATA_PATH = "data.csv"
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10000
    BATCH_SIZE = None
    RANDOM_SEED = 42
