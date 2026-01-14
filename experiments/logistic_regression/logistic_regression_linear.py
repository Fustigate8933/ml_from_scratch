from matplotlib import pyplot as plt
import seaborn as sns
from src.config import Config
from src.data.preprocessing import load_data, train_val_split, normalize_features
from src.models.logistic_regression import LogisticRegression
from src.training.train import train
import argparse

config = Config()

def main():
    parser = argparse.ArgumentParser(description="Train logistic regression model")
    parser.add_argument("--reg_lambda", type=float, default=0.0, help="L2 regularization strength")
    args = parser.parse_args()

    data = load_data(config.DATA_PATH)
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_split(data, categorical=True)
    X_train = normalize_features(X_train)
    X_val = normalize_features(X_val)

    model = LogisticRegression(num_features=X_train.shape[1], reg_lambda=args.reg_lambda)

    train_losses, val_losses = train(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_epochs=config.NUM_EPOCHS,
        lr=config.LEARNING_RATE,
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(config.NUM_EPOCHS), y=train_losses, label="Train Loss")
    sns.lineplot(x=range(config.NUM_EPOCHS), y=val_losses , label="Validation Loss")
    plt.savefig(f"training_validation_loss_regularization-{args.reg_lambda}.png")

if __name__ == "__main__":
    main()


