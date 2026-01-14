from matplotlib import pyplot as plt
import seaborn as sns
from src.config import Config
from src.data.preprocessing import load_data, train_val_split, normalize_features
from src.models.logistic_regression_1hidden import LogisticRegression1Hidden
from src.training.train import train
from src.utils.metrics import evaluate_classification
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
    X_test = normalize_features(X_test)

    model = LogisticRegression1Hidden(num_features=X_train.shape[1], hidden_dim=32, learning_rate=0.01, reg_lambda=args.reg_lambda)

    train_losses, val_losses = train(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_epochs=config.NUM_EPOCHS,
        lr=config.LEARNING_RATE,
    )

    # Plot training curves
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(config.NUM_EPOCHS), y=train_losses, label="Train Loss")
    sns.lineplot(x=range(config.NUM_EPOCHS), y=val_losses , label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.savefig(f"experiments/logistic_regression/plots/training_validation_loss_regularization-{args.reg_lambda}.png")
    plt.close()

    # Evaluate on test set
    print("\n" + "="*50)
    print("MODEL EVALUATION ON TEST SET")
    print("="*50)
    
    y_pred_test = model.predict(X_test)
    test_metrics = evaluate_classification(y_pred_test, y_test)
    
    print(f"\nAccuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1 Score:  {test_metrics['f1_score']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = test_metrics['confusion_matrix']
    print(f"  True Positives:  {cm['TP']}")
    print(f"  True Negatives:  {cm['TN']}")
    print(f"  False Positives: {cm['FP']}")
    print(f"  False Negatives: {cm['FN']}")
    
    # Also evaluate on training and validation sets
    print("\n" + "="*50)
    print("TRAINING SET PERFORMANCE")
    print("="*50)
    y_pred_train = model.predict(X_train)
    train_metrics = evaluate_classification(y_pred_train, y_train)
    print(f"Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"F1 Score:  {train_metrics['f1_score']:.4f}")
    
    print("\n" + "="*50)
    print("VALIDATION SET PERFORMANCE")
    print("="*50)
    y_pred_val = model.predict(X_val)
    val_metrics = evaluate_classification(y_pred_val, y_val)
    print(f"Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"F1 Score:  {val_metrics['f1_score']:.4f}")
    print()

if __name__ == "__main__":
    main()



