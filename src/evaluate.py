import os
from src.loader import Load
from src.model import MNISTModel
import yaml
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import seaborn as sns
import argparse
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plots confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()

def plot_training_history(history_1, history_2=None, title="Training History"):
    """
    Plots accuracy and loss curves from training history.
    """
    
    history_2 = history_2 or {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

    # Merge histories
    history = {
        "accuracy": history_1.history.get("accuracy", []) + history_2.history.get("accuracy", []),
        "val_accuracy": history_1.history.get("val_accuracy", []) + history_2.history.get("val_accuracy", []),
        "loss": history_1.history.get("loss", []) + history_2.history.get("loss", []),
        "val_loss": history_1.history.get("val_loss", []) + history_2.history.get("val_loss", []),
    }

    # Plot acc
    plt.figure(figsize=(10, 5))
    plt.plot(history["accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title(f"{title} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

def evaluate(config, model_name=None):
    if not model_name:
        model_name = config["training"]["model_name"]

    model_path = os.path.join(config["model"]["path"], model_name)

    # Loading data
    loader = Load(config)
    (_, _), (x_test, y_test) = loader.load_data()

    # Loading and compiling model
    model = MNISTModel(config)
    model.model.load_weights(model_path)
    model.model.compile(
        optimizer=tf.keras.optimizers.Adam(config['training']['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    results = model.model.evaluate(x_test, y_test, return_dict=True)
    
    # Predictions
    y_pred = model.model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Precision & recall
    precision = precision_score(y_test, y_pred_classes, average='weighted')
    recall = recall_score(y_test, y_pred_classes, average='weighted')

    return {
        "test_accuracy": results['accuracy'],
        "test_loss": results['loss'],
        "precision": precision,
        "recall": recall,
        "y_true": y_test,
        "y_pred": y_pred_classes
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick evaluation")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Specify the model name)"
    )
    args = parser.parse_args()
    
    with open("./config.yaml", "r") as r:
        config = yaml.safe_load(r)

    results = evaluate(config, args.model_name)
    print("\nResults:")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")