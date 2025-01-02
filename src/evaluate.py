import os
from src.loader import Load
from src.model import MNISTModel
import yaml
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score
import argparse


def evaluate(model_name=None):
    with open("config.yaml", "r") as r:
        config = yaml.safe_load(r)

    if not model_name:
        model_name = config["training"]["model_name"]

    model_path = os.path.join('models', model_name)

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

    # Precision & recall
    y_pred = model.model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    precision = precision_score(y_test, y_pred_classes, average='weighted')
    recall = recall_score(y_test, y_pred_classes, average='weighted')

    return {
        "test_accuracy": results['accuracy'],
        "test_loss": results['loss'],
        "precision": precision,
        "recall": recall
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

    results = evaluate(args.model_name)
    print("\nResults:")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")