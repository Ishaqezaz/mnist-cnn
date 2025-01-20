from src.train import train
from src.evaluate import evaluate, plot_training_history, plot_confusion_matrix
import tensorflow as tf
import warnings
import os
import yaml

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

def main(config=None):
    
    if config is None:
        with open("config.yaml", "r") as r:
            config = yaml.safe_load(r)

    print("Training")
    history_1, history_2 = train(config)

    print("\nEvaluating")
    results = evaluate(config)
    
    print("\nResults:")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")

    print("\nTraining history:")
    plot_training_history(history_1, history_2)
    
    print("\nConfusion matrix:")
    plot_confusion_matrix(results["y_true"], results["y_pred"])


if __name__ == "__main__":
    main()
