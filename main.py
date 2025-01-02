from src.train import train
from src.evaluate import evaluate
import tensorflow as tf
import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

def main():
    
    print("Training")
    train()

    print("\nEvaluating")
    results = evaluate(); # Latest trained model
    
    print("\nResults:")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")


if __name__ == "__main__":
    main()
