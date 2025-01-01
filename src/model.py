import tensorflow as tf
from tensorflow.keras import layers, models
import os


class MNISTModel:
    def __init__(self, config):
        """
        Initializing the CNN model
        """
        self.input_shape = tuple(config['model'].get('input_shape', (28, 28, 1)))
        self.num_classes = config['model'].get('num_classes', 10)
        self.path = config['model']["path"]
        self.model = self.build_model()
    
    def build_model(self):
        """
        Building layers
        """
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def summary(self):
        """
        Summary
        """
        self.model.summary()
    
    def save(self, model_name):
        """
        Saving the model
        """
        filepath = os.path.join(self.path, model_name)
        self.model.save(filepath)
