import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.loader import Load
import matplotlib.pyplot as plt


class Augmenter:
    def __init__(self):
        """
        Initializing augmentation parameters.
        """
        self.datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            shear_range=0.1,
            fill_mode='nearest'
        )

    def augment(self, x, y, batch_size=32):
        """
        Preparing augmentation on 32 batches
        """
        if len(x.shape) == 3:
            x = np.expand_dims(x, axis=-1)  # Tensorflow requires channel dim
        
        return self.datagen.flow(x, y, batch_size=batch_size)