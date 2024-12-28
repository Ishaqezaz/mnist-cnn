import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Augmenter:
    def __init__(self, config):
        """
        Initializing augmentation parameters.
        """
        self.datagen = ImageDataGenerator(
            rotation_range = config["augmentation"]["rotation"],
            width_shift_range = config["augmentation"]["width_shift"],
            height_shift_range = config["augmentation"]["height_shift"],
            zoom_range = config["augmentation"]["zoom"],
            shear_range = config["augmentation"]["shear"],
            fill_mode = config["augmentation"]["fill_mode"]
        )
        self.batch_size = config["augmentation"]["batch_size"]
        
    def augment(self, x, y, batch_size=None):
        """
        Preparing augmentation on 32 batches default
        """
        if len(x.shape) not in [3,4]: # Safety
            raise ValueError("Incorrect dimensions for augmentation")

        if len(x.shape) == 3:
            x = np.expand_dims(x, axis=-1)  # Tensorflow requires channel dim
        
        if batch_size is None:
            batch_size = self.batch_size
        
        return self.datagen.flow(x, y, batch_size)