import numpy as np
import struct
from array import array
from os.path import join, exists


class Load:
    def __init__(self, data_path):
        """
        Initializing path.
        """
        paths = [
            join(data_path, 'train-images.idx3-ubyte'),
            join(data_path, 'train-labels.idx1-ubyte'),
            join(data_path, 't10k-images.idx3-ubyte'),
            join(data_path, 't10k-labels.idx1-ubyte')
        ]

        for path in paths:
            if not exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")

        self.training_images_filepath, self.training_labels_filepath, \
        self.test_images_filepath, self.test_labels_filepath = paths 
    

    def read_images_labels(self, images_filepath, labels_filepath):
        """
        Reading images and labels from the data files.
        """
        # Labels
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f"Invalid label file magic number: {magic}")
            labels = array("B", file.read())

        # Images
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f"Invalid image file magic number: {magic}")
            image_data = array("B", file.read())

        images = np.array(image_data, dtype=np.uint8).reshape(size, rows, cols)
        return images, np.array(labels)

    def load_data(self):
        """
        Loading and preprocessing the training and test data.
        """
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)

        # Normalization
        x_train, x_test = x_train / 255.0, x_test / 255.0

        return (x_train, y_train), (x_test, y_test)
