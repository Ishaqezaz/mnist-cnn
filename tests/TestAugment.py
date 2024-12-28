import unittest
from src.loader import Load
from src.augmenter import Augmenter
import numpy as np
import yaml


class TestAugment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("config.yaml", "r") as r:
            cls.config = yaml.safe_load(r)
            
        cls.loader = Load(cls.config);
        (cls.x_train, cls.y_train), (cls.x_test, cls.y_test) = cls.loader.load_data();
        cls.augmenter = Augmenter(cls.config)
        
    def test_augment_dim(self):
        """
        Test if the augmented data dimisions are as expected
        """
        augmented_data = self.augmenter.augment(self.x_train, self.y_train, batch_size=32)
        x_batch, y_batch = next(augmented_data)
        
        self.assertEqual(x_batch.shape, (32, 28, 28, 1), "Augmented training data dim incorrect")
        self.assertEqual(y_batch.shape, (32,), "Augmented label data dim incorrect")

    def test_augment_dim_incorrect(self):
        """
        Test with incorrect data dimensions
        """
        incorrect_dim = np.random.rand(32, 28, 28, 3)
        
        with self.assertRaises(ValueError):
            self.augmenter.augment(incorrect_dim, self.y_train, batch_size=32)
    