import unittest
import os
from src.loader import Load


class TestLoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/mnist-dataset'))
        cls.loader = Load(data_path)
        
    def test_data_loaded(self):
        """
        Test if the dataset dimisions are correct.
        """
        (x_train, y_train), (x_test, y_test) = self.loader.load_data()
        
        self.assertEqual(x_train.shape, (60000, 28, 28), "x_train dim incorrect")
        self.assertEqual(y_train.shape, (60000,), "y_train dim incorrect")
        
        self.assertEqual(x_test.shape, (10000, 28, 28), "x_test dim incorrect")
        self.assertEqual(y_test.shape, (10000,), "y_test dim incorrect")
        
    def test_incorrect_path(self):
        """
        Test incorrect path to data.
        """
        with self.assertRaises(FileNotFoundError):
            Load('incorrect-path')


if __name__ == '__main__':
    unittest.main()
