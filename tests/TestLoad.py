import unittest
from src.loader import Load
import yaml


class TestLoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        
        with open("config.yaml", "r") as r:
            cls.config = yaml.safe_load(r);
        
        cls.loader = Load(cls.config)
        
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
        copy_config = self.config.copy()
        copy_config["data"]["train_imgs"] = "incorrect_path"
        
        with self.assertRaises(FileNotFoundError):
            Load(copy_config)


if __name__ == '__main__':
    unittest.main()
