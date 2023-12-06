import os
import numpy as np
import pandas as pd
from skimage.draw import circle_perimeter_aa
import random
import yaml
from typing import NamedTuple, Optional, Tuple, Generator, Dict, Any


class CircleParams(NamedTuple):
    row: int
    col: int
    radius: int

class DatasetGenerator:
    """
    A class to generate and split datasets for training, validation, and testing.

    Attributes:
        train_path (str): Path to save the training set CSV file.
        val_path (str): Path to save the validation set CSV file.
        test_path (str): Path to save the test set CSV file.
        size (int): Image size (width and height).
        max_radius (int): Maximum radius of circles.
        level_of_noise (float): Level of noise to be added to the images.
        total_images (int): Total number of images to generate.
        train_split (float): Proportion of the dataset to be used for training.
        val_split (float): Proportion of the dataset to be used for validation.
        test_split (float): Proportion of the dataset to be used for testing.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DatasetGenerator with configuration settings.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing dataset parameters and 
                                    directory paths for saving training, validation, and test sets.

        The initialization includes setting up image size, noise level, radius range, total image count, 
        and dataset split ratios. It also prepares the necessary directories for dataset storage.
        """

        self._validate_config(config)

        self.root_dir = config['DATA']['ROOT_DIR']
        self.train_path = config['DATA']['DATA_DIR']['TRAIN']
        self.val_path = config['DATA']['DATA_DIR']['VAL']
        self.test_path = config['DATA']['DATA_DIR']['TEST']
        self.size = config['DATA']['IMG_SIZE']
        self.min_radius = config['DATA']['MIN_RADIUS']
        self.max_radius = config['DATA']['MAX_RADIUS']
        self.level_of_noise = config['DATA']['NOISE']
        self.total_images = config['DATA']['SIZE']
        self.train_split = config['DATA']['TRAIN']
        self.val_split = config['DATA']['VAL']
        self.test_split = config['DATA']['TEST']
        self._prepare_directories()

    def _validate_config(self, config: Dict[str, Any]):
        """
        Validates the provided configuration for necessary keys.

        This internal method checks if all required configuration keys are present. If any key is missing,
        it raises a ValueError.

        Args:
            config (Dict[str, Any]): The configuration dictionary to validate.

        Raises:
            ValueError: If any required configuration key is missing.
        """
        required_keys = ['DATA_DIR', 
                        'TRAIN', 
                        'VAL', 
                        'TEST', 
                        'IMG_SIZE', 
                        'MAX_RADIUS', 
                        'NOISE', 
                        'SIZE']
        missing_keys = [key for key in required_keys if key not in config['DATA']]
        if missing_keys:
            raise ValueError(f"Missing configuration parameters: {missing_keys}")

    def draw_circle(self, img: np.ndarray, row: int, col: int, rad: int):
        """
        Draws a circle on an image array.

        This method draws a circle in the given numpy array at specified coordinates and with a given radius.
        The drawing is performed in-place.

        Args:
            img (np.ndarray): The image array where the circle will be drawn.
            row (int): The row index of the circle's center.
            col (int): The column index of the circle's center.
            rad (int): The radius of the circle.
        """
        rr, cc, val = circle_perimeter_aa(row, col, rad)
        valid = (
            (rr >= 0) &
            (rr < img.shape[0]) &
            (cc >= 0) &
            (cc < img.shape[1])
        )
        img[rr[valid], cc[valid]] = val[valid]
        return

    def noisy_circle(self) -> Tuple[np.ndarray, CircleParams]:
        """
        Generates an image of a noisy circle.

        This method creates an image of a specified size, draws a circle with a random radius and center, 
        and adds Gaussian noise to the image.

        Returns:
            Tuple[np.ndarray, CircleParams]: A tuple containing the generated image and the circle parameters.
        """

        img = np.zeros((self.size, self.size))

        radius = np.random.randint(self.min_radius, self.max_radius)

        row, col = np.random.randint(self.size, size=2)

        self.draw_circle(img, row, col, radius)

        added_noise = np.random.normal(0.5, self.level_of_noise, img.shape)
        img += added_noise

        return img, CircleParams(row, col, radius)
    
    def _prepare_directories(self):
        """
        Creates necessary directories for dataset storage.

        This internal method ensures that the root directory for storing the datasets exists. 
        It creates the directory if it does not already exist.
        """
        os.makedirs(self.root_dir, exist_ok=True)

    def generate_datasets(self):
        """
        Generates the datasets and splits them into training, validation, and test sets.
        The generated images are saved as .npy files and their parameters in CSV files.
        """
        records = []
        for i in range(self.total_images):
            img, params = self.noisy_circle()
            img_path = f"data{i}.npy"
            full_path = os.path.join(self.root_dir,img_path)
            np.save(full_path, img)
            records.append([img_path, params[0], params[1], params[2]])

        df = pd.DataFrame(records, columns=['IMG_PATH', 'ROW', 'COL', 'RADIUS'])
        self._split_and_save(df)

    def _split_and_save(self, df: pd.DataFrame):
        """
        Splits the dataframe into training, validation, and test sets and saves them as CSV files
        using paths defined in the configuration.

        Args:
            df (DataFrame): The complete dataset as a pandas DataFrame.
        """
        # Split the data
        train_df = df.sample(frac=self.train_split)
        remaining_df = df.drop(train_df.index)
        val_df = remaining_df.sample(frac=self.val_split / (self.val_split + self.test_split))
        test_df = remaining_df.drop(val_df.index)

        # Save to CSV
        train_df.to_csv(self.train_path, index=False)
        val_df.to_csv(self.val_path, index=False)
        test_df.to_csv(self.test_path, index=False)



if __name__ == '__main__':
    with open('/Users/utkucicek/Desktop/SlingShotAI_Challenge/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    generator = DatasetGenerator(config)
    generator.generate_datasets()
