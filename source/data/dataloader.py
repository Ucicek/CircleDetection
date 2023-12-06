import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional

class CircleDataset(Dataset):
    def __init__(self, config: dict, csv_file: str, transform: Optional[callable] = None, test: bool = False):
        """
        Initializes the CircleDataset with the given configuration, CSV file, and optional transform.

        Args:
            config (dict): Configuration dictionary specifying dataset details such as root directory.
            csv_file (str): Path to the CSV file containing image paths and circle parameters.
            transform (Optional[callable]): A function/transform that takes in an image and returns a transformed version.
            test (bool, optional): Flag to indicate if the dataset is for testing. If False, it's considered for training. 
                                    Defaults to False.
        """
        self.root_dir = config['DATA']['ROOT_DIR']
        self.file_dir = os.path.join(self.root_dir,csv_file)
        self.circle_frame = pd.read_csv(csv_file) if test else pd.read_csv(self.file_dir)
        self.transform = transform

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.circle_frame)

    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.root_dir, self.circle_frame.iloc[idx, 0])
            img_array = np.load(img_path)

            # Ensure the data is in the correct format

            circle_params = self.circle_frame.iloc[idx, 1:].values

            image = np.expand_dims(np.asarray(img_array), axis=0)
            image = torch.from_numpy(np.array(image, dtype=np.float32))

            circle_params = torch.from_numpy(np.array(np.asarray(circle_params), dtype=np.float32))

            if self.transform:
                image = self.transform(image)
            return image, circle_params
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            return None
        


class CircleDatasetGenerator(Dataset):
    def __init__(self, config: dict, generator: callable, dataset_length: int = 1000):
        """
        Initializes the CircleDatasetGenerator using a provided image generator function.

        Args:
            config (dict): Configuration dictionary specifying dataset details.
            generator (callable): A generator function that yields tuples of (image, circle parameters).
            dataset_length (int, optional): The total number of items that the dataset will have. 
                                             Defaults to 1000 or the 'SIZE' value in the configuration.

        """
        self.config = config
        self.generator = generator
        self.dataset_length = self.config['DATA'].get('SIZE', dataset_length)
        
    def __len__(self):
        """Return the number of items in the dataset."""
        return self.dataset_length

    def __getitem__(self, idx):
        try:
            img, params = next(self.generator)

            image = np.expand_dims(np.asarray(img), axis=0)  
            image = torch.from_numpy(image.astype(np.float32)) 

            params = torch.from_numpy(np.array(params, dtype=np.float32))

            return image, params
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            return None
