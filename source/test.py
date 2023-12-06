import pandas as pd
import numpy as np
import yaml
import os
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List

from data.dataloader import CircleDataset
from utils import get_network, iou



def sort_and_save_images_by_radius(csv_file: str, output_dir: str = 'test') -> None:
    """
    Sorts images by radius and saves them into separate CSV files based on specified radius ranges.
    Args:
        csv_file (str): Path to the CSV file containing image paths and radius information.
        output_dir (str, optional): Directory where the sorted CSV files will be saved. 
                                    Defaults to 'test'.

    Returns:
        None: This function does not return any value.
    """

    df = pd.read_csv(csv_file)

    radius_ranges = [(80, 60), (60, 40), (40, 0)]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for range_ in radius_ranges:

        filtered_df = df[(df['RADIUS'] < range_[0]) & (df['RADIUS'] >= range_[1])]
        

        filename = f"radius_{range_[1]}_to_{range_[0]}.csv"
        file_path = os.path.join(output_dir, filename)


        filtered_df.to_csv(file_path, index=False)

        print(f"Saved {len(filtered_df)} records to {file_path}")

def infer(network: torch.nn.Module, dataloader: DataLoader) -> Tuple[List, List]:
    """
    Performs inference using a neural network and a DataLoader, and returns predictions and labels.

    Args:
        network (torch.nn.Module): The neural network model to use for inference.
        dataloader (DataLoader): The DataLoader providing batches of data for inference.

    Returns:
        Tuple[List, List]: A tuple containing two lists - one for predictions and another for labels.
    """
    network.eval()  
    device = torch.device('cpu', 0) 
    predictions = []
    labels = []

    with torch.no_grad(): 
        for batch in dataloader:
            image, true_circle_params = batch
            
            image = image.to(device)
            
            predicted_circle_params = network(image)

            predictions.extend(predicted_circle_params.cpu().numpy())
            labels.extend(true_circle_params.numpy())

    return predictions, labels

def get_test_dataloader(csv_file, config, test=True):
    dataset = CircleDataset(config, csv_file, test=test)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader


def consolidate_results(iou_scores: Dict[str, List[float]],
                        radius_ranges: List[str]) -> pd.DataFrame:
    """
    Consolidates IoU scores from different radius ranges into a single DataFrame.

    Args:
        iou_scores (Dict[str, List[float]]): Dictionary mapping each radius range to a list of IoU scores.
        radius_ranges (List[str]): List of radius range strings.

    Returns:
        pd.DataFrame: A DataFrame containing the radius ranges and their corresponding average IoU scores.
    """
    ...
    combined_percentages = []
    for radius_range, scores in zip(radius_ranges, iou_scores.values()):
        mean_percentage = (sum(scores) / len(scores) * 100) if scores else 0
        combined_percentages.append({'Radius Range': radius_range, 'IoU Score (%)': mean_percentage})
    return pd.DataFrame(combined_percentages)



def main(threshold: float):
    with open('/Users/utkucicek/Desktop/SlingShotAI_Challenge/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    network = get_network(config, load_weights=True)
    
    csv_file = '/Users/utkucicek/Desktop/SlingShotAI_Challenge/dataset/test.csv'
    sort_and_save_images_by_radius(csv_file)

    csv_files = ['test/radius_60_to_80.csv', 
                 'test/radius_40_to_60.csv', 
                 'test/radius_0_to_40.csv']
    radius_ranges = ['80-60', '60-40', '40-0']
    
    results = {}
    accuracies = {}
    iou_scores = {}

    for csv_file in csv_files:
        dataloader = get_test_dataloader(csv_file, config)
        predictions, labels = infer(network, dataloader)
        results[csv_file] = (predictions, labels)
        iou_scores_per_file = [iou(pred, label, threshold) for pred, label in zip(predictions, labels)]
        iou_scores[csv_file] = iou_scores_per_file
        accuracies[csv_file] = sum(iou_scores_per_file) / len(iou_scores_per_file) if iou_scores_per_file else 0

    final_df = consolidate_results(iou_scores, radius_ranges)
    final_df.to_csv('combined_inference_results.csv', index=False)
    print(final_df)

if __name__ == "__main__":
    main(0.5)




