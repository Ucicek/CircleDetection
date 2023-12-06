import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple

from data.generator import generate_examples
from data.dataloader import CircleDataset, CircleDatasetGenerator
from networks.model import AttentionNetwork, Network

def get_custom_optimizer(config: dict, network: nn.Module) -> torch.optim.Optimizer:
    """
    Create a custom optimizer for a neural network with distinct parameter groups for biases, 
    batch normalization weights, and other weights, based on the configuration.

    Args:
        config (dict): Configuration dictionary containing training hyperparameters 
                       and optimizer specifications under 'TRAIN' and 'OPTIMIZER_HYP' keys.
        network (nn.Module): The neural network for which to configure the optimizer.

    Returns:
        torch.optim.Optimizer: Configured optimizer with separate parameter groups.
    """
    hyp = config['TRAIN']['OPTIMIZER_HYP']
    biases, batchnorm_weights, other_weights = [], [], []
    
    for module in network.modules():
        if hasattr(module, 'bias') and isinstance(module.bias, nn.Parameter):
            biases.append(module.bias)
        if isinstance(module, nn.BatchNorm2d):
            batchnorm_weights.append(module.weight)
        elif hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter):
            other_weights.append(module.weight)

    if hyp['OPTIMIZER'].lower() == 'adam':
        # Use Adam optimizer for batchnorm_weights without weight decay
        optimizer = torch.optim.Adam(batchnorm_weights, lr=hyp['LR0'], betas=(hyp['MOMENTUM'], 0.999))
    else:
        # Use SGD with Nesterov momentum for batchnorm_weights without weight decay
        optimizer = torch.optim.SGD(batchnorm_weights, lr=hyp['LR0'], momentum=hyp['MOMENTUM'], nesterov=True)

    optimizer.add_param_group({'params': other_weights, 'weight_decay': hyp['WEIGHT_DECAY']})
    optimizer.add_param_group({'params': biases})

    # Clear temporary lists to free memory
    del biases, batchnorm_weights, other_weights

    return optimizer

def get_optimizer(config: dict, network: nn.Module) -> torch.optim.Optimizer:
    """
    Create an optimizer for a neural network using provided configuration settings.

    Args:
        config (dict): Configuration dictionary containing optimizer hyperparameters 
                       and learning rate settings under 'TRAIN' and 'OPTIMIZER_HYP' keys.
        network (nn.Module): The neural network for which to configure the optimizer.

    Returns:
        torch.optim.Optimizer: Configured optimizer based on the 'OPTIMIZER' type in config.
    """

    train_hyp = config['TRAIN']['OPTIMIZER_HYP']
    lr = train_hyp.get('LR0', train_hyp['LR1'])  # Use 'LR0' if it exists, otherwise fall back to 'LR1'

    # Choose the optimizer based on the 'optimizer' setting from configuration
    if train_hyp['OPTIMIZER'].lower() == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    elif train_hyp['OPTIMIZER'].lower() == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=train_hyp['MOMENTUM'])

    return optimizer




def get_dataloader(config: dict, train: bool = True, generate: bool = True) -> DataLoader:
    """
    Create and return a DataLoader for the CircleDataset. This function supports creating DataLoaders 
    for either generated datasets or datasets loaded from a CSV file, based on the provided configuration.
    It also allows the creation of DataLoaders for either training or validation/test data.

    Args:
        config (dict): Configuration dictionary containing parameters for dataset creation 
                       (noise level, image size, min/max radius), DataLoader settings (batch size), 
                       and file paths for CSV data.
        train (bool, optional): Flag to specify whether to create a DataLoader for training 
                                or testing/validation. Defaults to True for training.
        generate (bool, optional): Flag to specify whether to generate a dataset using a 
                                   generator function or load from a CSV file. Defaults to True for generation.

    Returns:
        DataLoader: Configured DataLoader for the specified dataset, based on the specified conditions.
    """

    noise_level = config['DATA']['NOISE']
    img_size = config['DATA']['IMG_SIZE']
    min_radius = config['DATA']['MIN_RADIUS']
    max_radius = config['DATA']['MAX_RADIUS']
    batch_size = config['TRAIN']['BATCH_SIZE']

    if generate:
        genarator = generate_examples(noise_level, img_size, min_radius, max_radius)

        dataset = CircleDatasetGenerator(config, genarator)
        generate_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return generate_dataloader
    else:
        if train:
            csv_file = config['DATA']['DATA_DIR']['TRAIN']
        else:
            csv_file = config['DATA']['DATA_DIR']['TEST']

    dataset = CircleDataset(config,csv_file)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def get_network(config: dict, load_weights: bool = False) -> nn.Module:
    """
    Create a neural network based on the specified type in the configuration. Optionally load pretrained weights.

    Args:
        config (dict): Configuration dictionary containing network type and optionally a path 
                       to pretrained weights under 'TRAIN' and 'TEST' keys.
        load_weights (bool, optional): Flag to specify whether to load pretrained weights. 
                                       Defaults to False.

    Returns:
        nn.Module: The neural network, loaded onto the appropriate device (GPU or CPU).
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    network_type = config['TRAIN']['NETWORK']
    pretrained_weights_path = config['TEST'].get('WEIGHT_PATH', None)

    # Define or load your network here based on network_type
    if network_type == 'Attention':
        network = AttentionNetwork()  # Replace with your network class
    elif network_type == 'BasicCNN':
        network = Network()
    else:
        raise ValueError(f"Unsupported network type: {network_type}")

    # Load pretrained weights if path is provided and exists
    if load_weights and os.path.exists(pretrained_weights_path):
        network.load_state_dict(torch.load(pretrained_weights_path))
        print("Loaded pretrained weights from", pretrained_weights_path)
    else:
        print("No pretrained weights provided or file does not exist, initializing network with random weights.")

    return network.to(device)


def iou(circle1: Tuple[int, int, int], circle2: Tuple[int, int, int], threshold: float = 0.5) -> float:
    """
    Calculate the Intersection over Union (IoU) of two circles and compare it with a given threshold.

    Args:
        circle1 (Tuple[int, int, int]): Parameters [radius, row, col] for the first circle.
        circle2 (Tuple[int, int, int]): Parameters [radius, row, col] for the second circle.
        threshold (float, optional): Threshold for the IoU score to be considered significant. 
                                     Defaults to 0.5.

    Returns:
        float: IoU score if it's above the threshold, otherwise 0.
    """
    r1, r2 = circle1[0], circle2[0]
    d = np.linalg.norm(np.array(circle1[1:]) - np.array(circle2[1:]))
    if d > r1 + r2:
        return 0.0  # No overlap
    if d <= abs(r1 - r2):
        # One circle is inside the other
        larger_r, smaller_r = max(r1, r2), min(r1, r2)
        iou_score = smaller_r ** 2 / larger_r ** 2
        return iou_score if iou_score >= threshold else 0

    # Overlapping circles
    r1_sq, r2_sq = r1**2, r2**2
    d1 = (r1_sq - r2_sq + d**2) / (2 * d)
    d2 = d - d1
    sector_area1 = r1_sq * np.arccos(d1 / r1)
    triangle_area1 = d1 * np.sqrt(r1_sq - d1**2)
    sector_area2 = r2_sq * np.arccos(d2 / r2)
    triangle_area2 = d2 * np.sqrt(r2_sq - d2**2)
    intersection = sector_area1 + sector_area2 - (triangle_area1 + triangle_area2)
    union = np.pi * (r1_sq + r2_sq) - intersection
    iou_score = intersection / union

    return 1 if iou_score >= threshold else 0






