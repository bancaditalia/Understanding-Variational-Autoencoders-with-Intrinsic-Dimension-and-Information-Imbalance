## Standard libraries
import os
import math
import time
import numpy as np

## Imports for plotting
import matplotlib.pyplot as plt

# from IPython.display import set_matplotlib_formats

# set_matplotlib_formats("svg", "pdf")  # For export
from matplotlib.colors import to_rgb
import matplotlib

matplotlib.rcParams["lines.linewidth"] = 2.0
import seaborn as sns

sns.reset_orig()

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torchvision import datasets
import os

# Torchvision
import torchvision
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler


def discretize(sample):
    return (sample * 255).to(torch.int32)


def create_subset(
    dataset_name, num_samples, redim=32, save_path="./data"
):
    assert dataset_name in [
        "mnist",
        "cifar10",
    ], "Dataset name must be 'mnist' or 'cifar10'."

    # create save_path if it does not exist
    if not os.path.exists(save_path+"/subsets"):
        os.makedirs(save_path+"/subsets")

    transform_list = [transforms.Resize((redim, redim)), transforms.ToTensor()]
    transform = transforms.Compose(transform_list)

    subset_train_path = os.path.join(save_path, f"subsets/{dataset_name}-train.pth")
    subset_test_path = os.path.join(save_path, f"subsets/{dataset_name}-test.pth")
    
    # Check if files already exist
    if os.path.exists(subset_train_path) and os.path.exists(subset_test_path):
        print(
            f"Subset of {dataset_name}-train and {dataset_name}-test datasets already exist."
            "Skipping creation.."
        )
        return

    # Create the specified dataset
    if dataset_name == "mnist":
        dataset_train = MNIST(
            root=save_path, train=True, download=True, transform=transform
        )
        dataset_test = MNIST(
            root=save_path, train=False, download=True, transform=transform
        )
    elif dataset_name == "cifar10":
        dataset_train = CIFAR10(
            root=save_path, train=True, download=True, transform=transform
        )
        dataset_test = CIFAR10(
            root=save_path, train=False, download=True, transform=transform
        )
    
    # Create a Subset using these random indices
    subset_train = torch.utils.data.Subset(dataset_train, range(num_samples))#subset_indices)
    subset_test = torch.utils.data.Subset(dataset_test, range(num_samples))
    
    # Save the subset to a file
    torch.save(subset_train, subset_train_path)
    print(
        f"Subset of {dataset_name}-train dataset containing {num_samples} samples saved to {subset_train_path}"
    )
        
    torch.save(subset_test, subset_test_path)
    print(
        f"Subset of {dataset_name}-test dataset containing {num_samples} samples saved to {subset_test_path}"
    )


def load_dataset(
    dataset_name,
    local=True,
    redim=32,
    subset_path="./data",
    batch_size=64,
    random_seed=42,
):
    assert dataset_name in [
        "mnist",
        "cifar10",
    ], "Dataset name must be 'mnist' or 'cifar10'."
    assert not (
        local and subset_path is None
    ), "If loading locally, subset_path must be provided."
    train_loader, val_loader, test_loader = None, None, None
    transform_list = [transforms.Resize((redim, redim)), transforms.ToTensor()]
    transform = transforms.Compose(transform_list)
    
    
    load_path_train = os.path.join(subset_path,"subsets" ,dataset_name + "-train.pth")
        
    load_path_test = os.path.join(subset_path,"subsets" ,dataset_name + "-test.pth")

    if local:
        # Try to load the subset dataset from the file
        try:
            subset_test = torch.load(load_path_test)
            print(f"Subset of {dataset_name} dataset loaded from {load_path_test}.")
        except FileNotFoundError:
            print(f"Subset of {dataset_name} dataset not found at {load_path_test}.")
        try:
            subset_train = torch.load(load_path_train)
            print(f"Subset of {dataset_name} dataset loaded from {load_path_train}.")
        except FileNotFoundError:
            print(f"Subset of {dataset_name} dataset not found at {load_path_train}.")


    if not local:
        # Load the entire dataset from torchvision
        if dataset_name == "mnist":
            dataset_train = datasets.MNIST(
                root=subset_path, train=True, download=True, transform=transform
            )
            dataset_test = datasets.MNIST(
                root=subset_path, train=False, download=True, transform=transform
            )
        elif dataset_name == "cifar10":
            dataset_train = datasets.CIFAR10(
                root=subset_path, train=True, download=True, transform=transform
            )
            dataset_test = datasets.CIFAR10(
                root=subset_path, train=False, download=True, transform=transform
            )

    if local:
        data_train = subset_train
        data_test = subset_test
    else:
        data_train = dataset_train
        data_test = dataset_test
        
    seed = random_seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    dataset_size = len(data_train)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)  # Shuffle indices

    # Define the size of the subset (80% of the data)
    subset_size = int(0.8 * dataset_size)
    subset_indices = indices[:subset_size]

    # save the indices to a txt file
    with open(os.path.join(subset_path, f"subsets/{dataset_name}_indices.txt"), "w") as f:
        for idx in subset_indices:
            f.write(f"{idx}\n")
    
    # Create a SubsetRandomSampler for the random subset
    subset_sampler = SubsetRandomSampler(subset_indices)

    # Create the DataLoader using the SubsetRandomSampler
    train_loader = DataLoader(data_train, sampler=subset_sampler, batch_size=batch_size)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
