import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
import math
import os
import torch.nn as nn
from torchvision.models import inception_v3
from torchvision.transforms import transforms
from scipy.linalg import sqrtm
from PIL import Image

def stack_and_save_layers(activation_path_list, model_name, save_dir = "./data", batch_size=None):
    output_dir = os.path.join(save_dir, model_name, "activations")
    layers_activations_path_list = []
    # Create output directory if it doesn't exist

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = activation_path_list[0]
    loaded_data = torch.load(file_path, map_location=torch.device("cpu"))
    for i in range(len(loaded_data)):
        # Initialize stacked layers with empty lists for each layer
        stacked_layers = []

        # Loop over all files and stack corresponding layers
        for file_path in activation_path_list:
            loaded_data = torch.load(file_path, map_location=torch.device("cpu"))
            layer_data = loaded_data[i]
            stacked_layers.extend(layer_data)

        # Save stacked layers to a separate file
        output_file = os.path.join(
            output_dir, f"{model_name}_layer_{i + 1}_activation.npy"
        )
        
        # convert stacked_layers to CPU numpy
        stacked_layers = np.array(stacked_layers)
        
        if batch_size is None:
            batch_size = len(stacked_layers)
                
        stacked_layers = np.array(stacked_layers.reshape(batch_size, -1).tolist())
        print(f"Stacked layers shape: {stacked_layers.shape}")
        np.save(output_file, stacked_layers)
        layers_activations_path_list.append(output_file)
        print(f"Stacked layers {output_file} saved successfully!")
    print("Deleting the original batch files...")

    for file_path in activation_path_list:
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error deleting file {file_path}: {e}")
            raise e
    print("All original batch files deleted successfully!")

    return layers_activations_path_list

def numpy_to_pil(img_array):
    """Convert a numpy array to a PIL Image."""
    if img_array.ndim == 3 and img_array.shape[2] == 3:
        img_array = np.transpose(img_array, (1, 2, 0))  # Convert (C, H, W) to (H, W, C)
    elif img_array.ndim == 3 and img_array.shape[0] == 3:
        img_array = np.transpose(img_array, (1, 2, 0))  # Convert (C, H, W) to (H, W, C)
    if img_array.max() > 1.0:
        img_array = img_array / 255.0  # Normalize to [0, 1] if necessary
    return Image.fromarray((img_array * 255).astype(np.uint8))

def calculate_fid(real_images, generated_images):
    
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = nn.Identity()  # Remove the final classification layer
    model.eval()

    # Transformation to match CIFAR-10 dimensions and normalization
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    real_images = torch.stack([transform(numpy_to_pil(img)) for img in real_images])
    generated_images = torch.stack([transform(numpy_to_pil(img)) for img in generated_images])
    
    with torch.no_grad():
        act1 = model(real_images).detach().cpu().numpy()
        act2 = model(generated_images).detach().cpu().numpy()
    
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid