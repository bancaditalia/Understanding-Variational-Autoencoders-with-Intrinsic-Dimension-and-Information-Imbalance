import numpy as np
import matplotlib.pyplot as plt
from dadapy import Data
from joblib import Parallel, delayed
import json

global_rng = np.random.default_rng()

from dadapy._utils.metric_comparisons import _return_imbalance

# general variables to analyse

epochs = [1, 2, 5, 10, 20, 50, 100, 150, 200]
dims = [2, 4, 8, 16, 32, 64, 128]
layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def get_data(epoch, dim, layer):
    filename = f"data/vae_distances/epoch_{epoch}_dim_{dim}/vae/distances/distances_{layer}.npz"
    data = np.load(filename)
    return data

# for all dims, epochs and layers, get the distances and store them in a dictionary

distances = {}

for dim in dims:
    print("loading dim: ", dim)
    for epoch in epochs:
        for layer in layers:
            distances[f"dim_{dim}_epoch_{epoch}_layer_{layer}"] = get_data(epoch, dim, layer)


def get_cross_IIs_(dim1, dim2):
    
    epoch = epochs[-1]
    IIs_inter_dims = np.zeros((len(layers), len(layers)))
    
    for i, layer1 in enumerate(layers):
        for j, layer2 in enumerate(layers):

            print("dim1: ", dim1, "dim2: ", dim2, "layer1: ", layer1, "layer2: ", layer2)
            
            dist_indices1 = distances[f"dim_{dim1}_epoch_{epoch}_layer_{layer1}"]["dist_indices"]
            dist_indices2 = distances[f"dim_{dim2}_epoch_{epoch}_layer_{layer2}"]["dist_indices"]
            
            II1_2 = _return_imbalance(dist_indices1, dist_indices2, rng=global_rng)
            IIs_inter_dims[i, j] = II1_2

    
    return IIs_inter_dims

for i in range(len(dims)):
    for j in range(i+1):
        dim1 = dims[i]
        dim2 = dims[j]
        print("dim1: ", dim1, "dim2: ", dim2)
        IIs_inter_dims = get_cross_IIs_(dim1, dim2)
        np.save(f"dim_{dim1}_dim_{dim2}_cross_IIs.np", IIs_inter_dims)
        

    