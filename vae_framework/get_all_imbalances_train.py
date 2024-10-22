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


def get_IIs_l1_l2(epoch, dim, distances=distances):

    II_l2_to_l1 = []
    II_l1_to_l2 = []
    for i in range(len(layers)-1):
        layer1 = layers[i]
        data = distances[f"dim_{dim}_epoch_{epoch}_layer_{layer1}"] #get_data(epoch, dim, layers[l])
        dist_indices_l1 = data["dist_indices"]

        layer2 = layers[i+1]
        data = distances[f"dim_{dim}_epoch_{epoch}_layer_{layer2}"] #get_data(epoch, dim, layers[l+1])
        dist_indices_l2 = data["dist_indices"]

        print("dim: ", dim, "epoch: ", epoch, "layer_l: ", layer1, "layer_l+1: ", layer2)
        
        # II_from_last.append(_return_imbalance(dist_indices_last, dist_indices, rng=global_rng))
        II_l2_to_l1.append(_return_imbalance(dist_indices_l2, dist_indices_l1, rng=global_rng))
        II_l1_to_l2.append(_return_imbalance(dist_indices_l1, dist_indices_l2, rng=global_rng))
            
    return II_l2_to_l1, II_l1_to_l2


# loop over all epochs and dims

# combinations
combinations = [(dim, epoch) for dim in dims for epoch in epochs]

# parallelize the computation
IIs_results = Parallel(n_jobs=8, prefer="threads")(delayed(get_IIs_l1_l2)(epoch, dim, distances) for dim, epoch in combinations)

# unpack the IDs and store them in a dictionary
IIs = {}
for i, (dim, epoch) in enumerate(combinations):
    IIs["dim_{}_epoch_{}".format(dim, epoch)] = {}
    IIs["dim_{}_epoch_{}".format(dim, epoch)]["l2_to_l1"] = IIs_results[i][0]
    IIs["dim_{}_epoch_{}".format(dim, epoch)]["l1_to_l2"] = IIs_results[i][1]
    
# save the dictionary to a json file
with open("IIs_train.json", "w") as f:
    json.dump(IIs, f)
    
# load the dictionary from the json file [Not needed..]
with open("IIs_train.json", "r") as f:
    IIs = json.load(f)