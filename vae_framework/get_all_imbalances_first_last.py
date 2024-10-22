import numpy as np
import matplotlib.pyplot as plt
from dadapy import Data
from joblib import Parallel, delayed
import json

global_rng = np.random.default_rng()

from dadapy._utils.metric_comparisons import _return_imbalance

# general variables to analyse

epochs = [0, 1, 2, 5, 10, 20, 50, 100, 150, 200]
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


def get_IIs_fist_last(epoch, dim, distances=distances):

    II_to_last = []
    II_to_first = []

    layer = layers[-1]
    data = distances[f"dim_{dim}_epoch_{epoch}_layer_{layer}"] #get_data(epoch, dim, layers[-1])
    dist_indices_last = data["dist_indices"]

    layer = layers[0]
    data = distances[f"dim_{dim}_epoch_{epoch}_layer_{layer}"] #get_data(epoch, dim, layers[0])
    dist_indices_first = data["dist_indices"]

    for layer in layers:
        print("dim: ", dim, "epoch: ", epoch, "layer: ", layer)
        data = distances[f"dim_{dim}_epoch_{epoch}_layer_{layer}"]#get_data(epoch, dim, layer)
        dist_indices = data["dist_indices"]

        II_to_last.append(_return_imbalance(dist_indices, dist_indices_last, rng=global_rng))
        II_to_first.append(_return_imbalance(dist_indices, dist_indices_first, rng=global_rng))
        
    return II_to_first, II_to_last


# loop over all epochs and dims

# combinations
combinations = [(dim, epoch) for dim in dims for epoch in epochs]

# parallelize the computation
IIs_results = Parallel(n_jobs=8, prefer="threads")(delayed(get_IIs_fist_last)(epoch, dim, distances) for dim, epoch in combinations)

# unpack the IDs and store them in a dictionary
IIs = {}
for i, (dim, epoch) in enumerate(combinations):
    IIs["dim_{}_epoch_{}".format(dim, epoch)] = {}
    IIs["dim_{}_epoch_{}".format(dim, epoch)]["to_first"] = IIs_results[i][0]
    IIs["dim_{}_epoch_{}".format(dim, epoch)]["to_last"] = IIs_results[i][1]
    
# save the dictionary to a json file
with open("IIs.json", "w") as f:
    json.dump(IIs, f)
    
# load the dictionary from the json file [Not needed..]
with open("IIs.json", "r") as f:
    IIs = json.load(f)