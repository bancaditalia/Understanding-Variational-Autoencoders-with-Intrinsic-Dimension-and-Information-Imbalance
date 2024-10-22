import numpy as np
import matplotlib.pyplot as plt
from dadapy import Data

global_rng = np.random.default_rng()

epochs = [0, 1, 2, 5, 10, 20, 50, 100, 150, 200]
dims = [2, 4, 8, 16, 32, 64, 128]
layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def get_data(epoch, dim, layer):
    filename = f"data/vae_distances/epoch_{epoch}_dim_{dim}/vae/distances/distances_{layer}.npz"
    data = np.load(filename)
    return data

def get_id(epoch, dim, layer):
    print("dim: ", dim, "epoch: ", epoch, "layer: ", layer)
    data = get_data(epoch, dim, layer)
    d = Data(distances=(data["distances"], data["dist_indices"]))
    return d.compute_id_2NN()[0]

# get all IDs
from joblib import Parallel, delayed

# first get all combinations of dims, epochs and layers
combinations = [(dim, epoch, layer) for dim in dims for epoch in epochs for layer in layers]

# then parallelize the computation
IDs = Parallel(n_jobs=4)(delayed(get_id)(epoch, dim, layer) for dim, epoch, layer in combinations)

# unpack the IDs and store them in a dictionary
IDs_dict = {}
for i, (dim, epoch, layer) in enumerate(combinations):
    IDs_dict["dim_{}_epoch_{}_layer{}".format(dim, epoch, layer)] = IDs[i]
    
# save the dictionary to a json file
import json

with open("IDs.json", "w") as f:
    json.dump(IDs_dict, f)
    
# load the dictionary from the json file

with open("IDs.json", "r") as f:
    IDs_dict = json.load(f)
    