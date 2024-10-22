import numpy as np
from dadapy import Data

# add current directory to path
import sys
import os
sys.path.append(os.getcwd())

from vae_class import VAE_Model
# get all layers
model = VAE_Model()

# generate common dataset
model.load_sample_data("cifar10", subset_path='./vae_framework/')

for epoch in [0, 1, 2, 5, 10, 20, 50, 75, 100, 150, 200]:
    for dim in [2, 4, 8, 16, 32, 64, 128]:    
        model.load_network("epoch_" + str(epoch) + "_dim_" + str(dim) , network_folder = './vae_framework/data/vae_networks/')
        model.evaluate()
        model.compute_distances(save_path='./vae_framework/data/vae_distances/epoch_' + str(epoch) + '_dim_' + str(dim) + '/')
