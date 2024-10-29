# VAEs-through-ID-and-II
Code to reproduce the results of the paper "Understanding Variational Autoencoders with Intrinsic Dimension and Information Imbalance"

## Prerequisites
Ensure that you have Python installed (Python 3.11 or later is recommended) and install the necessary dependencies using the following command:

pip install -r requirements.txt 
 
## VAE Model Training Script

### Usage
The script vae_training.py trains a VAE model on a selected dataset, saving model checkpoints and logging losses as specified in the input arguments.

### Basic Command
To run the script, go into the vae_framework folder and use:

python vae_training.py --dataset <dataset_name> --epochs <number_of_epochs>

### Arguments
--dataset: (required) The dataset on which to train the VAE model. Options: cifar10 or mnist.
--epochs: (required) Number of epochs to train the model.
--save_interval: Specifies at which epochs to save the model checkpoints. Default is [1, 2, 5, 10, 20, 50, 75, 100, 150, 200].
--latent_dim: Sets the dimension(s) of the latent space. Default is [2, 4, 8, 16, 32, 64, 128].
--batch_size: Defines the batch size for training. Default is 256.

### Example Command
To train the model on the cifar10 dataset for 100 epochs with a batch size of 128 and latent dimensions of 16 and 32, saving checkpoints for epoch 1, 50, 100 and 200 epochs, run:

python vae_training.py --dataset cifar10 --epochs 100 --batch_size 128 --latent_dim 16 32 --save_interval 1 50 100 200

## VAE Analysis Functions
After training, you can use additional functions to analyze the saved networks. These functions are designed to compute and save various metrics, such as distances between network states, intrinsic dimensions, and imbalance metrics across layers. Each function outputs results as .json files for easy data handling and interpretation.

### Analysis Commands
get_all_distances: Computes and saves the distances between layers of the saved networks. Distances are stored as JSON files for further analysis.

get_all_ids: Computes the intrinsic dimensions for each layers of the trained networks. Results are saved in JSON format.

get_all_imbalances_first_last: Calculates and saves the imbalance between the first and last layers of each trained network. Results are stored in JSON files.

get_all_imbalances_train: Computes the imbalance between consecutive layers (i to i+1) within each network. Results are saved as JSON files.

get_all_imbalances_inter_dims: Calculates the imbalance across networks with different latent dimensions. Results are stored in JSON format.

## Disclaimer

This package is an outcome of a research project. All errors are those of
the authors. All views expressed are personal views, not those of Bank of Italy.


