import os
from dadapy.metric_comparisons import MetricComparisons
import numpy as np
import torch
from utils.dataset_utils import load_dataset, create_subset
from utils.vae_utils import train_vae, test_vae, evaluate_vae, compute_vae_fid
from utils.analysing_utils import stack_and_save_layers


class VAE_Model:

    def __init__(self, model_kwargs={}):
        self.model_name = "vae"
        self.network = None 
        self.model_kwargs = model_kwargs
        self.sample_data_test = None
        self.sample_data_train = None
        self.layers_activations_path_list = None
        self.distances_path_list = None
        self.optimizer = None        

    def train(
        self,
        dataset_name,
        num_epochs,
        num_samples=None,
        batch_size=256,
    ):
        """
        Train the VAE model on the specified dataset.
        - dataset_name: Name of the dataset to use.
        - num_epochs: Number of epochs to train for.
        - num_samples: If specified, a subset of samples is created.
        - batch_size: Batch size for loading data.
        """
        if num_samples is None:
            train_loader, _ = load_dataset(
                dataset_name,
                local=False,
                batch_size=batch_size,
                type=self.model_name,
            )
        else:
            create_subset(dataset_name, num_samples)
            train_loader, _ = load_dataset(
                dataset_name,
                local=True,
                batch_size=batch_size,
                type=self.model_name,
            )

        self.network, self.optimizer, loss = train_vae(
            train_loader,
            num_epochs,
            vae=self.network,
            optimizer=self.optimizer,
            **self.model_kwargs,
            )
        
        return loss

    def load_network(self, model_name="", network_folder = "./data/networks/"):
        """
        Load a pre-trained VAE model from the specified folder.
        - model_name: Specific name of the model to load.
        - network_folder: Folder path where the model is stored.
        """
        model_path = (
            network_folder
            + self.model_name
            + "/model_"
            + self.model_name
            + "_"
            + model_name
            + ".pth"
        )
        if os.path.exists(model_path):
            self.network = torch.load(model_path, map_location=torch.device('cpu'))
            print(f"Model loaded successfully from {model_path}")
        else:
            print(f"No model found at {model_path}. Cannot load.")
        # return self.network
    
    def save_network(self, model_name="", network_folder = "./data/vae_networks/"):
        """
        Save the trained VAE model to the specified folder.
        - model_name: Specific name to save the model as.
        - network_folder: Folder path to save the model in.
        """
        saving_path = os.path.join(network_folder, self.model_name)
        os.makedirs(saving_path, exist_ok=True)
        torch.save(
            self.network,
            saving_path + "/model_" + self.model_name + "_" + model_name + ".pth",
        )
        print(f"Model saved successfully at {saving_path}")

    def create_sample_data(self, dataset_name, num_samples, batch_size=256, redim=32, save_path="./data"):
        """
        Create sample data by extracting a subset of the dataset.
        - dataset_name: Name of the dataset.
        - num_samples: Number of samples to create.
        - redim: Resize the dataset images.
        - save_path: Path to save the generated sample data.
        """
        create_subset(
            dataset_name,
            num_samples,
            redim=redim,
            save_path=save_path
        )
                        
        self.load_sample_data(dataset_name, batch_size=batch_size, redim=redim, subset_path=save_path)
        print(f"Sample data for {dataset_name} created successfully!")
        
    def load_sample_data(self, dataset_name, batch_size=256, redim=32, subset_path="./data"):
        """
        Load sample data from the provided dataset.
        - dataset_name: Name of the dataset to load.
        - batch_size: Batch size for data loading.
        - redim: Image resizing dimensions.
        - subset_path: Path where the subset of data is stored.
        """
        self.sample_data_train, self.sample_data_test = load_dataset(
        dataset_name,
        batch_size=batch_size,
        local=True,
        redim=redim,
        subset_path=subset_path
        )
        
    def delete_sample_data(self, dataset_name, save_path="./data/subsets"):
        """
        Delete stored sample data files.
        - dataset_name: Name of the dataset whose data should be deleted.
        - save_path: Path where the sample data is stored.
        """
        if os.path.exists(f"{save_path}/{dataset_name}-test.pth"):
            os.remove(f"{save_path}/{dataset_name}-test.pth")
        else:
            print(
                f"Warning: No file found for {dataset_name}-test.pth, skipping deletion"
            )
        
    def evaluate(self, save_dir = "./data/vae_activations", train = False, **kwargs):
        """
        Evaluate the VAE model and compute the activations of the layers.
        - save_dir: Directory where activation outputs are saved.
        - train: Whether to evaluate on training data or test data.
        """
        if train == True:
            sample_data = self.sample_data_train
        else : 
            sample_data = self.sample_data_test
       
        activations_path = test_vae(self.network, sample_data, **kwargs)

        layers_activations_path_list = stack_and_save_layers(
            activations_path, self.model_name, save_dir=save_dir
        )
        self.layers_activations_path_list = layers_activations_path_list

    
    def get_vae_loss(self, train = False):
        """
        Compute the VAE loss on the training or test dataset.
        - train: If True, computes the loss on training data, else on test data.
        """
        sample_data = self.sample_data_train if train else self.sample_data_test
        return evaluate_vae(self.network, sample_data)
    
    def compute_vae_fid(self, train = False):
        """
        Compute the Frechet Inception Distance (FID) score for the VAE.
        - train: If True, compute the FID on training data, else on test data.
        """
        sample_data = self.sample_data_train if train else self.sample_data_test
        return compute_vae_fid(self.network, sample_data)
    
    def compute_distances(self, save_path="./data"):
        """
        Compute distance metrics for the stored layer activations.
        - save_path: Path to save computed distance metrics.
        """
        assert (
            self.layers_activations_path_list is not None
        ), "No activations have been loaded. Please run evaluate() first."
        distances_path_list = []
        for i in range(len(self.layers_activations_path_list)):

            layer_i = np.array(
                np.load(
                    self.layers_activations_path_list[i]
                ), dtype=np.float32
            )
            d_i = MetricComparisons(
                layer_i, maxk=layer_i.shape[0] - 1
            )
            d_i.compute_distances()

            if not os.path.exists(f"{save_path}/vae_distances/distances"):
                os.makedirs(f"{save_path}/vae_distances/distances")
      
            np.savez(
                f"{save_path}/vae_distances/distances/distances_{i}.npz",
                distances=d_i.distances,
                dist_indices=d_i.dist_indices,
            )
            distances_path_list.append(
                f"{save_path}/vae_distances/distances/distances_{i}.npz"
            )
        print(f"Distances computed and saved successfully at {save_path}/vae_distances/distances")
        self.distances_path_list = distances_path_list