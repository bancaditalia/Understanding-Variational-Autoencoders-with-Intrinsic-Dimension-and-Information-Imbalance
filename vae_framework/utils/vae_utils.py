import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
import math
import pytorch_lightning as pl
import os
import time
import shutil
from utils.analysing_utils import calculate_fid

class Encoder(nn.Module):
    def __init__(
        self, input_channels, input_size, latent_dim, hidden_dims_factors=[2, 4, 8, 8]
    ):
        self.hidden_dims_factors = hidden_dims_factors
        #  hidden_dims=[64, 128, 256]):
        super(Encoder, self).__init__()
        self.input_channels = input_channels
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.scale_factor = self.input_size // (2 ** len(self.hidden_dims_factors))
        self.hidden_dims = np.array(self.hidden_dims_factors) * self.input_size
        self.conv1 = nn.Conv2d(
            self.input_channels, self.hidden_dims[0], kernel_size=4, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            self.hidden_dims[0], self.hidden_dims[1], kernel_size=4, stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            self.hidden_dims[1], self.hidden_dims[2], kernel_size=4, stride=2, padding=1
        )
        self.conv4 = nn.Conv2d(
            self.hidden_dims[2], self.hidden_dims[3], kernel_size=4, stride=2, padding=1
        )
        self.fc_mean = nn.Linear(
            self.hidden_dims[3] * self.scale_factor * self.scale_factor, self.latent_dim
        )
        self.fc_log_var = nn.Linear(
            self.hidden_dims[3] * self.scale_factor * self.scale_factor, self.latent_dim
        )
        self.bn1 = nn.BatchNorm2d(self.hidden_dims[0])
        self.bn2 = nn.BatchNorm2d(self.hidden_dims[1])
        self.bn3 = nn.BatchNorm2d(self.hidden_dims[2])
        self.bn4 = nn.BatchNorm2d(self.hidden_dims[3])

    def forward(self, x, intermediate_outputs=None):
        if intermediate_outputs is None:
            intermediate_outputs = []
        intermediate_outputs.append(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        intermediate_outputs.append(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        intermediate_outputs.append(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        intermediate_outputs.append(x)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        intermediate_outputs.append(x)
        x = x.view(-1, self.hidden_dims[3] * self.scale_factor * self.scale_factor)
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        return z_mean, z_log_var


class Decoder(nn.Module):
    def __init__(
        self,
        output_channels,
        output_size,
        latent_dim,
        hidden_dims_factors=[8, 8, 4, 2],
    ):
        super(Decoder, self).__init__()
        self.hidden_dims_factors = hidden_dims_factors
        self.output_channels = output_channels
        self.output_size = output_size
        self.latent_dim = latent_dim
        self.scale_factor = self.output_size // (2 ** len(self.hidden_dims_factors))
        self.hidden_dims = np.array(self.hidden_dims_factors) * self.output_size

        self.fc1 = nn.Linear(
            self.latent_dim, self.hidden_dims[0] * self.scale_factor * self.scale_factor
        )
        self.conv1 = nn.ConvTranspose2d(
            self.hidden_dims[0],
            self.hidden_dims[1],
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0,
        )
        self.conv2 = nn.ConvTranspose2d(
            self.hidden_dims[1],
            self.hidden_dims[2],
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0,
        )
        self.conv3 = nn.ConvTranspose2d(
            self.hidden_dims[2],
            self.hidden_dims[3],
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0,
        )
        self.conv4 = nn.ConvTranspose2d(
            self.hidden_dims[3],
            self.output_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0,
        )
        self.bn_fc = nn.BatchNorm1d(
            self.hidden_dims[0] * self.scale_factor * self.scale_factor
        )
        self.bn1 = nn.BatchNorm2d(self.hidden_dims[1])
        self.bn2 = nn.BatchNorm2d(self.hidden_dims[2])
        self.bn3 = nn.BatchNorm2d(self.hidden_dims[3])

    def forward(self, x, intermediate_outputs=None):
        if intermediate_outputs is None:
            intermediate_outputs = []
        intermediate_outputs.append(x)
        x = F.leaky_relu(self.bn_fc(self.fc1(x)), 0.2)
        intermediate_outputs.append(x)
        x = x.view(-1, self.hidden_dims[0], self.scale_factor, self.scale_factor)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        intermediate_outputs.append(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        intermediate_outputs.append(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        intermediate_outputs.append(x)
        x = torch.sigmoid(self.conv4(x))
        intermediate_outputs.append(x)
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(
        self, channels=3, size=32, latent_dim=128, hidden_dims_factors=[2, 4, 8, 8]
    ):
        super().__init__()
        self.channels = channels
        self.size = size
        self.latent_dim = latent_dim
        self.hidden_dims_factors = hidden_dims_factors
        self.scale_factor = self.size // (2 ** len(self.hidden_dims_factors))
        self.encoder = Encoder(
            input_channels=self.channels,
            input_size=self.size,
            latent_dim=self.latent_dim,
            hidden_dims_factors=self.hidden_dims_factors,
        )
        self.decoder = Decoder(
            output_channels=self.channels,
            output_size=self.size,
            latent_dim=self.latent_dim,
            hidden_dims_factors=self.hidden_dims_factors[::-1],
        )
        self.intermediate_outputs = []

    def forward(self, x):
        self.intermediate_outputs = []
        z_mean, z_log_var = self.encoder(x, self.intermediate_outputs)
        z = self.reparameterize(z_mean, z_log_var)
        x_recon = self.decoder(z, self.intermediate_outputs)
        return x_recon, z_mean, z_log_var, self.intermediate_outputs

    def reparameterize(self, z_mean, z_log_var):
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

    def vae_loss(self, x, x_recon, z_mean, z_log_var):
        reconstruction_loss = F.binary_cross_entropy(
            x_recon.view(-1, self.channels, self.size, self.size),
            x.view(-1, self.channels, self.size, self.size),
            reduction="sum",
        )
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        return [reconstruction_loss, kl_loss]

    def generate(self, num_samples=10):
        sampling_activation = []
        with torch.no_grad():
            # Sample from a standard normal distribution
            z = torch.randn(num_samples, self.latent_dim).to(
                next(self.parameters()).device
            )
            # Decode the samples
            generated_samples = self.decoder(z, sampling_activation)
        return generated_samples, sampling_activation


def train_vae(
    train_loader,
    num_epochs=10,
    latent_dim=30,
    vae=None,
    optimizer=None,
):
    first_batch = next(iter(train_loader))
    first_image = first_batch[0][0]
    img_shape = first_image.shape
    channels, size = img_shape[0], img_shape[1]
    if vae is None:
        vae = VariationalAutoencoder(
            channels=channels, size=size, latent_dim=latent_dim
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else device

    vae.to(device)

    if optimizer is None:
        optimizer = torch.optim.Adam(vae.parameters(), weight_decay=1e-4)
    start_time = time.time()

    vae.train()
    loss = 0
    recon_loss = 0
    kl_loss = 0
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_recon_loss = 0.0 
        train_kl_loss = 0.0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, z_mean, z_log_var, _ = vae(data)
            recon_loss, kl_loss = vae.vae_loss(data, recon_batch, z_mean, z_log_var)
            loss = recon_loss + kl_loss
            loss.backward()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
            train_loss += loss.item()
            optimizer.step()
        train_recon_loss /= len(train_loader.dataset)
        train_kl_loss /= len(train_loader.dataset)
        train_loss /= len(train_loader.dataset)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Time: {time.time() - start_time:.2f}s"
        )
        loss = [train_loss, train_recon_loss, train_kl_loss]

    duration = time.time() - start_time

    print("Evaluation Results:")
    print(f"Training time : {duration}")
    print(len(train_loader.dataset))
    return vae, optimizer, loss


def evaluate_vae(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else device
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0
    i = 0.0
    with torch.no_grad():
        for data, _ in dataloader:
            i +=1
            data = data.to(device)
            recon_batch, z_mean, z_log_var, _ = model(data)
            reconstruction_loss, kl_loss = model.vae_loss(data, recon_batch, z_mean, z_log_var)
            loss = reconstruction_loss + kl_loss
            total_reconstruction_loss += reconstruction_loss.item() 
            total_kl_loss += kl_loss.item() 
            total_loss += loss.item() 
            
    total_reconstruction_loss /= len(dataloader.dataset)
    total_kl_loss /= len(dataloader.dataset)
    total_loss /= len(dataloader.dataset)
    return total_loss, reconstruction_loss, kl_loss

def compute_vae_fid(
    vae, 
    sample_loader
    ):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else device
    
    vae.eval()
    with torch.no_grad():
        for data, _ in sample_loader:
            data = data.to(device)
            recon_batch, _, _, _ = vae(data)
            data = data.cpu()
            recon_batch = recon_batch.cpu()
            fid = calculate_fid(data.numpy(), recon_batch.numpy())
    return fid


def test_vae(
    vae, sample_loader, mode="", activation_folder_path="./data/vae/activations/"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else device
    vae.to(device)
    vae.eval()
    activation_path_list = []

    if os.path.exists(activation_folder_path):
        shutil.rmtree(activation_folder_path)
        os.makedirs(activation_folder_path, exist_ok=True)
    else:
        os.makedirs(activation_folder_path, exist_ok=True)

    with torch.no_grad():
        if mode == "sample":
            for i, (data, _) in enumerate(sample_loader):
                generated_imgs, hidden_states = vae.generate(num_samples=len(data))
                filename = f"{activation_folder_path}hidden_states_{i}.pt"
                torch.save(hidden_states, filename)
                activation_path_list.append(filename)
                data = data.cpu()
                generated_imgs = generated_imgs.cpu()
                
            fig, axes = plt.subplots(1, 8, figsize=(8, 1))
            for i in range(8):
                # Check if the generated image is grayscale or RGB
                if generated_imgs[i].shape[0] == 1:
                    # Grayscale image
                    axes[i].imshow(
                        generated_imgs[i].squeeze().cpu().numpy(), cmap="gray"
                    )
                else:
                    # RGB image
                    axes[i].imshow(
                        np.transpose(generated_imgs[i].cpu().numpy(), (1, 2, 0))
                    )
                axes[i].axis("off")
            plt.show()

        else:
            for i, (data, _) in enumerate(sample_loader):
                data = data.to(device)

                recon_batch, _, _, hidden_states = vae(data)
                filename = f"{activation_folder_path}hidden_states_{i}.pt"
                torch.save(hidden_states, filename)
                activation_path_list.append(filename)
                data = data.cpu()
                recon_batch = recon_batch.cpu()

            plt.figure(figsize=(10, 2))
            for idx in range(5):
                plt.subplot(2, 5, idx + 1)
                # Check if the image is grayscale or RGB
                if data[idx].shape[0] == 1:
                    # Grayscale image
                    plt.imshow(data[idx].squeeze().numpy(), cmap="gray")
                else:
                    # RGB image
                    plt.imshow(np.transpose(data[idx], (1, 2, 0)))
                plt.title("Original")
                plt.axis("off")
                plt.subplot(2, 5, idx + 6)
                # Check if the image is grayscale or RGB
                if recon_batch[idx].shape[0] == 1:
                    # Grayscale image
                    plt.imshow(recon_batch[idx].squeeze().numpy(), cmap="gray")
                else:
                    # RGB image
                    plt.imshow(np.transpose(recon_batch[idx], (1, 2, 0)))
                
                plt.title("Reconstructed")
                plt.axis("off")
            plt.tight_layout()
            plt.show()
            
    return activation_path_list

