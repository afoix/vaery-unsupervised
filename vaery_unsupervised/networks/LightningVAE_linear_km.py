import torch
from lightning import LightningModule
import numpy as np
from .km_ryan_linearresnet import ResNet18Enc, ResNet18Dec
from  sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

def model_loss(x, recon_x, z_mean, z_log_var, beta = 1e-3):
  mse = torch.nn.functional.mse_loss(x, recon_x)
  kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
  loss = mse + beta * kl_loss
  return loss, mse, kl_loss

def reparameterize(mean, log_var):
  std = torch.exp(0.5 * log_var)
  epsilon = torch.randn_like(std)
  return mean + epsilon * std

class SpatialVAE_Linear(LightningModule):
  def __init__(self, channels_selection = [1,2,3],
               beta = 1e-3, latent_size = 128, 
               n_chan = 1, lr = 0.001, out_features = 128,
               latentspace_dir = ""):

    super().__init__()
    self.save_hyperparameters()
    self.out_features = out_features
    self.latentspace_dir = latentspace_dir
 
    self.encode = ResNet18Enc(nc = n_chan, z_dim = latent_size)
    self.decode = ResNet18Dec(nc = n_chan, z_dim = latent_size, out_features=self.out_features)

    # specify desired loss
    self.beta = beta
    self.loss = model_loss
    self.lr = lr
    self.channels_selection = channels_selection

  def forward(self, x):
    z_mean, z_log_var = self.encode(x)
    z = reparameterize(z_mean, z_log_var)
    return self.decode(z), z_mean, z_log_var

  def training_step(self, batch, batch_idx):
    input = batch["input"][:,self.channels_selection,:,:]
    x_hat, z_mean, z_log_var = self(input)
    target = batch["target"][:,self.channels_selection,:,:]
    loss, recon, kld = self.loss(target, x_hat, z_mean, z_log_var)

    self.log("train/loss", loss.item())
    self.log("train/loss/recon", recon.item())
    self.log("train/loss/kld", kld.item())

    if batch_idx == 0:
        tensorboard = self.logger.experiment
        tensorboard.add_image("train/target", target[0], self.current_epoch)
        tensorboard.add_image("train/reconstruction", x_hat[0], self.current_epoch)
    return loss 

  def validation_step(self, batch, batch_idx):
    input = batch["input"][:,self.channels_selection,:,:]
    x_hat, z_mean, z_log_var = self(input)
    target = batch["target"][:,self.channels_selection,:,:]
    loss, recon, kld = self.loss(target, x_hat, z_mean, z_log_var)
    self.log("val/loss", loss.item())
    self.log("val/loss/recon", recon.item())
    self.log("val/loss/kld", kld.item())

    # Check if this is the last batch and save z
    if batch_idx == 0:
        z = reparameterize(z_mean, z_log_var)
        epoch = self.current_epoch
        filename = f"{self.latentspace_dir}/z_val_epoch_{epoch}.npy"
        np.save(filename, z.clone().detach().cpu().numpy())
        print(f"Saved validation latent codes for epoch {epoch} from batch {batch_idx}")
        image_ids = batch["metadata"]["well_id"]
    
        # Debug the original shapes
        print(f"Original x_hat shape: {x_hat.shape}")
        print(f"Original target shape: {target.shape}")
        
        # Convert to 3 channels if needed
        if x_hat.shape[1] != 3:
            if x_hat.shape[1] == 1:  # Grayscale
                x_hat_rgb = x_hat.repeat(1, 3, 1, 1)
                target_rgb = target.repeat(1, 3, 1, 1)
            elif x_hat.shape[1] == 2:  # Two channels
                zeros = torch.zeros_like(x_hat[:, :1])
                x_hat_rgb = torch.cat([x_hat, zeros], dim=1)
                target_rgb = torch.cat([target, zeros], dim=1)
            else:
                # More than 3 channels - take first 3
                x_hat_rgb = x_hat[:, :3]
                target_rgb = target[:, :3]
        else:
            x_hat_rgb = x_hat
            target_rgb = target
        
        # Stack horizontally
        stacked_imgs = x_hat_rgb
        #stacked_imgs = torch.cat([x_hat_rgb, target_rgb], dim=-1)
        
        # Move to CPU
        z_cpu = z.detach().cpu()
        stacked_imgs_cpu = stacked_imgs.detach().cpu()

        tensorboard = self.logger.experiment
        tensorboard.add_image("val/target", target[0], self.current_epoch)
        tensorboard.add_image("val/reconstruction", x_hat[0], self.current_epoch)
        tensorboard.add_embedding(z_cpu, image_ids, stacked_imgs_cpu, global_step=self.current_epoch, tag = "embedding")

    



    #return z_mean, z_log_var

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), self.lr)
