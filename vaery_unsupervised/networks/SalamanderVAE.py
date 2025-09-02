import torch
from lightning import LightningModule

from .VAE_3D_resnet18 import ResNet18Enc, ResNet18Dec

def model_loss(x, recon_x, z_mean, z_log_var, beta = 1e-3):
  mse = torch.nn.functional.mse_loss(x, recon_x)
  kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
  loss = mse + beta * kl_loss
  return loss, mse, kl_loss

def reparameterize(mean, log_var):
  std = torch.exp(0.5 * log_var)
  epsilon = torch.randn_like(std)
  return mean + epsilon * std

class SalamanderVAE(LightningModule):

  def __init__(self, beta = 1e-3, 
               matrix_size = 32, 
               latent_size = 128, 
               n_chan = 1):

    super().__init__()

    self.encode = ResNet18Enc(nc = n_chan, z_dim = latent_size)
    self.decode = ResNet18Dec(nc = n_chan, z_dim = latent_size)

    # specify desired loss
    self.beta = beta
    self.loss = model_loss

  def forward(self, x):
    z_mean, z_log_var = self.encode(x)
    z = reparameterize(z_mean, z_log_var)
    return self.decode(z), z_mean, z_log_var

  def training_step(self, batch, batch_idx):
    print(batch.shape)
    x = batch
    x_hat, z_mean, z_log_var = self(x)
    loss, recon, kld = self.loss(x, x_hat, z_mean, z_log_var)
    self.log("train/loss", loss.item())
    self.log("train/loss/recon", recon.item())
    self.log("train/loss/kld", kld.item())
    return loss

  def validation_step(self, batch, batch_idx):
    x = batch
    x_hat, z_mean, z_log_var = self(x)
    loss, recon, kld = self.loss(x, x_hat, z_mean, z_log_var)
    self.log("val/loss", loss.item())
    self.log("val/loss/recon", recon.item())
    self.log("val/loss/kld", kld.item())

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters())
