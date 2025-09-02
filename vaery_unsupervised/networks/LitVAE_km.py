import torch
from lightning import LightningModule

from .model_VAE_resnet18 import ResNet18Enc, ResNet18Dec

def model_loss(x, recon_x, z_mean, z_log_var, beta = 1e-3):
  mse = torch.nn.functional.mse_loss(x, recon_x)
  kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
  loss = mse + beta * kl_loss
  return loss, mse, kl_loss

def reparameterize(mean, log_var):
  std = torch.exp(0.5 * log_var)
  epsilon = torch.randn_like(std)
  return mean + epsilon * std

class SpatialVAE(LightningModule):

  def __init__(self, channels_selection = [1,2,3],
               beta = 1e-3, latent_size = 128, 
               n_chan = 1, lr = 0.001, out_features = 128):

    super().__init__()
    self.out_features = out_features

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
        tensorboard.add_image("train/target", target[0])
        tensorboard.add_image("train/reconstruction", x_hat[0])
    return loss 

  def validation_step(self, batch, batch_idx):
    input = batch["input"][:,self.channels_selection,:,:]
    x_hat, z_mean, z_log_var = self(input)
    target = batch["target"][:,self.channels_selection,:,:]
    loss, recon, kld = self.loss(target, x_hat, z_mean, z_log_var)
    self.log("val/loss", loss.item())
    self.log("val/loss/recon", recon.item())
    self.log("val/loss/kld", kld.item())

    if batch_idx == 0:
    #     self.log("val/target", target[0])
    #     self.log("val/reconstruction", x_hat[0])
        tensorboard = self.logger.experiment
        tensorboard.add_image("val/target", target[0])
        tensorboard.add_image("val/reconstruction", x_hat[0])

    tensorboard.
    
    #return z_mean, z_log_var

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), self.lr)
