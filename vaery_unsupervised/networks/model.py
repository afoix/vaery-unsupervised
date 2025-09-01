import torch
import torchvision
from lightning import LightningModule

def model_loss(x, recon_x, z_log_var, z_mean, beta = 1e-3):
  mse = nn.MSELoss(x, recon_x)
  kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
  loss = mse + beta * 0
  return loss

class Encoder(torchvision.models.resnet18):
  pass

class Decoder)torchvision.models.resnet18):
  pass

def reparameterize(mean, log_var):
  std = torch.exp(0.5 * log_var)
  epsilon = torch.randn_like(std)
  return mean + epsilon * std

class VAELightning(LightningModule):

  def __init__(self, beta = 1e-3, matrix_size = 32, latent_size = 128):

    super().__init__()

    self.encode = Encoder()
    self.decode = Decoder()

    self.z_mean = nn.Linear(matrix_size, latent_size)
    self.z_log_var = nn.Linear(matrix_size, latent_size)

    # specify desired loss
    self.beta = beta
    self.loss = model_loss()

  def forward(self, x):
    x = self.encode(x)
    z_mean = self.z_mean(x)
    z_log_var = self.z_log_var(x)
    z = reparameterize(z_mean, z_log_var)
    return self.decode(z)

  def training_step(self, batch, batch_idx):
    x, _ = batch
    x_hat = self(x)
    loss = self.loss(x, x_hat)
    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters())
