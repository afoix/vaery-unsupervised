import logging

import torch
from lightning import LightningModule
from skimage.exposure import rescale_intensity
from typing import Sequence
import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from .VAE_3D_resnet18 import ResNet18Enc, ResNet18Dec

_logger = logging.getLogger("lightning.pytorch")
#_logger.setLevel(logging.DEBUG)


def model_loss(x, recon_x, z_mean, z_log_var, beta=1e-3):
    mse = torch.nn.functional.mse_loss(x, recon_x)
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    loss = mse + beta * kl_loss
    return loss, mse, kl_loss


def reparameterize(mean, log_var):
    std = torch.exp(0.5 * log_var)
    epsilon = torch.randn_like(std)
    return mean + epsilon * std


def render_images(
    imgs: Sequence[Sequence[np.ndarray]], cmaps: list[str] = []
) -> np.ndarray:
    """Render images in a grid.

    Parameters
    ----------
    imgs : Sequence[Sequence[np.ndarray]]
        Grid of images to render, output of `detach_sample`.
    cmaps : list[str], optional
        Colormaps for each column, by default []

    Returns
    -------
    np.ndarray
        Rendered RGB images grid.
    """
    images_grid = []
    for sample_images in imgs:
        images_row = []
        for i, image in enumerate(sample_images):
            if cmaps:
                cm_name = cmaps[i]
            else:
                cm_name = "gray" if i == 0 else "inferno"
            if image.ndim == 2:
                image = image[np.newaxis]
            for channel in image:
                channel = rescale_intensity(channel, out_range=(0, 1))
                render = get_cmap(cm_name)(channel, bytes=True)[..., :3]
                images_row.append(render)
        images_grid.append(np.concatenate(images_row, axis=1))
    return np.concatenate(images_grid, axis=0)


class SalamanderVAE(LightningModule):
    def __init__(self, beta=1e-3, matrix_size=32, latent_size=128, n_chan=1, z_dir=""):
        super().__init__()

        self.encode = ResNet18Enc(nc=n_chan, z_dim=latent_size)
        self.decode = ResNet18Dec(nc=n_chan, z_dim=latent_size)

        # specify desired loss
        self.save_hyperparameters()
        self.beta = beta
        self.loss = model_loss

        self.z_dir = z_dir

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = reparameterize(z_mean, z_log_var)
        return self.decode(z), z_mean, z_log_var

    def training_step(self, batch, batch_idx):
        # print(batch.shape)
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

        self.img_sample = []
        self.x_hat_sample = []
        for b_i in range(0, x.shape[0], 3):
            self.img_sample.append(batch[b_i].detach())
            self.x_hat_sample.append(x_hat[b_i].detach())
        
        if batch_idx == 0: 
            z = reparameterize(z_mean, z_log_var)
            self.z_sample = z.detach()

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()

        if self.z_dir:
            _logger.debug(f'Saving latent vectors to {self.z_dir}')
            np.savez_compressed(self.z_dir, z=self.z_sample.cpu().numpy())
        # get images    
        tensorboard = self.logger.experiment
        C = self.img_sample[0].shape[0]
        z_slice = self.img_sample[0].shape[-1]//2

        for b, (this_target, this_recon) in enumerate(zip(self.img_sample, self.x_hat_sample)):
            fig, ax = plt.subplots(C, 2, figsize=(3, 40))
            for c in range(C):
                # this_target_c = np.max(this_target[c].cpu().numpy(), axis=-1)
                # this_recon_c = np.max(this_recon[c].cpu().numpy(), axis=-1)
                this_target_c = this_target[c, :, :, z_slice].cpu().numpy()
                this_recon_c = this_recon[c, :, :, z_slice].cpu().numpy()
                ax[c, 0].imshow(this_target_c, cmap='gray')
                ax[c, 1].imshow(this_recon_c, cmap='gray')

                ax[c, 0].axis('off')
                ax[c, 1].axis('off')
            plt.tight_layout()

            tensorboard.add_figure(f"val/recon_{b}", fig, self.current_epoch)



    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
