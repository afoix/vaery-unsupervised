#%% testing
import numpy as np
from pathlib import Path
from vaery_unsupervised.dataloaders.dataloader_km_ryans_template import (
    simple_masking, 
    DATASET_NORM_DICT, 
    SpatProteoDatasetZarr,
    SpatProtoZarrDataModule,
)
from vaery_unsupervised.networks.LightningVAE_linear_km import SpatialVAE_Linear
import yaml
from vaery_unsupervised.km_utils import plot_batch_sample,plot_dataloader_output
import monai.transforms as transforms
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path

dataset_path = Path("/mnt/efs/aimbl_2025/student_data/S-KM/")


dataset_zarr = SpatProteoDatasetZarr(
    dataset_path/"converted_crops_with_metadata.zarr",
    masking_function=simple_masking,
    dataset_normalisation_dict=DATASET_NORM_DICT,
    transform_both=None,
    transform_input=None
)
plot_dataloader_output(dataset_zarr[0])
# %%
lightning_module = SpatProtoZarrDataModule(
    dataset_path/"converted_crops_with_metadata.zarr",
    masking_function=simple_masking,
    dataset_normalisation_dict=DATASET_NORM_DICT,
    transform_both=None,
    transform_input=None,
    num_workers=8,
    batch_size=16,
)
lightning_module.setup("predict")

loader = lightning_module.predict_dataloader()
#%%
for batch in loader:
    batch = batch
    break

#%%

#%%
checkpoint_path = "/mnt/efs/aimbl_2025/student_data/S-KM/logs/linear_VAE_tmuxtraining2/version_0/checkpoints/epoch=145-step=6278.ckpt"
model = SpatialVAE_Linear.load_from_checkpoint(checkpoint_path=checkpoint_path)
#%%
import torch


# Compare with current model
checkpoint = torch.load(checkpoint_path, map_location='cpu')
print("Keys in checkpoint state_dict:")
for key in checkpoint['state_dict'].keys():
    print(key)

#%%
model = SpatialVAE_Linear()  # Initialize with same params as training
print("\nKeys in current model:")
for name, param in model.named_parameters():
    print(name)

#%%
from vaery_unsupervised.networks.LitVAE_km import reparameterize

# %%
for batch in loader:
    image_ids = batch["metadata"]['well_id']
    input = batch["input"].to(device = model.device)
    reconstruction, z_mean, z_log_var = model(input[:,model.channels_selection,:,:])
    z = reparameterize(z_mean, z_log_var)
    break


# %%
input[:,[model.channels_selection],:,:].shape
# %%
model.channels_selection
# %%
model.device
# %%
all_image_ids = []
all_input = []
all_reconstruction = []
all_z_mean = []
all_z_log_var = []
all_z = []
#%%
model.eval()
#%%
for batch in loader:
    image_ids = batch["metadata"]['well_id']
    input = batch["input"][:,model.channels_selection,:,:].to(device = model.device)
    reconstruction, z_mean, z_log_var = model(input[:,model.channels_selection,:,:])
    z = reparameterize(z_mean, z_log_var)

    all_image_ids.append(image_ids) 
    all_input.append(input.detach().cpu())
    all_reconstruction.append(reconstruction.detach().cpu())
    all_z_mean.append(z_mean.detach().cpu())
    all_z_log_var.append(z_log_var.detach().cpu())
    all_z.append(z.detach().cpu())

all_image_ids = np.concatenate(all_image_ids, axis = 0) 
import torch
all_input = torch.cat(all_input, dim = 0).numpy()
all_reconstruction = torch.cat(all_reconstruction, dim = 0).numpy() 
all_z_mean = torch.cat(all_z_mean, dim = 0).numpy() 
all_z_log_var = torch.cat(all_z_log_var, dim = 0).numpy() 
all_z = torch.cat(all_z, dim = 0).numpy()
    




# %%

# %%

