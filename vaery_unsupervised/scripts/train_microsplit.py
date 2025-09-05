#%%
from datetime import datetime
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from careamics.lightning import VAEModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from monai.transforms import RandRotate, RandFlip
import cmap
import numpy as np

from microsplit_reproducibility.configs.factory import (
    create_algorithm_config,
    get_likelihood_config,
    get_loss_config,
    get_model_config,
    get_optimizer_config,
    get_training_config,
    get_lr_scheduler_config,
)
from microsplit_reproducibility.configs.parameters._base import SplittingParameters

from vaery_unsupervised.microsplit.utils import (
    compute_metrics,
    show_metrics,
    get_MicroSplit_predictions,
    stitch_tiles,
    full_frame_evaluation
)
from vaery_unsupervised.microsplit.microsplit_dataloader import MicroSplitHCSDataModule


#%%
# Create Data Module
ome_zarr_path = "/mnt/efs/aimbl_2025/student_data/S-RM/full_dataset/RM_project_ome2.zarr"

data_module = MicroSplitHCSDataModule(
    ome_zarr_path=ome_zarr_path,
    source_channel_names=['mito', 'er', 'nuclei'],
    crop_size=(128, 128),
    crops_per_position=4,
    batch_size=32,
    num_workers=6,
    split_ratios=(0.8, 0.1, 0.1),
    augmentations=[
        RandRotate(range_x=[90, 90], prob=0.2),
        RandFlip(prob=0.2, spatial_axis=-1),
        RandFlip(prob=0.2, spatial_axis=-2)
    ],
    mix_coeff_range=(0.5, 1.0)
)

data_module.setup()

#%%
# Visualize data examples
train_loader = data_module.train_dataloader()
batch_sample = next(iter(train_loader))
print(batch_sample[0].shape, batch_sample[1].shape)

#%%
cmap_nuc=cmap.Colormap('cmap:cyan').to_mpl()
cmap_er=cmap.Colormap('cmap:green').to_mpl()
cmap_mito=cmap.Colormap('chrisluts:BOP_Orange').to_mpl()
fig, ax = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
ax[0, 0].imshow(batch_sample[0].numpy()[0, 0], cmap='gray')
ax[0, 0].set_title("Mixed Image")
ax[0, 1].set_axis_off()
ax[0, 2].set_axis_off()
ax[1, 0].imshow(batch_sample[1].numpy()[0, 0], cmap=cmap_mito)
ax[1, 0].set_title("Channel 0")
ax[1, 1].imshow(batch_sample[1].numpy()[0, 1], cmap=cmap_er)
ax[1, 1].set_title("Channel 1")
ax[1, 2].imshow(batch_sample[1].numpy()[0, 2], cmap=cmap_nuc)
ax[1, 2].set_title("Channel 2")
plt.show()

#%%
# Set microsplit parameters
experiment_params = SplittingParameters(
    algorithm="musplit",
    loss_type="musplit", # no denoising
    img_size=(128, 128), # this should be consistent with the dataset
    target_channels=len(['mito', 'er', 'nuclei']),
    multiscale_count=1,
    lr=1e-3,
    num_epochs=100,
    lr_scheduler_patience=25,
    earlystop_patience=50,
    nm_paths=None, # data is not noise, we don't need explicit noisy models
).model_dump()


#%%
# Get other configs
# setting up training losses and model config (using default parameters)
loss_config = get_loss_config(**experiment_params)
model_config = get_model_config(**experiment_params)
gaussian_lik_config, noise_model_config, nm_lik_config = get_likelihood_config(
    **experiment_params
)
training_config = get_training_config(**experiment_params)

# setting up learning rate scheduler and optimizer (using default parameters)
lr_scheduler_config = get_lr_scheduler_config(**experiment_params)
optimizer_config = get_optimizer_config(**experiment_params)

# finally, assemble the full set of experiment configurations...
experiment_config = create_algorithm_config(
    algorithm=experiment_params["algorithm"],
    loss_config=loss_config,
    model_config=model_config,
    gaussian_lik_config=gaussian_lik_config,
    nm_config=noise_model_config,
    nm_lik_config=nm_lik_config,
    lr_scheduler_config=lr_scheduler_config,
    optimizer_config=optimizer_config,
)

#%%
# Initialize model (it is a LightningModule)
model = VAEModule(algorithm_config=experiment_config)

#%%
# Setup Trainer
logging_path = Path("/mnt/efs/aimbl_2025/student_data/S-RM/RM_microsplit_logs")
logging_path.mkdir(exist_ok=True)
logger = TensorBoardLogger(
    save_dir=logging_path,
    name=f"MicroSplit_{datetime.now().strftime('%Y-%m-%d_%H')}",
)
trainer = Trainer(
    max_epochs=training_config.num_epochs,
    accelerator="gpu",
    precision="32",
    logger=logger,
    gradient_clip_val=training_config.gradient_clip_val,
    gradient_clip_algorithm=training_config.gradient_clip_algorithm,
    callbacks=[
        ModelCheckpoint(
            save_last=True, save_top_k=3, monitor='val_loss', every_n_epochs=1
        )
    ]
)

#%%
# Train model
trainer.fit(
    model=model,
    train_dataloaders=data_module.train_dataloader(),
    val_dataloaders=data_module.val_dataloader()
)

#%%
# Or load previous checkpoints
ckpt_path = "/mnt/efs/aimbl_2025/student_data/S-RM/microsplit_logs/MicroSplit_2025-09-04_13/version_0/checkpoints/last.ckpt"
ckpt = torch.load(ckpt_path, map_location="cuda")
model.load_state_dict(ckpt['state_dict'], strict=True)
model.to("cuda")
print("Model loaded from checkpoint:", ckpt_path)

#%%
# Evaluate model on test dataset
unmixed_predictions, unmixed_stds = get_MicroSplit_predictions(
    model=model,
    dloader=data_module.test_dataloader(),
    mmse_count=2
)

#%%
# Visualize predictions
# get the target and input from the test dataset for visualization purposes
inputs = []
targets = []
for batch in data_module.test_dataloader():
    targets.append(batch[1].numpy())
    input_patches = batch[0].numpy()
    patch_info = batch[2]
    stitched_input = stitch_tiles(input_patches, patch_info)
    inputs.append(stitched_input)
    
    
#%%
plt.imshow(inputs[0][0][0], cmap='gray')


#%%
fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
axes[0].imshow(targets[0][0][0], cmap=cmap_mito)
axes[1].imshow(targets[0][0][1], cmap=cmap_er)
axes[2].imshow(targets[0][0][2], cmap=cmap_nuc)


#%%
fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
fig.patch.set_facecolor("black")
axes[0, 0].imshow(targets[40][0][0], cmap=cmap_mito)
axes[0, 1].imshow(targets[40][0][1], cmap=cmap_er)
axes[0, 2].imshow(targets[40][0][2], cmap=cmap_nuc)
axes[1, 0].imshow(unmixed_predictions[40][0], cmap=cmap_mito)
axes[1, 1].imshow(unmixed_predictions[40][1], cmap=cmap_er)
axes[1, 2].imshow(unmixed_predictions[40][2], cmap=cmap_nuc)

#%%
fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
fig.patch.set_facecolor("black")
for ax in axes.flatten():
    ax.set_axis_off()
axes[0, 0].imshow(targets[40][0][0, 300:700, 300:700], cmap=cmap_mito,
                  vmin=np.percentile(targets[40][0][0, 300:700, 300:700], 1),
                  vmax=np.percentile(targets[40][0][0, 300:700, 300:700], 99))
axes[0, 1].imshow(targets[40][0][1, 300:700, 300:700], cmap=cmap_er,
                  vmin=np.percentile(targets[40][0][1, 300:700, 300:700], 1),
                  vmax=np.percentile(targets[40][0][1, 300:700, 300:700], 99))
axes[0, 2].imshow(targets[40][0][2, 300:700, 300:700], cmap=cmap_nuc,
                  vmin=np.percentile(targets[40][0][2, 300:700, 300:700], 1),
                  vmax=np.percentile(targets[40][0][2, 300:700, 300:700], 99))

axes[1, 0].imshow(unmixed_predictions[40][0, 300:700, 300:700], cmap=cmap_mito,
                  vmin=np.percentile(unmixed_predictions[40][0, 300:700, 300:700], 1),
                  vmax=np.percentile(unmixed_predictions[40][0, 300:700, 300:700], 99))
axes[1, 1].imshow(unmixed_predictions[40][1, 300:700, 300:700], cmap=cmap_er,
                  vmin=np.percentile(unmixed_predictions[40][1, 300:700, 300:700], 1),
                  vmax=np.percentile(unmixed_predictions[40][1, 300:700, 300:700], 99))
axes[1, 2].imshow(unmixed_predictions[40][2, 300:700, 300:700], cmap=cmap_nuc,
                  vmin=np.percentile(unmixed_predictions[40][2, 300:700, 300:700], 1),
                  vmax=np.percentile(unmixed_predictions[40][2, 300:700, 300:700], 99))

#%%
fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
fig.patch.set_facecolor("black")
ax.imshow(
    inputs[40][0][0, 300:700, 300:700],
    cmap='gray',
    vmin=np.percentile(inputs[40][0][0, 300:700, 300:700], 1),
    vmax=np.percentile(inputs[40][0][0, 300:700, 300:700], 99)
)
ax.set_axis_off()


#%%
# Comment out the metrics you don't want to use
METRICS = [
    "PSNR",
    "Pearson",
    "SSIM",
    "MS-SSIM",
    "MicroSSIM",
    "MicroMS3IM",
    "LPIPS",
]

#%%
# prepare predictions for training
targets_arr = []
for target in targets:
    targets_arr.append(target[0])
targets_arr = np.array(targets_arr)
unmixed_predictions_arr = np.array(unmixed_predictions)

#%%
metrics_dict = compute_metrics(targets_arr, unmixed_predictions_arr, metrics=METRICS)
show_metrics(metrics_dict)