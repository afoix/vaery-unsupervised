#%%
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torchview
from careamics.lightning import VAEModule
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from monai.transforms import RandRotate, RandFlip

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
from microsplit_reproducibility.notebook_utils.custom_dataset_2D import (
    get_unnormalized_predictions,
    get_target,
    get_input,
    show_sampling,
    pick_random_patches_with_content,
)
from utils import (
    compute_metrics,
    show_metrics,
    full_frame_evaluation,
)

from vaery_unsupervised.dataloaders.microsplit.microsplit_dataloader import MicroSplitHCSDataModule


#%%
# Create Data Module
ome_zarr_path = "/mnt/efs/aimbl_2025/student_data/S-RM/full_dataset/RM_project_ome2.zarr"

data_module = MicroSplitHCSDataModule(
    ome_zarr_path=ome_zarr_path,
    source_channel_names=['mito', 'er', 'nuclei'],
    crop_size=(128, 128),
    crops_per_position=4,
    batch_size=32,
    num_workers=0,
    split_ratio=0.85,
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
print(batch_sample['input'].shape, batch_sample['target'].shape)

#%%
fig, ax = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
ax[0, 0].imshow(batch_sample['input'].numpy()[0, 0], cmap='gray')
ax[0, 0].set_title("Mixed Image")
ax[0, 1].set_axis_off()
ax[0, 2].set_axis_off()
ax[1, 0].imshow(batch_sample['target'].numpy()[0, 0], cmap='magma')
ax[1, 0].set_title("Channel 0")
ax[1, 1].imshow(batch_sample['target'].numpy()[0, 1], cmap='magma')
ax[1, 1].set_title("Channel 1")
ax[1, 2].imshow(batch_sample['target'].numpy()[0, 2], cmap='magma')
ax[1, 2].set_title("Channel 2")
plt.show()

#%%
# Set microsplit parameters
experiment_params = SplittingParameters(
    algorithm="musplit",
    loss_type="musplit", # no denoising
    img_size=(128, 128), # this should be consistent with the dataset
    target_channels=len(['mito', 'er', 'nuclei']),
    multiscale_count=3,
    lr=1e-3,
    num_epochs=30,
    lr_scheduler_patience=10,
    earlystop_patience=20,
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
logging_path = Path("/mnt/efs/aimbl_2025/student_data/S-RM/logs/microsplit")
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
            save_last=True, save_top_k=8, monitor='loss/val', every_n_epochs=1
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
# Evaluate model on val dataset
# %% tags=[]
stitched_predictions, _, stitched_stds = (
    get_unnormalized_predictions(
        model,
        data_module.val_dataset,
        data_key=..., # FIXME
        mmse_count=10,
        grid_size=64,
        num_workers=3,
        batch_size=32,
    )
)


#%%
# Visualize predictions
# get the target and input from the test dataset for visualization purposes
tar = get_target(data_module.val_dataset)
inp = get_input(data_module.val_dataset).sum(-1)

#%%
frame_idx = 0 # Change this index to visualize different frames
assert frame_idx < len(stitched_predictions), f"Frame index {frame_idx} out of bounds. Max index is {len(stitched_predictions) - 1}."

full_frame_evaluation(stitched_predictions[frame_idx], tar[frame_idx], inp[frame_idx], same_scale=False)

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
# Extract gt dataset

#%%
metrics_dict = compute_metrics(gt_target, stitched_predictions, metrics=METRICS)
show_metrics(metrics_dict)
