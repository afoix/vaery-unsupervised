# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# from pytorch_lightning.loggers import TensorBoardLogger
# import torch
# from pytorch_metric_learning.losses import NTXentLoss

# from vaery_unsupervised.networks.marlin_contrastive import ResNetEncoder, ContrastiveModule
# from vaery_unsupervised.dataloaders.hcs_dataloader_ryan import HCSDataModule

# def main():
#     # 1. Create the encoder
#     encoder = ResNetEncoder(
#         backbone="resnet18",
#         in_channels=3,  # mito, er, nuclei channels
#         spatial_dims=2,  # 2D images
#         embedding_dim=512,
#         mlp_hidden_dims=768,
#         projection_dim=128,
#         pretrained=True
#     )

#     # 2. Create the loss function
#     loss_fn = NTXentLoss(temperature=0.07)  # Standard SimCLR temperature

#     # 3. Create the model
#     model = ContrastiveModule(
#         encoder=encoder,
#         loss=loss_fn,
#         lr=1e-4,  # Starting with a lower learning rate
#         optimizer=torch.optim.AdamW  # Using AdamW optimizer
#     )

#     # 4. Create the data module
#     datamodule = HCSDataModule(
#         ome_zarr_path="/mnt/efs/aimbl_2025/student_data/S-RM/full_dataset/RM_project_ome.zarr",  # Replace with your data path
#         source_channel_names=['mito', 'er', 'nuclei'],
#         weight_channel_name='nuclei',
#         crop_size=(256, 256),
#         crops_per_position=4,
#         batch_size=32,
#         num_workers=4,
#         split_ratio=0.8,
#         normalization_transform=[
#             # Add your normalization transforms here if needed
#         ],
#         augmentations=[]  # Add any additional augmentations if needed
#     )

#     # 5. Set up logging and callbacks
#     logger = TensorBoardLogger(
#         save_dir='logs',
#         name='contrastive_learning'
#     )

#     callbacks = [
#         ModelCheckpoint(
#             dirpath='checkpoints',
#             filename='contrastive-{epoch:02d}-{val_loss:.2f}',
#             monitor='val_loss',
#             mode='min',
#             save_top_k=3,
#             save_last=True,
#         ),
#         LearningRateMonitor(logging_interval='step')
#     ]

#     # 6. Create trainer
#     trainer = pl.Trainer(
#         max_epochs=100,
#         accelerator='gpu' if torch.cuda.is_available() else 'cpu',
#         devices=1,
#         logger=logger,
#         callbacks=callbacks,
#         gradient_clip_val=1.0,  # Add gradient clipping for stability
#         precision='16-mixed' if torch.cuda.is_available() else '32',  # Use mixed precision if on GPU
#     )

#     # 7. Train!
#     trainer.fit(model, datamodule)

# if __name__ == "__main__":
#     main()
#%%
import lightning as L
from typing import Callable
import numpy as np
import pytorch_lightning as pl
import torch
from iohub import open_ome_zarr
from iohub.ngff import Position
from monai.data import set_track_meta
from monai.transforms import Compose, ToTensord
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from monai.transforms import Compose, RandSpatialCrop, RandRotate, RandWeightedCrop, CenterSpatialCrop
import logging
from typing import Literal
import torch.nn.functional as F
from lightning import LightningModule
from monai.networks.nets.resnet import ResNetFeatures
from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss
from torch import Tensor, nn
from typing_extensions import TypedDict

from vaery_unsupervised.dataloaders.hcs_dataloader_ryan import HCSDataModule
from vaery_unsupervised.networks.hcs_contrastive import ResNetEncoder, ContrastiveModule

#%%
ome_zarr_path = "/mnt/efs/aimbl_2025/student_data/S-RM/full_dataset/RM_project_ome.zarr"

data_module = HCSDataModule(
        ome_zarr_path=ome_zarr_path,
        source_channel_names=['mito','er','nuclei'],
        weight_channel_name='nuclei',
        crop_size=(128, 128),
        crops_per_position=4,
        batch_size=32,
        num_workers=10,
        split_ratio=0.8,
        normalization_transform=[],
        augmentations=[]
    )

data_module.setup(stage='train')
#%%
train_loader = data_module.train_dataloader()

batch_sample = next(iter(train_loader))
print(batch_sample['anchor'].shape, batch_sample['positive'].shape)

#%%
hcs_encoder_config = {
    "backbone": "resnet18",
    "in_channels": 3,
    "spatial_dims": 3,
    "embedding_dim": 512,
    "mlp_hidden_dims": 768,
    "projection_dim": 128,
    "pretrained": False,
}

hcs_encoder = ResNetEncoder(**hcs_encoder_config)

#%%
hcs_contrastive_config = {
    "encoder": hcs_encoder,
    "loss": SelfSupervisedLoss(NTXentLoss(temperature=0.07)),
    "lr": 1e-3,
}

hcs_contrastive = ContrastiveModule(**hcs_contrastive_config)
#%%
emb, proj = hcs_encoder(batch_sample['anchor'])

# print(batch_sample['anchor'])

#%%
trainer = L.Trainer(
    devices="auto",
    accelerator="gpu",
    strategy="auto",
    precision="16-mixed",
    max_epochs=100,
)

trainer.fit(
    model = hcs_contrastive,
    train_dataloaders = data_module.train_dataloader()
)
#%%


