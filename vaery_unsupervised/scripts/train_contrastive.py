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
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
import torchview
#%%
ome_zarr_path = "/mnt/efs/aimbl_2025/student_data/S-RM/full_dataset/RM_project_ome2.zarr"

data_module = HCSDataModule(
        ome_zarr_path=ome_zarr_path,
        source_channel_names=['mito','er','nuclei'],
        weight_channel_name='nuclei',
        crop_size=(128, 128),
        crops_per_position=4,
        batch_size=32,
        num_workers=6,
        split_ratio=0.8,
        normalization_transform=[],
        augmentations=[]
    )

data_module.setup(stage='train')
#%%
train_loader = data_module.train_dataloader()

#%%
batch_sample = next(iter(train_loader))
print(batch_sample['anchor'].shape, batch_sample['positive'].shape)
#%%
import matplotlib.pyplot as plt

# plt.imshow(batch_sample['anchor'][0].numpy(), cmap='gray')
# plt.imshow(batch_sample['positive'][0].numpy(), cmap='gray')

#%%
fig, ax = plt.subplots(3, 2, figsize=(10, 15))
ax[0, 0].imshow(batch_sample['anchor'].numpy()[0,0], cmap='gray')
ax[0, 1].imshow(batch_sample['positive'].numpy()[0,0], cmap='gray')
ax[1, 0].imshow(batch_sample['anchor'].numpy()[0,1], cmap='gray')
ax[1, 1].imshow(batch_sample['positive'].numpy()[0,1], cmap='gray')
ax[2, 0].imshow(batch_sample['anchor'].numpy()[0,2], cmap='gray')
ax[2, 1].imshow(batch_sample['positive'].numpy()[0,2], cmap='gray')
plt.show()

# plt.imshow(batch_sample['anchor'].numpy()[0,0], cmap='gray')
# plt.imshow(batch_sample['positive'].numpy()[0,0], cmap='gray')

#%%
# from tqdm import tqdm
# means = []
# stds = []
# for batch in tqdm(train_loader):
#     means.append(batch['anchor'].mean(dim=[0,2,3]))
#     means.append(batch['positive'].mean(dim=[0,2,3]))
#     stds.append(batch['anchor'].std(dim=[0,2,3]))
#     stds.append(batch['positive'].std(dim=[0,2,3]))

#%%
hcs_encoder_config = {
    "backbone": "resnet18",
    "in_channels": 3,
    "spatial_dims": 2,
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
    "lr": 1e-4,
}

hcs_contrastive = ContrastiveModule(**hcs_contrastive_config)

model_graph = torchview.draw_graph(
    hcs_contrastive,
    input_size=(32, 3, 360, 360),
)

model_graph.visual_graph

#%%

logging_path = Path("/mnt/efs/aimbl_2025/student_data/S-RM/logs")
logging_path.mkdir(exist_ok=True)
logger = TensorBoardLogger(save_dir=logging_path,name="contrastive_first")
def main(*args, **kwargs):
    trainer = L.Trainer(
        max_epochs = 350, accelerator = "gpu", precision = "32", logger=logger,
        callbacks=[
            ModelCheckpoint(
                save_last=True, save_top_k=8, monitor='loss/val', every_n_epochs=1
            )
        ]
    )

    trainer.fit(
        model = hcs_contrastive,
        train_dataloaders = data_module.train_dataloader(),
        val_dataloaders = data_module.val_dataloader()
    )
    
#%%
main()


#%%



