
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
    "lr": 1e-3,
}

hcs_contrastive = ContrastiveModule(**hcs_contrastive_config)

model_graph = torchview.draw_graph(
    hcs_contrastive,
    input_size=(32, 3, 360, 360),
)

model_graph.visual_graph
# Save as figure
# model_graph.save("~/hcs_contrastive_rm.png")





#%%
# emb, proj = hcs_encoder(batch_sample['anchor'])

# print(batch_sample['anchor'])

#%%
# trainer = L.Trainer(
#     devices="auto",
#     accelerator="gpu",
#     strategy="auto",
#     precision="16-mixed",
#     max_epochs=100,
# )


logging_path = Path("/mnt/efs/aimbl_2025/student_data/S-RM/logs")
logging_path.mkdir(exist_ok=True)
logger = TensorBoardLogger(save_dir=logging_path,name="contrastive_first")
def main(*args, **kwargs):
    trainer = L.Trainer(max_epochs = 100, accelerator = "gpu", precision = "16-mixed", logger=logger,
                              callbacks=[ModelCheckpoint(save_last=True,save_top_k=8,monitor='loss/val',every_n_epochs=1)]

                              )



    trainer.fit(
    model = hcs_contrastive,
    train_dataloaders = data_module.train_dataloader(),
    val_dataloaders = data_module.val_dataloader()
    )
    
#%%
main()


#%%



