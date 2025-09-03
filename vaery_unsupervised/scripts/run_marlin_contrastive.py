#%%
%load_ext autoreload
%autoreload 2
import lightning as L
from pathlib import Path
from typing import Callable, Literal, Sequence, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
import random

from vaery_unsupervised.dataloaders.marlin_dataloader.marlin_dataloader import MarlinDataModule, MarlinDataset

from vaery_unsupervised.networks.marlin_contrastive import ResNetEncoder, ContrastiveModule

from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss

from monai.transforms import (
    Compose,
    NormalizeIntensity,
    RandFlipd,
)

# MEAN_OVER_DATASET = 13
# STD_OVER_DATASET = 11

transforms = Compose([
    # NormalizeIntensity(
    #     subtrahend=MEAN_OVER_DATASET,
    #     divisor=STD_OVER_DATASET,
    #     nonzero=False,
    #     channel_wise=False,
    #     dtype = np.float32,
    # ),
    RandFlipd(
        keys=["positive"],
        spatial_axis=0,
        prob=1,
    ),
    RandFlipd(
        keys=["positive"],
        spatial_axis=1,
        prob=1,
    )
])
# transforms=None

#%%
HEADPATH = Path('/mnt/efs/aimbl_2025/student_data/S-GL/')
METADATA_PATH = HEADPATH / '2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded.pkl'
METADATA_COMPACT_PATH  = HEADPATH / '2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded_select_grnas_allT.pickle'
#Adjust Metadata
# metadata = pd.read_pickle(HEADPATH/'2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded.pkl')
# metadata_compact = (metadata
#     .loc[lambda df_:df_['Gene'].isin(['ftsZ', 'csrA', 'lolE'])]
# )
# metadata_compact.to_pickle(HEADPATH/'2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded_compact.pkl')
#%%

data_module = MarlinDataModule(
    data_path=HEADPATH/'Ecoli_lDE20_Exps-0-1/',
    metadata_path=METADATA_COMPACT_PATH,
    split_ratio=0.8,
    batch_size=32,
    num_workers=4,
    prefetch_factor=2,
    transforms=transforms
)

data_module._prepare_data()
data_module.setup(stage='fit')
#%%
data_module.metadata
#%%
data_module.train_dataset[0]
# import matplotlib.pyplot as plt
# batch_sample['anchor'].shape
#%%
plt.hist(batch_sample['anchor'].numpy().flatten(), bins=50)
print(np.mean(batch_sample['anchor'].numpy()))
print(np.std(batch_sample['anchor'].numpy()))
#%%
fig, ax = plt.subplots(1, 2, figsize=(10, 10))
fig.patch.set_facecolor('black')
# ax[0].set("off")
# ax[1].set("off")
ax[0].imshow(batch_sample['anchor'].numpy()[0, 0][:,65:85], cmap='gray')
ax[1].imshow(batch_sample['positive'].numpy()[0, 0][:,65:85], cmap='gray')


#%%
plt.hist(batch_sample['positive'].numpy()[0, 0], bins=50)
#%%
dataloader = data_module.train_dataloader()
# batch_sample = next(iter(dataloader))
# print(batch_sample['anchor'].shape)
import time

for batch in dataloader:
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    # fig.patch.set_facecolor('black')
    ax[0].imshow(batch['anchor'].numpy()[0, 0], cmap='gray')
    ax[1].imshow(batch['positive'].numpy()[0, 0], cmap='gray')
    # Pause for 2 seconds
    plt.show()
    time.sleep(1)
    plt.close()
#%%
marlin_encoder_config = {
    "backbone": "resnet18",
    "in_channels": 1,
    "spatial_dims": 2,
    "embedding_dim": 512,
    "mlp_hidden_dims": 768,
    "projection_dim": 128,
    "pretrained": False,
}

marlin_encoder = ResNetEncoder(**marlin_encoder_config)

#%%
marlin_contrastive_config = {
    "encoder": marlin_encoder,
    "loss": SelfSupervisedLoss(NTXentLoss(temperature=0.07)),
    "lr": 1e-3,
}

marlin_contrastive = ContrastiveModule(**marlin_contrastive_config)
#%%
marlin_contrastive(batch_sample['anchor'], batch_sample['positive'])


#%%
# emb, proj = marlin_encoder(batch_sample['anchor'])
feature_map = marlin_encoder.resnet(batch_sample['anchor'])[-1]
# embedding = marlin_encoder.resnet.avgpool(feature_map)
# embedding = embedding.view(embedding.size(0), -1)
# print(embedding.shape)
#%%
#%%
import torchview
model_graph = torchview.draw_graph(
    marlin_contrastive,
    input_size=(32, 1, 150, 150),
)

model_graph.visual_graph
# Save as figure
# model_graph.save("~/marlin_contrastive_graph.png")
#%%
from lightning.pytorch.loggers import TensorBoardLogger
logger = TensorBoardLogger(
    save_dir=HEADPATH/"tb_logs",
    name="marlin_contrastive",
)

#%%



trainer = L.Trainer(
    devices="auto",
    accelerator="gpu",
    strategy="auto",
    precision="16-mixed",
    max_epochs=100,
    fast_dev_true=False,
    logger=logger,
    callback=[
        L.callbacks.ModelCheckpoint(
            monitor="loss/val",
            save_top_k=8)
    ]

)

trainer.fit(
    model = marlin_contrastive,
    train_dataloaders = data_module.train_dataloader()
)

# %%
from monai.transforms import (
    Flip,
    Rotate90,
    Compose,
    NormalizeIntensity,
    Lambda,
)

MEAN_OVER_DATASET = 13
STD_OVER_DATASET = 11

transforms = Compose([
    NormalizeIntensity(
        subtrahend=MEAN_OVER_DATASET,
        divisor=STD_OVER_DATASET,
        nonzero=False,
        channel_wise=False,
        dtype = np.float32,
    ),
    # Lambda(func=center_image,
    #        inv_func=None,
    #        track_meta=True)
])

a = transforms(data_module.dataset[0]['anchor'])