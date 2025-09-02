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

from vaery_unsupervised.

from dataloaders.marlin_dataloader.marlin_dataloader import MarlinDataModule, MarlinDataset
from networks.marlin_contrastive import ResNetEncoder, ContrastiveModule
from networks.marlin_contrastive import projection_mlp
from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss

from monai.transforms import (
    Compose,
    NormalizeIntensity,
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


#%%
HEADPATH = Path('/mnt/efs/aimbl_2025/student_data/S-GL/')
METADATA_PATH = HEADPATH / '2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded.pkl'
METADATA_COMPACT_PATH  = HEADPATH / '2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded_compact.pkl'
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
data_module.setup(stage='train')
#%%
dataloader = data_module.predict_dataloader()
batch_sample = next(iter(dataloader))
#%%
marlin_encoder_config = {
    "backbone": "resnet34",
    "in_channels": 1,
    "spatial_dims": 3,
    "embedding_dim": 512,
    "mlp_hidden_dims": 768,
    "projection_dim": 128,
    "pretrained": True,
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
emb, proj = marlin_encoder(batch_sample['anchor'])
# feature_map = marlin_encoder.resnet(batch_sample['anchor'])[-1]
# embedding = marlin_encoder.resnet.avgpool(feature_map)
# embedding = embedding.view(embedding.size(0), -1)
# print(embedding.shape)

#%%
import torchview
model_graph = torchview.draw_graph(
    marlin_contrastive,
    input_size=(32, 1, 1, 150, 150),
)

model_graph.visual_graph
# Save as figure
model_graph.save("~/marlin_contrastive_graph.png")
#%%
trainer = L.Trainer(
    devices="auto",
    accelerator="gpu",
    strategy="auto",
    precision="16-mixed",
    max_epochs=1,
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