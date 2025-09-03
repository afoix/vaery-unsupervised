#%%
%load_ext autoreload
%autoreload 2
from lightning import LightningModule
from pathlib import Path
import torch.nn.functional as F
import torch
from vaery_unsupervised.networks.marlin_contrastive import ContrastiveModule, ResNetEncoder
from monai.networks.nets.resnet import ResNetFeatures
from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss

import torch
from monai.transforms import (
    Compose,
    NormalizeIntensity,
    RandFlipd,
)

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
from typing import Callable, Literal, Sequence, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt


from vaery_unsupervised.dataloaders.marlin_dataloader.marlin_dataloader import MarlinDataModule, MarlinDataset

from vaery_unsupervised.networks.marlin_contrastive import ResNetEncoder, ContrastiveModule

from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss
from lightning.pytorch.loggers import TensorBoardLogger
from monai.transforms import (
    Compose,
    NormalizeIntensity,
    RandFlipd,
)

# Get the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

LOGS_HEADPATH = Path('/mnt/efs/aimbl_2025/student_data/S-GL/tb_logs/marlin_contrastive/')
checkpoint_path = LOGS_HEADPATH / 'version_7/checkpoints/last.ckpt'

marlin_encoder_config = {
    "backbone": "resnet18",
    "in_channels": 1,
    "spatial_dims": 2,
    "embedding_dim": 512,
    "mlp_hidden_dims": 256,#768,
    "projection_dim": 32,
    "pretrained": False,
    }

marlin_encoder = ResNetEncoder(**marlin_encoder_config)
loss = SelfSupervisedLoss(NTXentLoss(temperature=0.07))
lr = 1e-3
optimizer = torch.optim.Adam(marlin_encoder.parameters(), lr=lr)

model = ContrastiveModule.load_from_checkpoint(
    checkpoint_path,
    encoder=marlin_encoder,
    loss=loss,
    lr=lr,
    optimizer=optimizer,
)
model.to(device)
model.eval()

HEADPATH = Path('/mnt/efs/aimbl_2025/student_data/S-GL/')
METADATA_PATH = HEADPATH / '2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded.pkl'
METADATA_COMPACT_PATH  = HEADPATH / '2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded_filtered_266-trenches.pkl'
#Adjust Metadata
# metadata = pd.read_pickle(HEADPATH/'2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded.pkl')
# metadata_compact = (metadata
#     .loc[lambda df_:df_['Gene'].isin(['ftsZ', 'csrA', 'lolE'])]
# )
# metadata_compact.to_pickle(HEADPATH/'2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded_compact.pkl')
from vaery_unsupervised.dataloaders.marlin_dataloader.marlin_dataloader import MarlinDataModule
from sklearn.decomposition import PCA
data_module = MarlinDataModule(
    data_path=HEADPATH/'Ecoli_lDE20_Exps-0-1/',
    metadata_path=METADATA_COMPACT_PATH,
    split_ratio=0.8,
    batch_size=256,#64,
    num_workers=4,
    prefetch_factor=2,
    transforms=transforms,
)

data_module.setup(stage='predict')

trainer = L.Trainer(
        devices="auto",
        accelerator="gpu",
        strategy="auto",
        precision="16-mixed",
        max_epochs=1000,
        fast_dev_run=False,
        # logger=logger,
        log_every_n_steps=5,
        callbacks=[
            ModelCheckpoint(
                monitor="loss/val",
                save_top_k=8,
                every_n_epochs=1,
                save_last=True
            )
        ],
    )
#%%
embeddings = trainer.predict(
    model=model,
    dataloaders=data_module.predict_dataloader(),
    return_predictions=True
)
#%%

#Get projection matrix
projection_matrix = np.zeros((len(data_module.dataset.metadata), embeddings[0][1].shape[1]))

current_datapoint = 0
for batch in embeddings:
    embeddings_batch, projection_batch = batch
    len_batch = embeddings_batch.shape[0]
    projection_matrix[current_datapoint:current_datapoint + len_batch] = projection_batch
    current_datapoint += len_batch

#%%
from sklearn.decomposition import PCA
# Perform PCA
pca = PCA(n_components=3)
pcs = pca.fit_transform(projection_matrix)

# pcs is of shape (num_datapoints, 3), containing the first 3 principal components
print("First 3 principal components shape:", pcs.shape)

#%%
metadata = data_module.dataset.metadata

# Add other info to metadata
#%%
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    pcs[:, 0], pcs[:, 1],
    c=metadata['gene_id'].astype('category').cat.codes,
    cmap='tab10',
    alpha=0.2
)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Projection onto First 2 Principal Components')
plt.grid(True)
cbar = plt.colorbar(scatter, ticks=range(len(metadata['gene_id'].unique())))
cbar.ax.set_yticklabels(metadata['gene_id'].astype('category').cat.categories)
plt.show()
#%%