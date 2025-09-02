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
)

MEAN_OVER_DATASET = 13
STD_OVER_DATASET = 11

HEADPATH = Path('/mnt/efs/aimbl_2025/student_data/S-GL/')
METADATA_PATH = HEADPATH / '2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded.pkl'
METADATA_COMPACT_PATH  = HEADPATH / '2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded_compact.pkl'

transforms = Compose([
    NormalizeIntensity(
        subtrahend=MEAN_OVER_DATASET,
        divisor=STD_OVER_DATASET,
        nonzero=False,
        channel_wise=False,
        dtype = np.float32,
    )
])

def main(*args, **kwargs):

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
    marlin_contrastive_config = {
        "encoder": marlin_encoder,
        "loss": SelfSupervisedLoss(NTXentLoss(temperature=0.07)),
        "lr": 1e-3,
    }

    marlin_contrastive = ContrastiveModule(**marlin_contrastive_config)


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
    
    logger = TensorBoardLogger(
        save_dir=HEADPATH/"tb_logs",
        name="marlin_contrastive",
    )

    trainer = L.Trainer(
        devices="auto",
        accelerator="gpu",
        strategy="auto",
        precision="16-mixed",
        max_epochs=100,
        fast_dev_run=False,
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                monitor="loss/val",
                save_top_k=8,
                every_n_epochs=1,
                save_last=True
            )
        ]
    )

    trainer.fit(
        model = marlin_contrastive,
        train_dataloaders = data_module.train_dataloader()
    )

# Run main
if __name__ == "__main__":
    main()