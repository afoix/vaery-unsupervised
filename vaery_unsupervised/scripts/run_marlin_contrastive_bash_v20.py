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
from vaery_unsupervised.networks.marlin_resnet import SmallObjectResNet10Encoder
from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss
from lightning.pytorch.loggers import TensorBoardLogger
from monai.transforms import (
    Compose,
    NormalizeIntensity,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSharpen,
    RandGaussianSmoothd
)
MEAN_OVER_DATASET = 13
STD_OVER_DATASET = 11

HEADPATH = Path('/mnt/efs/aimbl_2025/student_data/S-GL/')
# METADATA_PATH = HEADPATH / '2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded.pkl'
# METADATA_COMPACT_PATH  = HEADPATH / '2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded_filtered_266-trenches.pkl'
# METADATA_COMPACT_PATH  = HEADPATH / '2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded_select_grnas_allT.pickle'
METADATA_BALANCED_PATH = HEADPATH / '2025-09-03_lDE20_Final_balanced_30tr-per-gene.pkl'

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
        prob=0.5,
    ),
    RandFlipd(
        keys=["positive"],
        spatial_axis=1,
        prob=0.5,
    ),
    RandGaussianNoised(
        keys=["positive"],
        prob=0.5,
        mean=0,
        std=0.5,
        sample_std=True,
    ),
    
    RandGaussianSmoothd(
        keys=["positive"],
        sigma_x=(0.5, 1.5),
        sigma_y=(0.5, 1.5),
        prob=0.5
    )
])

def main(*args, **kwargs):

    # marlin_encoder_config = {
    # "backbone": "resnet18",
    # "in_channels": 1,
    # "spatial_dims": 2,
    # "embedding_dim": 512,
    # "mlp_hidden_dims": 256,#768,
    # "projection_dim": 32,
    # "pretrained": False,
    # }

    marlin_encoder = SmallObjectResNet10Encoder(
        in_channels=1,
        widths=(32, 64, 128, 256),   # slim to reduce overfitting
        feature_stage="layer4",      # use "layer3" if you want even fewer params
        layer4_dilate=3,             # captures long rods without further downsampling
        norm="batch", gn_groups=8,   # better than BN for small batches
        drop_path_rate=0.05,         # mild stochastic depth
        mlp_hidden_dims=512,         # set = embedding_dim if you prefer
        projection_dim=128,
    )

    # marlin_encoder = ResNetEncoder(**marlin_encoder_config)
    # marlin_encoder = CustomResNetEncoder(**marlin_encoder_config)
    marlin_contrastive_config = {
        "encoder": marlin_encoder,
        "loss": SelfSupervisedLoss(NTXentLoss(temperature=0.07)),
        "lr": 1e-3,#1e-3,
    }

    marlin_contrastive = ContrastiveModule(**marlin_contrastive_config)


    data_module = MarlinDataModule(
        data_path=HEADPATH/'Ecoli_lDE20_Exps-0-1/',
        metadata_path=METADATA_BALANCED_PATH,
        split_ratio=0.8,
        batch_size=128+64,#256,#256,
        num_workers=8,
        prefetch_factor=2,
        transforms=transforms,
    )
    data_module._prepare_data()
    data_module.setup(stage='fit')
    
    logger = TensorBoardLogger(
        save_dir=HEADPATH/"tb_logs",
        name="marlin_contrastive",
    )

    trainer = L.Trainer(
        devices="auto",
        accelerator="gpu",
        strategy="auto",
        precision="16-mixed",
        max_epochs=1000,
        fast_dev_run=False,
        logger=logger,
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

    trainer.fit(
        model = marlin_contrastive,
        datamodule = data_module
    )

# Run main
if __name__ == "__main__":
    main()

    