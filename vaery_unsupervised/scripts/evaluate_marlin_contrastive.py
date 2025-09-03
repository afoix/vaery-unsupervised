#%%
from lightning import LightningModule
from pathlib import Path

import torch
from vaery_unsupervised.networks.marlin_contrastive import ContrastiveModule, ResNetEncoder
from monai.networks.nets.resnet import ResNetFeatures
from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss


LOGS_HEADPATH = Path('/mnt/efs/aimbl_2025/student_data/S-GL/tb_logs/marlin_contrastive/')
checkpoint_path = LOGS_HEADPATH / 'version_4/checkpoints/last.ckpt'

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

#%%
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
data_module = MarlinDataModule(
    data_path=HEADPATH/'Ecoli_lDE20_Exps-0-1/',
    metadata_path=METADATA_COMPACT_PATH,
    split_ratio=0.8,
    batch_size=32,
    num_workers=4,
    prefetch_factor=2,
    transforms=None
)

data_module._prepare_data()