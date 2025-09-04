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

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from vaery_unsupervised.dataloaders.marlin_dataloader.marlin_dataloader import MarlinDataModule, MarlinDataset
from vaery_unsupervised.networks.marlin_contrastive import ResNetEncoder, ContrastiveModule
from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss
import pickle
from vaery_unsupervised.dataloaders.marlin_dataloader.marlin_dataloader import MarlinDataModule
from sklearn.decomposition import PCA

# Get the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOGS_HEADPATH = Path('/mnt/efs/aimbl_2025/student_data/S-GL/tb_logs/marlin_contrastive/')
checkpoint_path = LOGS_HEADPATH / 'version_7/checkpoints/last.ckpt'

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
# marlin_encoder = ResNetEncoder(**marlin_encoder_config)
loss = SelfSupervisedLoss(NTXentLoss(temperature=0.07))
lr = 1e-3
optimizer = torch.optim.Adam(marlin_encoder.parameters(), lr=lr)

#%%
model = ContrastiveModule.load_from_checkpoint(
    checkpoint_path,
    encoder=marlin_encoder,
    loss=loss,
    lr=lr,
    optimizer=optimizer,
)
model.to(device)
model.eval()
#%%

HEADPATH = Path('/mnt/efs/aimbl_2025/student_data/S-GL/')
METADATA_PATH = HEADPATH / '2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded.pkl'
METADATA_COMPACT_PATH  = HEADPATH / '2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded_select_grnas_allT.pickle'
EMBEDDINGS_PATH = HEADPATH / 'embeddings_4gene_allt_v1.pkl'
#%%
data_module = MarlinDataModule(
    data_path=HEADPATH/'Ecoli_lDE20_Exps-0-1/',
    metadata_path=METADATA_COMPACT_PATH,
    split_ratio=0.8,
    batch_size=1024,#64,
    num_workers=4,
    prefetch_factor=2,
    transforms=None,
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

embeddings = trainer.predict(
    model=model,
    dataloaders=data_module.predict_dataloader(),
    return_predictions=True
)

#%%

with open(EMBEDDINGS_PATH, "wb") as f:
    pickle.dump(embeddings, f)

#%%
EMBEDDINGS_PATH = HEADPATH / 'embeddings_4gene_allt_v1.pkl'
with open(EMBEDDINGS_PATH, "rb") as f:
    embeddings = pickle.load(f)


#%%


#%%
#Get projection matrix

projection_matrix = np.zeros((len(data_module.dataset.metadata), embeddings[0][1].shape[1]))
embedding_matrix = np.zeros((len(data_module.dataset.metadata), embeddings[0][0].shape[1]))

#%%
current_datapoint = 0
for batch in embeddings:
    embeddings_batch, projection_batch = batch
    len_batch = embeddings_batch.shape[0]
    projection_matrix[current_datapoint:current_datapoint + len_batch] = projection_batch
    embedding_matrix[current_datapoint:current_datapoint + len_batch] = embeddings_batch
    current_datapoint += len_batch

#%%
def random_subset_embeddings_and_metadata(
    embedding_matrix,
    projection_matrix,
    metadata,
    n_samples=1000
):
    total_samples = embedding_matrix.shape[0]
    subset_indices = np.random.choice(total_samples, n_samples, replace=False)
    subset_embeddings = embedding_matrix[subset_indices]
    subset_projection = projection_matrix[subset_indices]
    subset_metadata = metadata.iloc[subset_indices]
    return subset_embeddings, subset_projection, subset_metadata

def subset_late_timepoints_embeddings_and_metadata(
    embedding_matrix,
    projection_matrix,
    metadata,
    timepoint_threshold=45
):
    late_timepoint_mask = metadata['timepoints'] > timepoint_threshold
    subset_embeddings = embedding_matrix[late_timepoint_mask]
    subset_projection = projection_matrix[late_timepoint_mask]
    subset_metadata = metadata[late_timepoint_mask]
    return subset_embeddings, subset_projection, subset_metadata

from sklearn.decomposition import PCA
def perform_pca_on_embeddings(embedding_matrix):
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(embedding_matrix)
    return pcs

#%%
pcs_emb = perform_pca_on_embeddings(embedding_matrix)
pcs_proj = perform_pca_on_embeddings(projection_matrix)

#%%
random_subset_pcs_emb, random_subset_pcs_proj, random_subset_metadata = random_subset_embeddings_and_metadata(
    pcs_emb,
    pcs_proj,
    data_module.dataset.metadata,
    n_samples=5000
)

late_timepoint_pcs_emb, late_timepoint_pcs_proj, late_timepoint_metadata = subset_late_timepoints_embeddings_and_metadata(
    pcs_emb,
    pcs_proj,
    data_module.dataset.metadata,
    timepoint_threshold=50
)

random_subset_late_timepoint_pcs_emb, random_subset_late_timepoint_pcs_proj, random_subset_late_timepoint_metadata = random_subset_embeddings_and_metadata(
    late_timepoint_pcs_emb,
    late_timepoint_pcs_proj,
    late_timepoint_metadata,
    n_samples=5000
)

#%%
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    random_subset_pcs_emb[:, 0], random_subset_pcs_emb[:, 1],
    c=random_subset_metadata['gene_id'].astype('category').cat.codes,
    cmap='tab10',
    alpha=0.5
)

#%%
fig,axs = plt.subplots(1, 4, figsize=(20, 6))
scatter1 = axs[0].scatter(
    random_subset_pcs_emb[:, 0], random_subset_pcs_emb[:, 1],
    c=random_subset_metadata['gene_id'].astype('category').cat.codes,
    cmap='tab10',
    alpha=0.5
)
axs[0].set_title('PCA on Embeddings')
axs[0].set_xlabel('PC1')
axs[0].set_ylabel('PC2')

scatter2 = axs[1].scatter(
    random_subset_pcs_emb[:, 0], random_subset_pcs_emb[:, 1],
    c=random_subset_metadata['timepoints'],
    cmap='jet',
    alpha=0.5
)
axs[1].set_title('PCA on Projections'); axs[1].set_xlabel('PC1'); axs[1].set_ylabel('PC2')

scatter3 = axs[2].scatter(
    random_subset_late_timepoint_pcs_emb[:, 0], random_subset_late_timepoint_pcs_emb[:, 1],
    c=random_subset_late_timepoint_metadata['gene_id'].astype('category').cat.codes,
    cmap='tab10',
    alpha=0.5
)

#%%
from sklearn.manifold import TSNE
def perform_tsne_on_embeddings(embedding_matrix):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(embedding_matrix)
    return tsne_results

#%%
late_emb, late_proj, late_metadata = subset_late_timepoints_embeddings_and_metadata(
    embedding_matrix,
    projection_matrix,
    data_module.dataset.metadata,
    timepoint_threshold=50
)

sub_late_emb, sub_late_proj, sub_late_metadata = random_subset_embeddings_and_metadata(
    late_emb,
    late_proj,
    late_metadata,
    n_samples=5000
)
#%%
mask_last_t = sub_late_metadata['timepoints'] > 58
last_t_emb = sub_late_emb[mask_last_t]
last_t_proj = sub_late_proj[mask_last_t]
last_t_metadata = sub_late_metadata[mask_last_t]

# last_t_emb, last_t_proj, last_t_metadata = subset_late_timepoints_embeddings_and_metadata(
#     embedding_matrix,
#     projection_matrix,
#     data_module.dataset.metadata,
#     timepoint_threshold=58
# )



# last_t_tsne_emb = perform_tsne_on_embeddings(last_t_emb)

#%%

tsne_late_emb = perform_tsne_on_embeddings(sub_late_emb)

#%%
tsne_late_emb_last = tsne_late_emb[mask_last_t]
#%%
fig, ax = plt.subplots(1, 3, figsize=(12, 6))
scatter1 = ax[0].scatter(
    tsne_late_emb[:, 0], tsne_late_emb[:, 1],
    c=sub_late_metadata['gene_id'].astype('category').cat.codes,
    cmap='tab10',
    alpha=0.5
)
ax[0].set_title('tSNE on Late Timepoint Embeddings')
ax[0].set_xlabel('tSNE-1')
ax[0].set_ylabel('tSNE-2')
scatter2 = ax[1].scatter(
    tsne_late_emb[:, 0], tsne_late_emb[:, 1],
    c=sub_late_metadata['timepoints'],
    cmap='jet',
    alpha=0.5
)
ax[1].set_title('tSNE on Late Timepoint Embeddings Colored by Timepoint')
ax[1].set_xlabel('tSNE-1')
ax[1].set_ylabel('tSNE-2')

scatter3 = ax[2].scatter(
    tsne_late_emb_last[:, 0], tsne_late_emb_last[:, 1],
    c=last_t_metadata['gene_id'].astype('category').cat.codes,
    cmap='tab10',
    alpha=0.5
)

# Show cmap for scatter 3
cbar = plt.colorbar(scatter3, ticks=range(len(last_t_metadata['gene_id'].unique())))
cbar.ax.set_yticklabels(last_t_metadata['gene_id'].astype('category').cat.categories)
plt.show()
#%%
# Get unique gene_ids with gene_names
unique_gene_ids = data_module.dataset.metadata[['gene_id', 'Gene']].drop_duplicates().reset_index(drop=True)
#%%

N = 1000  # Number of elements to use for tSNE
subset_indices = np.random.choice(embedding_matrix.shape[0], N, replace=False)
embedding_subset = embedding_matrix[subset_indices]
metadata_subset = metadata.iloc[subset_indices]

# Perform tSNE
tsne = TSNE(n_components=2, random_state=42, perplexity=50)
tsne_results = tsne.fit_transform(embedding_subset)

print("tSNE results shape:", tsne_results.shape)
#%%
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    tsne_results[:, 0], tsne_results[:, 1],
    c=metadata_subset['gene_id'].astype('category').cat.codes,
    cmap='tab10',
    alpha=0.8
)
plt.xlabel('tSNE-1')
plt.ylabel('tSNE-2')
plt.title(f'tSNE Projection (N={N})')
plt.grid(True)
cbar = plt.colorbar(scatter, ticks=range(len(metadata_subset['gene_id'].unique())))
cbar.ax.set_yticklabels(metadata_subset['gene_id'].astype('category').cat.categories)
plt.show()

#%%
metadata_complete = pd.read_pickle(METADATA_COMPACT_PATH)
metadata_complete[metadata_complete['gene_id'].isin(metadata['gene_id'].unique())].groupby('gene_id').first()

#%%
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    tsne_results[:, 0], tsne_results[:, 1],
    c=metadata_subset['timepoints'],
    cmap='jet',
    alpha=0.8
)
plt.xlabel('tSNE-1')
plt.ylabel('tSNE-2')
plt.title(f'tSNE Projection Colored by Timepoint (N={N})')
plt.grid(True)
cbar = plt.colorbar(scatter)
cbar.set_label('Timepoint')
plt.show()

#%
