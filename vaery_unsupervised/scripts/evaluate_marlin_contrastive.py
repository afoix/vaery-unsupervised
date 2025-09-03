#%%
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
checkpoint_path = LOGS_HEADPATH / 'version_10/checkpoints/last.ckpt'

marlin_encoder_config = {
    "backbone": "resnet18",
    "in_channels": 1,
    "spatial_dims": 2,
    "embedding_dim": 512,
    "mlp_hidden_dims": 768,
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
    transforms=transforms,
)
# data_module.eval()

#%%
data_module._prepare_data()
data_module.setup(stage='predict')
#%%
dataloader = data_module.predict_dataloader()
imgs_anchor = []
imgs_positive = []
for i, batch in enumerate(dataloader):
    img_anchor = batch['anchor'][0]
    img_positive = batch['positive'][0]
    imgs_anchor.append(img_anchor)
    imgs_positive.append(img_positive)
    if i == 20:
        break
#%%

#%%
emb_anchor, proj_anchor = model(img_anchor.to(device).unsqueeze(0))
emb_positive, proj_positive = model(img_positive.to(device).unsqueeze(0))
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 10))
print(F.cosine_similarity(proj_anchor, proj_positive))
fig.patch.set_facecolor('black')
ax[0].imshow(img_anchor.numpy()[0], cmap='gray')
ax[1].imshow(img_positive.numpy()[0], cmap='gray')
#%% Get embeddings
index = 1
emb_anchor, proj_anchor = model(img_anchor[index].to(device).unsqueeze(0))
emb_positive, proj_positive = model(img_positive[index].to(device).unsqueeze(0))
# emb_2, proj_2 = model(second)
#%%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 10))
print(F.cosine_similarity(proj_anchor, proj_positive))
fig.patch.set_facecolor('black')
ax[0].imshow(batch['anchor'].numpy()[0, 0], cmap='gray')
ax[1].imshow(batch['positive'].numpy()[0, 0], cmap='gray')

#%%
single_anchor = dataloader.dataset[0]['anchor']
single_positive = dataloader.dataset[0]['positive']
#%%
emb_single_anchor, proj_single_anchor = model(torch.from_numpy(single_anchor).unsqueeze(0).to(device))
emb_single_positive, proj_single_positive = model(torch.from_numpy(single_positive).unsqueeze(0).to(device))
#%%
single_anchor

#%%
import time
import matplotlib.pyplot as plt
# for batch in dataloader:
fig, ax = plt.subplots(1, 2, figsize=(10, 10))
fig.patch.set_facecolor('black')
print(F.cosine_similarity(emb_anchor, emb_positive))
ax[0].imshow(single_anchor[0], cmap='gray')
ax[1].imshow(single_positive[0], cmap='gray')
# # Pause for 2 seconds
# plt.show()
# # time.sleep(1)
# plt.close()
