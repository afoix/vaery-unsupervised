#%% testing
import numpy as np
from pathlib import Path
from vaery_unsupervised.dataloaders.dataloader_km_ryans_template import (
    simple_masking, 
    DATASET_NORM_DICT, 
    SpatProteoDatasetZarr,
    SpatProtoZarrDataModule,
)
from vaery_unsupervised.networks.LightningVAE_linear_km import SpatialVAE_Linear
from vaery_unsupervised.networks.km_ryan_linearresnet import (ResNet18Dec, ResNet18Enc)
import yaml
from vaery_unsupervised.km_utils import plot_batch_sample,plot_dataloader_output
import monai.transforms as transforms
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path

dataset_path = Path("/mnt/efs/aimbl_2025/student_data/S-KM/")


dataset_zarr = SpatProteoDatasetZarr(
    dataset_path/"converted_crops_with_metadata.zarr",
    masking_function=simple_masking,
    dataset_normalisation_dict=DATASET_NORM_DICT,
    transform_both=None,
    transform_input=None
)
plot_dataloader_output(dataset_zarr[0])
# %%
lightning_module = SpatProtoZarrDataModule(
    dataset_path/"converted_crops_with_metadata.zarr",
    masking_function=simple_masking,
    dataset_normalisation_dict=DATASET_NORM_DICT,
    transform_both=None,
    transform_input=None,
    num_workers=8,
    batch_size=16,
)
lightning_module.setup("predict")

loader = lightning_module.predict_dataloader()
#%%
for batch in loader:
    batch = batch
    break

#%%

#%%
checkpoint_path = "/mnt/efs/aimbl_2025/student_data/S-KM/logs/linear_VAE_latentsize_512/version_0/checkpoints/epoch=97-step=4214.ckpt"
model = SpatialVAE_Linear.load_from_checkpoint(checkpoint_path=checkpoint_path, strict = True)

#%% Compact model loading with error handling

#%%
import torch

#%%
from vaery_unsupervised.networks.LightningVAE_linear_km import reparameterize


# %%
for batch in loader:
    image_ids = batch["metadata"]['well_id']
    input = batch["input"].to(device = model.device)
    reconstruction, z_mean, z_log_var = model(input[:,model.channels_selection,:,:])
    z = reparameterize(z_mean, z_log_var)
    break


# %%
all_image_ids = []
all_input = []
all_reconstruction = []
all_z_mean = []
all_z_log_var = []
all_z = []
#%%
model.eval()
#%%
for batch in loader:
    image_ids = batch["metadata"]['well_id']
    input = batch["input"][:,model.channels_selection,:,:].to(device = model.device)
    reconstruction, z_mean, z_log_var = model(input)  
    z = reparameterize(z_mean, z_log_var)

    all_image_ids.append(image_ids) 
    all_input.append(input.detach().cpu())
    all_reconstruction.append(reconstruction.detach().cpu())
    all_z_mean.append(z_mean.detach().cpu())
    all_z_log_var.append(z_log_var.detach().cpu())
    all_z.append(z.detach().cpu())

all_image_ids = np.concatenate(all_image_ids, axis = 0) 
import torch
all_input = torch.cat(all_input, dim = 0).numpy()
all_reconstruction = torch.cat(all_reconstruction, dim = 0).numpy() 
all_z_mean = torch.cat(all_z_mean, dim = 0).numpy() 
all_z_log_var = torch.cat(all_z_log_var, dim = 0).numpy() 
all_z = torch.cat(all_z, dim = 0).numpy()
    
#%%
from vaery_unsupervised.plotting_utils import *
from sklearn.decomposition import PCA

#%%
df = pd.DataFrame(data = all_z, index = all_image_ids, columns = np.arange(1,all_z.shape[1]+1))
#%%
import matplotlib.pyplot as plt
import plotly.express as px
import anndata as ad
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from umap.umap_ import UMAP
#%%
#%%
metadata_ad = ad.read_h5ad("/mnt/efs/aimbl_2025/student_data/S-KM/001_Used_Zarrs/fullanndata_batch1")

metadata_ai_ad = ad.read_h5ad("/mnt/efs/aimbl_2025/student_data/S-KM/001_Used_Zarrs/fullanndata_aimbl")
#%%
metadata_ad.obs[f'ANXA2_log2'] = metadata_ad[:, metadata_ad.var_names == "ANXA2"].layers["log2"].flatten()
metadata_ai_ad.obs[f'ANXA2_log2'] = metadata_ai_ad[:, metadata_ai_ad.var_names == "ANXA2"].layers["log2"].flatten()
#%%
metadata = pd.DataFrame(metadata_ad.obs)
metadata_ai = pd.DataFrame(metadata_ai_ad.obs)
#%%
anxa2 = metadata_ad.var_names[metadata_ad.var_names == "ANXA2"]
anxa2
#%%
df_z = pd.DataFrame(data = all_z, index=all_image_ids, columns = np.arange(1,all_z.shape[1]+1))
scaler = StandardScaler()
df_z_scaled = scaler.fit_transform(df_z)

#%%
umap_z = UMAP(n_neighbors = 15, n_components=3).fit_transform(df_z_scaled)
umap_z = pd.DataFrame(umap_z, index=all_image_ids)
umap_z_columnnames = [f'umap{i}' for i in np.arange(1,umap_z.shape[1]+1)]
umap_z.columns = umap_z_columnnames
umap_z = umap_z.reset_index().rename(columns = {"index":"plate_id"})
fig = plt.scatter(x = umap_z["umap1"], y = umap_z["umap2"])
plt.show(fig)

#%%
umap_z = umap_z.merge(metadata[["trajectory", "area", "distance_to_SE", "plate_id", "pg_cnt", "ANXA2_log2", "patient_id"]], 
                    how="left", on="plate_id")
umap_z = umap_z.merge(metadata_ai[["plate_id", "pg_cnt", "ANXA2_log2", "patient_id"]], 
                    how="left", on="plate_id", suffixes=('', '_ai'))

# Fill NaN values in original columns with values from _ai columns
umap_z['pg_cnt'] = umap_z['pg_cnt'].fillna(umap_z['pg_cnt_ai'])
umap_z['ANXA2_log2'] = umap_z['ANXA2_log2'].fillna(umap_z['ANXA2_log2_ai'])
umap_z['patient_id'] = umap_z['patient_id'].astype("str").fillna(umap_z['patient_id_ai'])

# Drop the duplicate columns
umap_z = umap_z.drop(columns=['pg_cnt_ai', 'ANXA2_log2_ai'])

#%%
fig = px.scatter(umap_z, x='umap1', y='umap2', hover_data = "plate_id", color = "area")
fig.show()


#%%
tsne_z = TSNE(n_components=3).fit_transform(df_z_scaled, y=None)
tsne_z = pd.DataFrame(tsne_z, index=all_image_ids)
tsne_z_columnnames = [f'tsne{i}' for i in np.arange(1,tsne_z.shape[1]+1)]
tsne_z.columns = tsne_z_columnnames
tsne_z = tsne_z.reset_index().rename(columns = {"index":"plate_id"})
fig = plt.scatter(x = tsne_z["tsne1"], y = tsne_z["tsne2"])
plt.show(fig)

#%%
tsne_z = tsne_z.merge(metadata[["trajectory", "area", "distance_to_SE", "plate_id", "pg_cnt", "ANXA2_log2", "patient_id"]], 
                    how="left", on="plate_id")
tsne_z = tsne_z.merge(metadata_ai[["plate_id", "pg_cnt", "ANXA2_log2", "patient_id"]], 
                    how="left", on="plate_id", suffixes=('', '_ai'))

# Fill NaN values in original columns with values from _ai columns
tsne_z['pg_cnt'] = tsne_z['pg_cnt'].fillna(tsne_z['pg_cnt_ai'])
tsne_z['ANXA2_log2'] = tsne_z['ANXA2_log2'].fillna(tsne_z['ANXA2_log2_ai'])
tsne_z['patient_id'] = tsne_z['patient_id'].astype("str").fillna(tsne_z['patient_id_ai'])

# Drop the duplicate columns
tsne_z = tsne_z.drop(columns=['pg_cnt_ai', 'ANXA2_log2_ai', "patient_id_ai"])

#%%
fig = px.scatter(tsne_z, x='tsne1', y='tsne2', hover_data = "plate_id", color = "area")
fig.show()


#%%
pca_z = PCA(n_components=3).fit_transform(df_z_scaled, y=None)
pca_z = pd.DataFrame(pca_z, index=all_image_ids)
pca_z_columnnames = [f'pc{i}' for i in np.arange(1,pca_z.shape[1]+1)]
pca_z.columns = pca_z_columnnames
pca_z = pca_z.reset_index().rename(columns = {"index":"plate_id"})
fig = plt.scatter(x = pca_z["pc1"], y = pca_z["pc2"])
plt.show(fig)

#%%
pca_z = pca_z.merge(metadata[["trajectory", "area", "distance_to_SE", "plate_id", "pg_cnt", "ANXA2_log2"]], 
                    how="left", on="plate_id")
pca_z = pca_z.merge(metadata_ai[["plate_id", "pg_cnt", "ANXA2_log2"]], 
                    how="left", on="plate_id", suffixes=('', '_ai'))

# Fill NaN values in original columns with values from _ai columns
pca_z['pg_cnt'] = pca_z['pg_cnt'].fillna(pca_z['pg_cnt_ai'])
pca_z['ANXA2_log2'] = pca_z['ANXA2_log2'].fillna(pca_z['ANXA2_log2_ai'])

# Drop the duplicate columns
pca_z = pca_z.drop(columns=['pg_cnt_ai', 'ANXA2_log2_ai'])
#%%
fig = px.scatter(pca_z, x='pc1', y='pc2', hover_data = "plate_id", color = "area")
fig.show()

#%%


#%%
fig = px.scatter_3d(pca_z, x='pc1', y='pc2', z= "pc3", hover_data = "plate_id", color = "area")


# %%
import plotly.io as pio
pio.renderers.default = "browser"  # or "png", "svg", etc.
fig.show()

#%%
pca = pca.reset_index().rename(columns = {"index":"label"})
#
#colors = [color_map[mapping[label]] for label in labels]

import pandas as pd
import io
import base64
import seaborn as sns
from dash import dcc, html, Input, Output, no_update, Dash
import plotly.graph_objects as go

from PIL import Image
import numpy as np
from matplotlib.colors import to_hex

#%%
pca
#%%
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
fig.show()


#%%
pca_fixed = pca.copy()
pca_fixed['label'] = pca_fixed['label'].astype(str)
#%%
import anndata as ad
#%%
from vaery_unsupervised.plotting_utils import *

#%%
metadata_batch1 = pd.DataFrame(metadata.obs)

#%%
tsne_z["trajectory"] = tsne_z["trajectory"].astype("str").fillna("NaN")
#%%
images_converted = (
    np.moveaxis(
        all_reconstruction/np.max(all_reconstruction, axis = (0,2,3))[np.newaxis,:,np.newaxis,np.newaxis], 1,-1) *255
    ).astype("uint8")

images_converted = (
    np.moveaxis(all_input, 1,-1)).astype("uint8")

#%%
app = get_dash_app_3D_scatter_hover_images(
    umap_z, 
    hue="patient_id", 
    images=images_converted, 
    plot_keys=["umap1", "umap2", "umap3"]
)



# %%
import webbrowser
import threading
import time

def open_browser():
    time.sleep(1.5)  # Wait for server to start
    webbrowser.open_new('http://127.0.0.1:8050/')

# Start browser in a separate thread
threading.Thread(target=open_browser).start()

app.run(port=8050)
# %%

# %%
