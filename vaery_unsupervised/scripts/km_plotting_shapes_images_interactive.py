#%% testing
import numpy as np
from pathlib import Path
from vaery_unsupervised.dataloaders.dataloader_km_ryans_template import (
    simple_masking, 
    DATASET_NORM_DICT, 
    SpatProteoDatasetZarr,
    SpatProtoZarrDataModule,
)
import torch
from vaery_unsupervised.networks.LightningVAE_linear_km import SpatialVAE_Linear, reparameterize
from vaery_unsupervised.km_utils import plot_batch_sample,plot_dataloader_output
from pathlib import Path
from vaery_unsupervised.plotting_utils import plot_pca_reconstructions
import matplotlib.pyplot as plt
import pandas as pd

data_loc = Path("/mnt/efs/aimbl_2025/student_data/S-KM")
latents = np.load(data_loc/"test_latent_space.npy")
latents
#%%
labels_plates =np.load(data_loc/"test_labels.npy")
labels_plates
#%%
import os
names_recon_shape = [n.split("_")[2] +"_" + n.split("_")[3] +"_" + n.split("_")[4] for n in os.listdir(data_loc/"contours_anna/reconstructed_contours")]
names_recon_shape
#%%
shape_df = pd.DataFrame(latents,columns=[f"latent_{i}" for i in range(latents.shape[1])])
shape_df["plate_idx"] = labels_plates
shape_df["well_id"] = names_recon_shape
shape_df
# %%
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap.umap_ import UMAP
umap = UMAP(n_neighbors = 15, n_components=2).fit_transform(latents)
tsne = TSNE().fit_transform(latents)
pca = PCA(n_components=3).fit_transform(latents)
shape_df[["TSNE1","TSNE2"]]=tsne
shape_df[["PCA_0","PCA_1","PCA_2"]]=pca
shape_df[["UMAP1","UMAP2"]]=umap
shape_df
# %%
import seaborn as sns
sns.scatterplot(shape_df,x="TSNE1",y="TSNE2",hue="plate_idx")
# %%
fig, axs = plt.subplots(4,4,figsize = (12,12))
hues = [] # choose 16 hues
for ax, hue in zip(axs.flatten(),hues):
    sns.scatterplot(shape_df,x="TSNE1",y="TSNE2",hue=hue,ax =ax)
    ax.set_title(hue)
#%%
import zarr
import json
zarr_path = Path("/mnt/efs/aimbl_2025/student_data/S-KM/converted_crops_with_metadata.zarr")
all_imgs = np.array(zarr.open_group(zarr_path)["crop_128_px"])
with open(zarr_path/"metadata_crop_128_px.json",'r')as f:
    metadata_imgs = json.load(f)
mapping_well_to_img_idx = {
    well_id:i for i,well_id in enumerate(metadata_imgs["well_id"])
}
img_indices = [mapping_well_to_img_idx[well_id] for well_id in shape_df["well_id"]]
images_sorted = all_imgs[img_indices][:,[1,2,3],:,:]/ all_imgs[img_indices][:,[1,2,3],:,:].max(axis = (2,3))[:,:,np.newaxis,np.newaxis]
images_sorted = np.moveaxis(images_sorted,1,-1)  * all_imgs[img_indices][:,4,:,:,np.newaxis]
# %%
from vaery_unsupervised.plotting_utils import get_dash_app_2D_scatter_hover_images
app = get_dash_app_2D_scatter_hover_images(shape_df,plot_keys=["PCA_0","PCA_1"],hue="plate_idx", images=(images_sorted*255).astype('uint8'))
app.run(
    port=6009
)
#%%
all_imgs.max(axis = (2,3)).shape
# %%
images_sorted.shape
# %%

# %%
