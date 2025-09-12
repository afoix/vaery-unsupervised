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
def get_inference(model,dataloader):
    all_metadata = []
    all_input = []
    all_reconstruction = []
    all_z_mean = []
    all_z_log_var = []
    all_z = []
    for batch in dataloader:
        all_metadata.append(pd.DataFrame(batch["metadata"]))
        input = batch["input"][:,model.channels_selection,:,:].to(device = model.device)
        reconstruction, z_mean, z_log_var = model(input)  
        z = reparameterize(z_mean, z_log_var)
        all_input.append(input.detach().cpu())
        all_reconstruction.append(reconstruction.detach().cpu())
        all_z_mean.append(z_mean.detach().cpu())
        all_z_log_var.append(z_log_var.detach().cpu())
        all_z.append(z.detach().cpu())

    all_metadata = pd.concat(all_metadata,axis=0,ignore_index=True)

    all_input = torch.cat(all_input, dim = 0).numpy()
    all_reconstruction = torch.cat(all_reconstruction, dim = 0).numpy() 
    all_z_mean = torch.cat(all_z_mean, dim = 0).numpy() 
    all_z_log_var = torch.cat(all_z_log_var, dim = 0).numpy() 
    all_z = torch.cat(all_z, dim = 0).numpy()

    return all_z,all_reconstruction,all_input,all_metadata


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
checkpoint_path = "/mnt/efs/aimbl_2025/student_data/S-KM/logs/modelv2_1_perimagenorm_5epochs/version_0/checkpoints/epoch=47-step=1056.ckpt"
model = SpatialVAE_Linear.load_from_checkpoint(checkpoint_path=checkpoint_path, strict = True)

#%%


z,recons,inputs,metadata = get_inference(model,loader)

#%%
model.eval()


#%%
out = plot_pca_reconstructions(model.decode,model.device,latent_space=z,n_images=8,whiten=False)
fig,ax = plt.subplots(1,1,figsize=(10,10))
ax.imshow(out/out.max(axis=(0,1))[np.newaxis,np.newaxis,:])
ax.set_ylabel("PC_2")
ax.set_yticks([])
ax.set_xlabel("PC_1")
ax.set_xticks([])
# %%
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap.umap_ import UMAP
umap = UMAP(n_neighbors = 15, n_components=2).fit_transform(z)
tsne = TSNE().fit_transform(z)
pca = PCA(n_components=3).fit_transform(z)
metadata[["TSNE1","TSNE2"]]=tsne
metadata[["PCA_0","PCA_1","PCA_2"]]=pca
metadata[["UMAP1","UMAP2"]]=umap
metadata
#%%
inputs.shape
# %%
import seaborn as sns
sns.scatterplot(metadata,x="TSNE1",y="TSNE2",hue="dataset")
# %%
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn import datasets
writer = SummaryWriter("./embeddings_first_correct") #could save the embedding thing to the model checkpoint
images = inputs
# Example embeddings: 100 samples, 3-dimensional
embeddings = z

# Example metadata (labels for each embedding point)
labels = metadata["dataset"].to_numpy()
print(images.shape)
#%%
# # Save embeddings for TensorBoard projector
writer.add_embedding(
    mat=embeddings,
    metadata=labels,
    tag="example",
    label_img=torch.Tensor(images),
)

writer.close()
# %%
