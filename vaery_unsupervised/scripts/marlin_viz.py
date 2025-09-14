#%%
from pathlib import Path
import pickle
import glob
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
def perform_pca_on_embeddings(embedding_matrix):
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(embedding_matrix)
    return pcs



HEADPATH = Path("/mnt/efs/aimbl_2025/student_data/S-GL/")
embeddings_directory = HEADPATH / "embeddings/embeddings_custom_resnet_v1/"
# embeddings_directory = HEADPATH / "embeddings_30tr-per-gene_all-t_v21/"
metadata_path = HEADPATH / "2025-09-03_lDE20_Final_balanced_30tr-per-gene.pkl"

#%%
with open(embeddings_directory / "embeddings_full_matrix.pkl", "rb") as f:
    embeddings_data = pickle.load(f)

embedding_matrix = embeddings_data['embeddings']
metadata = embeddings_data['metadata'].reset_index()
#%%
metadata_w_indices = metadata
# %%
metadata = pd.read_pickle(metadata_path)
# %%
metadata_w_indices = pd.read_pickle(embeddings_directory / "metadata_with_indices.pkl")

# %% Load embedding_matrix.npy
embedding_matrix = np.load(embeddings_directory / "projection_matrix.npy")


#%% Get a subset of metadata
metadata_subset = metadata_w_indices[metadata_w_indices['Gene'].isin(['ftsN', 'rplQ', 'mreD', 'alaS', 'rplO', 'mreB'])]
embedding_subset = embedding_matrix[metadata_subset.index]
#%% Perform tSNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42, perplexity=20)
tsne_results = tsne.fit_transform(embedding_subset)

# %% Visualize tSNE results
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=metadata_subset['Gene'].astype('category').cat.codes)
plt.colorbar()
plt.title("tSNE visualization of selected gene embeddings")
plt.xlabel("tSNE Component 1")
plt.ylabel("tSNE Component 2")
plt.show()

#%% Visualize tSNE results for timepoints > 50
mask = (metadata_subset['timepoints'] > 50) & (metadata_subset['timepoints'] < 59)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    tsne_results[mask, 0],
    tsne_results[mask, 1],
    c=metadata_subset.loc[mask, 'Gene'].astype('category').cat.codes,
    cmap='jet'
)
plt.colorbar()
plt.title("tSNE visualization (timepoints > 50)")
plt.xlabel("tSNE Component 1")
plt.ylabel("tSNE Component 2")

# Add legend with gene names
gene_codes = metadata_subset.loc[mask, 'Gene'].astype('category').cat.codes
gene_names = metadata_subset.loc[mask, 'Gene'].astype('category').cat.categories
handles = [
    plt.Line2D([0], [0], marker='o', color='w', label=gene, 
               markerfacecolor=scatter.cmap(scatter.norm(code)), markersize=8)
    for gene, code in zip(gene_names, range(len(gene_names)))
]
plt.legend(handles=handles, title="Gene", loc="best")

plt.show()

#%% Same as above but color according to the column 'timepoints' using the jet colormap
plt.figure(figsize=(8, 6))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=metadata_subset['timepoints'], cmap='jet')
plt.colorbar()
plt.title("tSNE visualization of selected gene embeddings colored by timepoints")
plt.xlabel("tSNE Component 1")
plt.ylabel("tSNE Component 2")
plt.show()


# Visualize tSNE results for only the gene 'mreD'
mreD_mask = metadata_subset['Gene'] == 'mreD'
plt.figure(figsize=(8, 6))
plt.scatter(
    tsne_results[mreD_mask, 0],
    tsne_results[mreD_mask, 1],
    c=metadata_subset.loc[mreD_mask, 'timepoints'],
    cmap='jet'
)
plt.colorbar(label='Timepoints')
plt.title("tSNE visualization for gene 'mreD' colored by timepoints")
plt.xlabel("tSNE Component 1")
plt.ylabel("tSNE Component 2")
plt.show()

# %%
