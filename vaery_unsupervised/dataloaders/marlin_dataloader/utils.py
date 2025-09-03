#%%
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

def expand_metadata_on_time(
    metadata: pd.DataFrame,
    n_timepoints: int,
    cols_to_keep: list[str] = ['Multi-Experiment Phenotype Trenchid', 'oDEPool7_id', 
                'Gene', 'gene_id', 'Experiment #', 'grna_file_trench_index']
) -> pd.DataFrame:
    """
    Expand the metadata DataFrame to include all timepoints for each sample.
    """
    metadata_expanded = (metadata
        .loc[:, cols_to_keep]
        .reset_index()
        .rename(columns={'index': 'gene_grna_trench_index'})
        .apply(lambda x: x.repeat(n_timepoints).reset_index(drop=True))
        .assign(timepoints=lambda df_: df_.index % n_timepoints,
                is_last_timepoint=lambda df_: df_['timepoints'] == n_timepoints - 1
        )
        .astype(
            {'Experiment #': 'category'}
        )

    )

    metadata_expanded.index.name = 'gene_grna_trench_timepoint_index'
    return metadata_expanded

def compute_mean_and_std_over_sample(
    data_path:str,
    metadata:pd.DataFrame,
):
    '''
    Compute the mean and standard deviation of the fluorescence images over a sample of genes and gRNAs.
    '''
    COLS_TO_KEEP = ['gene_grna_trench_index', 'gene_id', 'oDEPool7_id', 'grna_file_trench_index']
    
    metadata_only_gene_grnas = (metadata
        .loc[:, COLS_TO_KEEP]
        .groupby(['gene_id', 'oDEPool7_id'])
        .first()
        .reset_index()
    )

    gene_ids_sample = np.random.choice(metadata_only_gene_grnas['gene_id'].unique(), size=20, replace=False)
    metadata_only_gene_sample = metadata_only_gene_grnas[metadata_only_gene_grnas['gene_id'].isin(gene_ids_sample)]
    # Go through all images indexed on that table

    n_images = 0
    weighted_mean = 0
    weighted_std = 0

    for i, row in metadata_only_gene_sample.iterrows():
        print(i)
        gene_id = row['gene_id']
        grna_id = row['oDEPool7_id']
        filename = data_path / f"{gene_id}/{grna_id}.hdf5"
        with h5py.File(filename, 'r') as f:
            img_fl = f[KEY_FL]
            mean_ = np.mean(img_fl)
            std_ = np.std(img_fl)
            n_images += img_fl.shape[0]
            weighted_mean += mean_ * n_images
            weighted_std += std_ * n_images

    mean_sample = weighted_mean / n_images
    std_sample = weighted_std / n_images
    return n_images, mean_sample, std_sample


def filter_metadata_by_grna(
    metadata_filename: str,
    output_filename: str,
    grna_ids_to_keep: list[int],
    include_time: bool
) -> pd.DataFrame:
    '''
    Filter the metadata DataFrame to include only the specified gRNA IDs.
    '''
    if not include_time:
        metadata = (pd
            .read_pickle(metadata_filename)
            .groupby('gene_grna_trench_index')
            .last()
            .reset_index()
        )
    else:
        metadata = (pd
            .read_pickle(metadata_filename)
        )

    metadata_filtered = (metadata
        .loc[metadata['oDEPool7_id'].isin(grna_ids_to_keep)]
        .reset_index(drop=True)
    )
    metadata_filtered.to_pickle(output_filename)
    return metadata_filtered



#%%



if not include_time:
    print('Get only last frame')
    metadata = (pd
        .read_pickle(METADATA_FILENAME)
        .groupby('gene_grna_trench_index')
        .last()
        .reset_index()
    )
else:
    metadata = (pd
        .read_pickle(METADATA_FILENAME)
    )

metadata_filtered = (metadata
    .loc[metadata['oDEPool7_id'].isin(GRNAS_IDS_TO_KEEP)]
    .reset_index(drop=True)
)
metadata_filtered.to_pickle(HEADPATH / '2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded_filtered_266-trenches.pkl')
# genes_to_separate
metadata_filtered

#%%
HEADPATH = Path('/mnt/efs/aimbl_2025/student_data/S-GL/')
METADATA_FILENAME = HEADPATH / '2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded.pkl'
OUTPUT_FILENAME = HEADPATH / '2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded_filtered_266-trenches.pkl'
include_time = True
GRNAS_IDS_TO_KEEP = [9577, 6197, 1809, 7591, 11017]
KEY_FL = 'fluorescence'
KEY_SEG = 'segmentation'
# filter_metadata_by_grna(
#     metadata_filename=METADATA_FILENAME,
#     output_filename=OUTPUT_FILENAME,
#     grna_ids_to_keep=GRNAS_IDS_TO_KEEP,
#     include_time=include_time
# )
#%%
metadata_filt = pd.read_pickle(OUTPUT_FILENAME)
metadata_filt
# Generate a lookup dictionary
for gene_id in metadata_lookup.keys():
    metadata_lookup[gene_id]['oDEPool_ids']
#%%
metadata = pd.read_pickle(METADATA_FILENAME).groupby('gene_grna_trench_index').last().reset_index()
#%%
gene = 'ftsN'
print(metadata[metadata['Gene']==gene]['oDEPool7_id'].unique())
#%%
print(metadata[metadata['Gene']==gene]['oDEPool7_id'].unique())
grna_id = 24209,
metadata_grna = metadata[metadata['oDEPool7_id'] == grna_id]
print(len(metadata_grna))
# Show the first N images as a horizontal stack
N = np.min([7,len(metadata_grna)])
H = 150
W = 21
image_stack = np.zeros((H, W*N))
for i in range(N):
    gene_id, grna_id, grna_file_trench_index = metadata_grna.iloc[i][['gene_id', 'oDEPool7_id', 'grna_file_trench_index']]
    with h5py.File(HEADPATH / 'Ecoli_lDE20_Exps-0-1' / f"{gene_id}/{grna_id}.hdf5", 'r') as f:
        img_fl = f[KEY_FL][grna_file_trench_index][-1]
        # img_seg = f[KEY_SEG][grna_file_trench_index][:]
    image_stack[:, i*W:(i+1)*W] = img_fl
plt.imshow(image_stack)
plt.xticks([]);plt.yticks([])

# Load an image given an oDEPool7_id
grna_id = 11017 #ftsN: 9577 Choose this, rplQ: 6197, alaS:1809 (not much of a ptype) # mreD: 7591 # hemC: 11017

grnas_chosen = {
    'ftsN': ['9577', '9586', '9588', '24197', '24199','24200', '24205', '24207', '24208', '24209'],
    'rplQ': ['6197', '6199', '6201', '6202', '6203'],
    'alaS': ['1807', '1809','1815','1816', '1817', '1819','18731',
             '18734','18735','18738','18744'],
    'mreD': ['7591', '7592','7593', '7594', '22799', 
             '22801', '22802', '22803', '22804', '22805'],
    'hemC': ['11016','11017','11018','11019','11021', '11022', 
             '11036', '11039','25196']
}

all_grnas_chosen = [int(item) for sublist in grnas_chosen.values() for item in sublist]
all_grnas_chosen
#%% Filter for the genes and gRNAs chosen
metadata_grnas_to_keep = (metadata
    .loc[metadata['oDEPool7_id'].isin(all_grnas_chosen)]
    .sort_values(['gene_id', 'oDEPool7_id'])
    .reset_index(drop=True)
    # Assign new gene
)
metadata_grnas_to_keep.to_pickle(HEADPATH / '2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded_filtered_lastT.pickle')
#%% Reproduce sampling
metadata = pd.read_pickle(HEADPATH / '2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded_filtered_lastT.pickle')
#%%

def find_image_in_hdf5_file(
    gene_id, grna_id, grna_file_trench_index, timepoint
):
    filename = HEADPATH / 'Ecoli_lDE20_Exps-0-1' / f"{gene_id}/{grna_id}.hdf5"
    with h5py.File(filename, 'r') as f:
        img_fl = f[KEY_FL][grna_file_trench_index, timepoint]
        # img_seg = f[KEY_SEG][grna_file_trench_index, timepoint]
    return img_fl#, img_seg

index = 2
metadata_sample = metadata.iloc[index]
gene_id, grna_id, grna_file_trench_index, timepoint = (metadata_sample
    [['gene_id', 'oDEPool7_id', 'grna_file_trench_index', 'timepoints']]
)
img_fl_anchor = find_image_in_hdf5_file(
    gene_id, grna_id, grna_file_trench_index, -1
)

# Get the potential positives
metadata_pos = (metadata
    .loc[lambda df_: (df_['gene_id'] == gene_id) 
         & (df_['grna_file_trench_index'] != grna_file_trench_index)]
    .sample(n=1)
    .iloc[0]
)
# Choose a random row
metadata_pos_sample = metadata_pos.sample(n=1).iloc[0]
gene_id, grna_id, grna_file_trench_index, timepoint = (metadata_pos_sample
    [['gene_id', 'oDEPool7_id', 'grna_file_trench_index', 'timepoints']]
)
img_fl_pos = find_image_in_hdf5_file(
    gene_id, grna_id, grna_file_trench_index, -1
)

plt.imshow(np.hstack([img_fl_anchor, img_fl_pos]))

    # img_seg_anchor = f[KEY_SEG][grna_file_trench_index, timepoint]
    # img_seg_pos = f[KEY_SEG][grna_file_trench_index, (timepoint+1)%T] # NO CORRECTION
# Using only the next timepoint, reset to the first timepoint if chose the last timepoint
# filename = self.data_path / f"{gene_id}/{grna_id}.hdf5"
#%% gRNAs to select
metadata_grnas_to_keep = metadata[metadata['oDEPool7_id'].isin(grna_ids_to_keep)]
# metadata_grnas_to_keep.tail(50)
#%%
# from pathlib import Path
# HEADPATH = Path('/mnt/efs/aimbl_2025/student_data/S-GL/')
# KEY_FL = 'fluorescence'
# KEY_SEG = 'segmentation' 
# metadata = pd.read_pickle(HEADPATH / '2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded.pkl')

# #%%
# n_images, mean_sample, std_sample = compute_mean_and_std_over_sample(
#     data_path=HEADPATH / 'Ecoli_lDE20_Exps-0-1/',
#     metadata=metadata
# )
# print(n_images, mean_sample, std_sample)
# # %%
# N_IMAGES = 15290
# WEIGHTED_MEAN = 200183.2210227729
# WEIGHTED_STD = 175738.43323474968

# SAMPLED_MEAN = WEIGHTED_MEAN/N_IMAGES
# SAMPLED_STD = WEIGHTED_STD/N_IMAGES

# print(SAMPLED_MEAN, SAMPLED_STD)