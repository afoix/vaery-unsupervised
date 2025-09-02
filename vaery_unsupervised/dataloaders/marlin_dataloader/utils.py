#%%
import pandas as pd
import numpy as np
import h5py

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