#%%
from pathlib import Path
# from iohub import open_ome_zarr
import h5py
import pandas as pd
import numpy as np

# HEADPATH = Path('/home/lag36/scratch/lag36/Ecoli/2023-01-18_lDE20_Merged_Analysis/')
HEADPATH = Path('/mnt/efs/aimbl_2025/student_data/S-GL/')
metadata_path = HEADPATH / '2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering.pkl'
output_zarr_store = HEADPATH / 'Ecoli_lDE20_Exps-0-1.zarr'
path_to_dataset = HEADPATH / 'Ecoli_lDE20_Exps-0-1'

def load_array_from_hdf5(
    file_idx: int,
    headpath: Path,
    prefix: str,
    key: str) -> np.ndarray:
        filepath = headpath / f"{prefix}{file_idx}.hdf5"
        with h5py.File(filepath, 'r') as f:
            images = f[key][:]
        return images

#%% 
metadata = pd.read_pickle(metadata_path)

position_keys = list(
    metadata.loc
        [:, ['gene_id', 'oDEPool7_id', 'grna_file_trench_index']
    ]
    .itertuples(index=False, name=None)
)
position_keys
#%%
gene_id, grna_id, grna_trench_id = position_keys[0]
sample_image = load_array_from_hdf5(
    file_idx=f'{grna_id,
    headpath=path_to_dataset,
    prefix=f'{gene_id}/{grna_id}',
    key='fluorescence'
)

# Get dimensions from a sample file 