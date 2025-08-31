"""
Convert nii.gz volumes (the output of ANTs registration) to ome.zarr. Requires ANTsPyX

Examples:
https://czbiohub-sf.github.io/iohub/main/auto_examples/run_multi_fov_hcs_ome_zarr.html#sphx-glr-auto-examples-run-multi-fov-hcs-ome-zarr-py

For Nikon or other supported file formats:
https://github.com/glencoesoftware/bioformats2raw?tab=readme-ov-file#usage
"""
from iohub import open_ome_zarr
from tifffile import tifffile
import numpy as np
from typing import Union
from pathlib import Path
import ants

def load_registered(dset_id: str, 
                    chan: str, 
                    registration_version: str, 
                    data_dir: Union[str, Path]) -> np.ndarray:
    """
    load_registered
    Silly little loader for nii.gz
    BEWARE: Files have a very project specific directory structure and filename pattern

    Args:
        dset_id (str): Dataset identifier, e.g. '31A'
        chan (str): Channel identifier, e.g. 'Gbx2'
        registration_version: str, name of the registration model. 
    """

    # Registered data from v3 onwards has a diff filename convention pattern
    pattern = f'{dset_id}*/*{registration_version}_{chan}.nii.gz'
    matches = list(data_dir.glob(pattern))
    assert len(matches) == 1
    chan_path = matches[0]
    #chan_path = data_dir / f"{dset_id}_registration_v{registration_version}" / f'{id}_v{registration_version}_{chan}.nii.gz'
    if chan_path.is_file():
        return ants.image_read(str(chan_path)).numpy()
    else:
        print(f"File {chan_path} not found")
        return None


def collect_nii_files(directory_path):
    """
    Collect all .nii.gz file paths from a directory and its subdirectories,
    excluding files that end with 'Warp.nii.gz'
    
    Args:
        directory_path (str): Path to the root directory
        
    Returns:
        list: List of full file paths matching the criteria
    """
    nii_files = []
    
    # Convert to Path object for easier handling
    root_dir = Path(directory_path)
    
    # Check if the directory exists
    if not root_dir.exists():
        print(f"Error: Directory '{directory_path}' does not exist.")
        return nii_files
    
    if not root_dir.is_dir():
        print(f"Error: '{directory_path}' is not a directory.")
        return nii_files
    
    # Walk through all directories and files
    for file_path in root_dir.rglob('*'):
        if file_path.is_file():
            filename = file_path.name
            # Check if file ends with .nii.gz but not Warp.nii.gz
            if filename.endswith('.nii.gz') and not filename.endswith('Warp.nii.gz'):
                nii_files.append(str(file_path.absolute()))
    
    return nii_files




output_zarr_store = '/output/dataset.zarr'
path_to_dataset = '/mnt/data1/shared/pleuro-brain-atlas/template_V3-1_d10_64bins_registered/'
tifffile.imread("test.czi")

# This creates a zarr store
with open_ome_zarr(output_zarr_store, mode='w') as store:
    # TODO: this encodes for /row/column/fov 
    row = 0
    column = 0
    fov = 0
    position = store.create_position(row,column,fov)

    # TODO: replace for the array. All the arrays should be the same shape
    position["0"] = np.random.randint(
        0, np.iinfo(np.uint16).max, size=(5, 3, 2, 32, 32), dtype=np.uint16
    )

#%%
from iohub.ngff.utils import create_empty_plate

output_zarr_store = '/output/dataset.zarr'

#TODO: position keys for the HCS dataset format.
position_keys = [(row,column,fov),...]

T,C,Z,Y,X = array.shape #TODO: get the shape of the array to be converted. (T,C,Z,Y,X)

#TODO: 
create_empty_plate(store_path=output_zarr_store,
    position_keys = position_keys,
    channel_names = ['channel_1'], #TODO: replace for the channel names
    shape = (T,C,Z,Y,X), #TODO: replace for the shape (T,C,Z,Y,X)
    chunks = None, #TODO: this is important for training (T,C,Z,Y,X)
    scale = (1, 1, 1, 1, 1), #TODO: [Optional] replace for the scale in um. (T,C,Z,Y,X)
    dtype= np.float32, 
)

# Parallel processing
from concurrent.futures import ProcessPoolExecutor

def convert_file(file_path:str, position_keys:tuple):
    # Logic to open a file and fill in the plate
    image = tifffile.imread(file_path)
    with open_ome_zarr(output_zarr_store+f'/{position_keys[0]}/{position_keys[1]}/{position_keys[2]}', mode='r+') as store:
        store["0"] = image

file_paths = ['file_path1', 'file_path2', 'file_path3']
with ProcessPoolExecutor(max_workers=10) as executor:
    # TODO: make the matching files with the position keys
    file_paths = [file_path for file_path in file_paths]
    position_keys = [position_key for position_key in position_keys]
    executor.map(convert_file, file_paths, position_keys)