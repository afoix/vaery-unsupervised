"""
Convert nii.gz volumes (the output of ANTs registration) to ome.zarr. Requires ANTsPyX

Examples:
https://czbiohub-sf.github.io/iohub/main/auto_examples/run_multi_fov_hcs_ome_zarr.html#sphx-glr-auto-examples-run-multi-fov-hcs-ome-zarr-py

For Nikon or other supported file formats:
https://github.com/glencoesoftware/bioformats2raw?tab=readme-ov-file#usage
"""
#%%
from iohub import open_ome_zarr
from iohub.ngff.utils import create_empty_plate
from tifffile import tifffile
import numpy as np
from typing import Union, List
from pathlib import Path
import ants

def load_registered(chan_path: Union[Path, str]) -> np.ndarray:
    """
    load_registered
    Silly little loader for nii.gz
    BEWARE: Files have a very project specific directory structure and filename pattern

    Args:
        dset_id (str): Dataset identifier, e.g. '31A'
        chan (str): Channel identifier, e.g. 'Gbx2'
        registration_version: str, name of the registration model. 
    """
    #assert chan_path.exists()
    
    #chan_path = data_dir / f"{dset_id}_registration_v{registration_version}" / f'{id}_v{registration_version}_{chan}.nii.gz'
    if type(chan_path) is str:
        chan_path = Path(chan_path)
    if chan_path.is_file():
        file_name = chan_path.stem

        dset_id = str(file_name).split("_")[0]
        chan_id = str(file_name).split("_")[-1].split(".")[0]

        print(f"Loading {dset_id}_{chan_id}")
        return f"{dset_id}_{chan_id}", ants.image_read(str(chan_path)).numpy()
    else:
        print(f"File {chan_path} not found")
        return None


def collect_nii_files(directory_path: Union[Path, str]) -> List:
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
    if type(directory_path) is str:
        root_dir = Path(directory_path)
    else:
        root_dir = directory_path
    
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
            if filename.endswith('.nii.gz') and not filename.endswith('Warp.nii.gz') and not filename.endswith("nuc.nii.gz"):
                nii_files.append(file_path.absolute())
    
    return nii_files


#%%
output_zarr_store = Path('/mnt/data0/home/jnc2161/mbl/registered_salamander_data.zarr') # output zarr filename


_test_name, test_arr = load_registered(
    Path("/mnt/data1/shared/pleuro-brain-atlas/template_V3-1_d10_64bins_registered/21B_registration_template_V3-1_d10_64bins/21B_template_V3-1_d10_64bins_Pax6.nii.gz")
    )

# stuff for the zarr 
position_keys = [('0','0','0')] 
T,C,Z,Y,X = (1, 1, *test_arr.shape) #TODO: get the shape of the array to be converted. (T,C,Z,Y,X)

### path and label for template and mask
template_path = Path("/mnt/data1/shared/pleuro-brain-atlas/template_V3-1/template_V3-1.nii.gz")
mask_path = Path("/mnt/data1/shared/pleuro-brain-atlas/template_V3-1/template_V3-1_d3-mask.nii.gz")
assert template_path.is_file(), f"Template not found at {template_path}"
assert mask_path.is_file(), f"Mask not found at {mask_path}"

# where all the registered data lives
path_to_datasets = '/mnt/data1/shared/pleuro-brain-atlas/template_V3-1_d10_64bins_registered/'
input_files = sorted(collect_nii_files(path_to_datasets))

# datasets are appended to these lists (bypasses the for loop below)
input_files_filtered = [template_path, mask_path]
dset_ids = ["template", "template"]
channel_ids = ["V3-1", "d3-mask"]

exclude_dsets = ['21B', '21C', '22D']
exclude_chans = ['Cfos', 'NBion']

for this_file in input_files:  # doesn't include the template or mask
    this_dset = this_file.stem.split("_")[0]
    this_chan = this_file.stem.split("_")[-1].split(".")[0]

    if this_dset in exclude_dsets:
        print(f"Excluding {this_file}")
        continue
    elif this_chan in exclude_chans:
        print(f"Excluding {this_file}")
        continue
    else:
        dset_ids.append(this_dset)
        channel_ids.append(this_chan)
        input_files_filtered.append(this_file)

dset_labels = [f"{d}_{c}" for d, c in zip(dset_ids, channel_ids)]
input_files = input_files_filtered

print(f"{len(dset_labels)} files after exclusion")
print(f"Here are the dset labels: {dset_labels}")
print(f"Including the following files: {input_files}")

#%%
create_empty_plate(store_path=output_zarr_store,
    position_keys = position_keys,
    channel_names = dset_labels, #TODO: replace for the channel names
    shape = (1,len(dset_labels),Z,Y,X), #TODO: replace for the shape (T,C,Z,Y,X)
    chunks = (1, 1, 128, 128, 128), #TODO: this is important for training (T,C,Z,Y,X)
    scale = (1, 1, 7.51, 7.51, 7.51), #TODO: [Optional] replace for the scale in um. (T,C,Z,Y,X)
    dtype= np.float32, 
)


