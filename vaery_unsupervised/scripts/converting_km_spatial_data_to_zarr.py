#%%
import numpy as np

from lightning.pytorch import LightningDataModule
import time
from torch.utils.data import DataLoader, Dataset
import zarr
from pathlib import Path
from monai.transforms.compose import Compose
import json
from vaery_unsupervised.dataloaders.dataloader_km_ryans_template import (
    simple_masking, 
    DATASET_NORM_DICT, 
    load_sdata_files,
    extract_image_mask_poly_well,
    transpose_polygon_to_image
)
import torch
from vaery_unsupervised.km_utils import plot_batch_sample,plot_dataloader_output
import monai.transforms as transforms
out_path = Path("/mnt/efs/aimbl_2025/student_data/S-KM/")

class SpatConversionDS(Dataset):
    def __init__(
            self, 
            file_list:list[str],
            patient_id_list: list[str],
            polygon_sdata_name = 'affine_transformed',
            crop_size = 128,
    ):
        super().__init__()
        # adding dataset_specific information
        self.patient_id_list = patient_id_list
        self.image_name_list = [f"czi_{key}" for key in patient_id_list]
        self.coordinate_name_list = [f"aligned_{key}" for key in patient_id_list]
        self.polygon_sdata_name = polygon_sdata_name

        # getting datasets and mapping
        self.file_list = file_list
        sdata_objects, mapping, total_length = load_sdata_files(
            file_list,
            shapes_name=polygon_sdata_name
        )
        self.datasets = sdata_objects
        self.mapping = mapping
        self.total_length = total_length

        # defining functions for masking and normalisation
        # normalisation function expects (image, patient_id)
        self.crop_size = crop_size

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        # Logic to get a sample from the dataset
        dataset_idx, row_idx = self.mapping[index]

        image, mask, poly, well = extract_image_mask_poly_well(
            self.datasets[dataset_idx], 
            row_idx,
            name_transformed_poly = self.polygon_sdata_name,
            name_image=self.image_name_list[dataset_idx],
            name_coordinates=self.coordinate_name_list[dataset_idx],
            size = self.crop_size
        )

        patient_id_sample = self.patient_id_list[dataset_idx]
        metadata = {
            "dataset":patient_id_sample, 
            "dataset_file":self.file_list[dataset_idx],
            "row_idx":row_idx,
            "well_id":well,
        }

        return np.concat((image,mask[np.newaxis,:,:]),axis=0), metadata, poly
#%%

prefix = "/mnt/efs/aimbl_2025/student_data/S-KM/001_Used_Zarrs/onlybatch1/"
suffix = "_sdata_at"
name_list = ["450","453","456_2","457","493_2", "543_2"] 
file_list = [prefix + name + suffix for name in name_list]
file_list
# %%
dataset = SpatConversionDS(
    file_list=file_list,
    patient_id_list=name_list,
    polygon_sdata_name='affine_transformed',
)
# %%
dataset[0]
# %%
import pandas as pd
from tqdm. auto import tqdm
all_data = []
all_metadata = []
all_polys = []
for i in tqdm(range(len(dataset))):
    images, metadata, poly = dataset[i]
    all_data.append(images)
    all_metadata.append(metadata)
    all_polys.append(poly)

all_data = np.array(all_data)
print(all_data.shape)
all_metadata = pd.DataFrame(all_metadata)
#%%
all_metadata_attrs_dict = {
    k:list(all_metadata[k]) for k in all_metadata.columns
}
all_metadata_attrs_dict
#%%


zarr_array = zarr.array(all_data)
for k, v in all_metadata_attrs_dict.items():
    zarr_array.attrs[k] = v
#%%
zarr_array.attrs.keys()
# %%
crop_image_space_polys = [
    transpose_polygon_to_image(poly, 128)
    for poly in all_polys
]

test_poly = crop_image_space_polys[0]
crop_image_space_coordinates = []
for test_poly in crop_image_space_polys:
    xx, yy = test_poly.exterior.coords.xy
    crop_image_space_coordinates.append((list(xx),list(yy)))

aligned_space_polys = all_polys
aligned_space_coordinates = []
for test_poly in aligned_space_polys:
    xx, yy = test_poly.exterior.coords.xy
    aligned_space_coordinates.append((list(xx),list(yy)))
poly_dict = {
    "polys_crop_image_space_(x,y)": crop_image_space_coordinates,
    "polys_aligned_space_(x,y)":aligned_space_coordinates,
}
all_metadata_attrs_dict.update(poly_dict)
all_metadata_attrs_dict
# %%
group = zarr.open_group(
    out_path/"converted_crops_with_metadata.zarr",
)

crop_group = group.create_dataset(
    "crop_128_px",
    data = zarr_array,
    chunks=zarr_array.shape,
    dtype = zarr_array.dtype,
    overwrite=True,
)
#%%
with open(out_path/"converted_crops_with_metadata.zarr"/"metadata_crop_128_px.json",'w') as f:
    json.dump(all_metadata_attrs_dict, f)
