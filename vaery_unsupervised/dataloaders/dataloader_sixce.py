import lightning as L
import os
from multiprocessing import Pool
import random
import numpy as np
from tifffile import imwrite
import sparse
import torch
from collections import defaultdict
import zarr
import numpy as np
from scipy import signal
import os
import pandas as pd
from shapely import wkb
from shapely.geometry import Polygon, MultiPolygon
import tifffile
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import zarr
from skimage.draw import polygon
import scanpy as sc
import anndata as ad
from scipy.ndimage import gaussian_filter
import sys
import torch
import math, statistics
from torch.utils.data import Dataset, DataLoader, random_split, Subset

import torch
import dask.array as da
from monai.transforms import (NormalizeIntensity, ScaleIntensityRange, Compose, RandGaussianSmooth, RandGaussianNoise,
                              OneOf, Rotate90, RandFlip)

from pytorch_lightning import LightningModule



class SixceDataModule(LightningModule):
    def __init__(self,
                 data_path: str,
                 dataset_export_path: str,
                 batch_size: int=10,
                 prefetch_factor: int=2,
                 pin_memory: bool=False,
                 persistent_workers: bool=False,
                 n_workers: int=10,
                 val_fraction: float=0.2,
                 zarr_attributes=None,
                 transcripts_df=None,
                 cell_metadata_df=None,
                 cell_boundary_df=None,
                 input_dim=200,
                 n_genes=140,
                 existing_dataset=None,
                 selected_stains=None,
                 n_masks=None,
                 padding=1,
                 apply_data_augmentation=False,
                 tensor_type=torch.float32,
                 include_stains=False,
                 representation_type='psf',
                 gaussian_patch_sigma=1.0,
                 gaussian_kernel_size=9,
                 psf_normalization='peak',
                 random_subset=-1,
                 peak_norm_peak_val=0.8,
                 circular_psf=False,
                 global_seed=137
                 ):
        super().__init__()
        self.data_path = data_path
        self.dataset_export_path = dataset_export_path
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.val_fraction = val_fraction

        # Store dataset params
        self.zarr_attributes = zarr_attributes
        self.transcripts_df = transcripts_df
        self.cell_metadata_df = cell_metadata_df
        self.cell_boundary_df = cell_boundary_df
        self.input_dim = input_dim
        self.n_genes = n_genes
        self.existing_dataset = existing_dataset
        self.selected_stains = selected_stains
        self.n_masks = n_masks
        self.padding = padding
        self.apply_data_augmentation = apply_data_augmentation
        self.tensor_type = tensor_type
        self.include_stains = include_stains
        self.representation_type = representation_type
        self.gaussian_patch_sigma = gaussian_patch_sigma
        self.gaussian_kernel_size = gaussian_kernel_size
        self.psf_normalization = psf_normalization
        self.random_subset = random_subset
        self.peak_norm_peak_val = peak_norm_peak_val
        self.circular_psf = circular_psf
        self.global_seed = global_seed

        # Initialize datasets
        self.full_dataset = existing_dataset
        self.train_dataset = None
        self.val_dataset = None


    def prepare_data(self):
        pass

    def _build_full_dataset(self):
        if self.existing_dataset is not None:
            return self.existing_dataset
        # Instantiate SpotStainDataset with your stored args
        _dataset = SpotStainDataset(
            ome_zarr_path=self.data_path,
            zarr_attributes=self.zarr_attributes,
            transcripts_df=self.transcripts_df,
            cell_metadata_df=self.cell_metadata_df,
            cell_boundary_df=self.cell_boundary_df,
            input_dim=self.input_dim,
            n_genes=self.n_genes,
            selected_stains=self.selected_stains,
            n_masks=self.n_masks,
            padding=self.padding,
            apply_data_augmentation=self.apply_data_augmentation,
            tensor_type=self.tensor_type,
            include_stains=self.include_stains,
            representation_type=self.representation_type,
            gaussian_patch_sigma=self.gaussian_patch_sigma,
            gaussian_kernel_size=self.gaussian_kernel_size,
            psf_normalization=self.psf_normalization,
            random_subset=self.random_subset,
            peak_norm_peak_val=self.peak_norm_peak_val,
            circular_psf=self.circular_psf,
            global_seed=self.global_seed
        )
        # Save the pt file
        print("Saving Dataset to disk...")
        torch.save(_dataset, self.dataset_export_path)

        return _dataset

    def setup(self, stage: str | None = None):
        # if no existing dataset is passed, build full dataset
        if self.full_dataset is None:
            print("Constructing Dataset...")
            self.full_dataset = self._build_full_dataset()

        # split into train and val
        n_total = len(self.full_dataset)
        n_val = max(1, int(self.val_fraction * n_total))
        n_train = n_total - n_val
        gen = torch.Generator().manual_seed(self.global_seed)
        self.train_dataset, self.val_dataset = random_split(self.full_dataset, [n_train, n_val], generator=gen)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.n_workers,
                          pin_memory=True, prefetch_factor=self.prefetch_factor,
                          persistent_workers=self.persistent_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.n_workers,
                          pin_memory=True, prefetch_factor=self.prefetch_factor,
                          persistent_workers=self.persistent_workers, drop_last=False)

    def predict_dataloader(self):
        return DataLoader(self.full_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.n_workers,
                          pin_memory=True,prefetch_factor=self.prefetch_factor,
                          persistent_workers=self.persistent_workers)

    def on_exception(self, exception):
        pass


class SpotStainDataset(Dataset):
    def __init__(self,
                 ome_zarr_path,
                 zarr_attributes,
                 transcripts_df,
                 cell_metadata_df,
                 cell_boundary_df,
                 input_dim,
                 n_genes,
                 selected_stains=None,
                 n_masks=None,
                 n_workers=8,
                 padding=1,
                 apply_data_augmentation=False,
                 tensor_type=torch.float32,
                 include_stains=False,
                 representation_type='psf',
                 gaussian_patch_sigma=1.0,
                 gaussian_kernel_size=11,
                 psf_normalization='sum_corrected',
                 random_subset=-1,
                 peak_norm_peak_val=0.8,
                 circular_psf=False,
                 global_seed=137
                 ):

        self.input_dim = input_dim
        self.n_genes = n_genes
        self.apply_data_augmentation = apply_data_augmentation
        self.tensor_type = tensor_type
        self.ome_zarr_path = ome_zarr_path
        self.lazy_loading = not include_stains
        self.channel_names = [c["label"] for c in zarr_attributes["omero"]["channels"]]
        self.chan_to_idx = {name: i for i, name in enumerate(self.channel_names)}
        self.representation_type = representation_type
        self.gaussian_patch_sigma = gaussian_patch_sigma
        self.gaussian_kernel_size = gaussian_kernel_size
        self.psf_normalization = psf_normalization
        self.random_subset = random_subset
        self.peak_norm_peak_val = peak_norm_peak_val
        self.circular_psf = circular_psf
        self.n_masks = n_masks
        self.global_seed = global_seed

        self.stain_transformations = Compose([
            RandGaussianSmooth(prob=0.5, sigma_x=(0.8, 1.2), sigma_y=(0.8, 1.2)),
            OneOf([RandGaussianNoise(prob=0.8, mean=0.0, std=0.05),
                  RandGaussianNoise(prob=0.8, mean=0.0, std=0.075),
                  RandGaussianNoise(prob=0.8, mean=0.0, std=0.1)])
        ])

        self.stain_spot_transformations = Compose([
            OneOf([Rotate90(k=1), Rotate90(k=2), Rotate90(k=3)]),
            RandFlip(prob=0.5, spatial_axis=0),
            RandFlip(prob=0.5, spatial_axis=1),
        ])

        # Compute the index of all cells in the dataset that have a transcript count >10
        cell_index_list = cell_metadata_df[cell_metadata_df["transcript_count"] > 10]["EntityID"].unique()

        # Random subset
        if random_subset != -1:
            np.random.seed(self.global_seed)
            cell_index_list = np.random.choice(cell_index_list, size=self.random_subset, replace=False) # mkw test subset


        # Split the list of cells into chunks for multiprocessing
        cell_index_split_list = np.array_split(cell_index_list, n_workers)

        # Filter transcripts_df not to only cell_indices, but to include any spots within the crop zone of the cells
        # Reason: when we create the 3D spot data object, filtering the transcript DF to the cells of interest would
        # result in some cells not getting spatial context (orphaned or other cell transcripts), but this will ensure
        # all crop zones have all the spots and not just those belonging to the cell that the crop is on
        def get_transcripts_within_crop_bounds(transcripts_df, cell_metadata_df, cell_indices, crop_size,
                                               zarr_attributes):
            """
            Filters transcripts_df to include only spots whose pixel-converted XY coords fall within the crop bounds
            of any cell of an EntityID that is in cell_indices.
            """
            # Extract transform
            transformation_list = zarr_attributes["multiscales"][0]["datasets"][0]["coordinateTransformations"]
            scale_vec = next(t["scale"] for t in transformation_list if t["type"] == "scale")
            translation_vec = next(t["translation"] for t in transformation_list if t["type"] == "translation")
            _, um_per_px_y, um_per_px_x = scale_vec
            _, trans_um_y, trans_um_x = translation_vec

            # Convert spot coords
            transcripts_df = transcripts_df.copy()
            transcripts_df["global_x_px"] = ((transcripts_df["global_x"] - trans_um_x) / um_per_px_x).round().astype(
                int)
            transcripts_df["global_y_px"] = ((transcripts_df["global_y"] - trans_um_y) / um_per_px_y).round().astype(
                int)

            # Get pixel-space crop bounds for each cell
            subset = cell_metadata_df[cell_metadata_df["EntityID"].isin(cell_indices)]
            x_centers_px = ((subset["center_x"] - trans_um_x) / um_per_px_x).round().astype(int)
            y_centers_px = ((subset["center_y"] - trans_um_y) / um_per_px_y).round().astype(int)

            x_min = x_centers_px.min() - crop_size // 2
            x_max = x_centers_px.max() + crop_size // 2
            y_min = y_centers_px.min() - crop_size // 2
            y_max = y_centers_px.max() + crop_size // 2

            # Filter
            mask = (
                    (transcripts_df["global_x_px"] >= x_min) &
                    (transcripts_df["global_x_px"] <= x_max) &
                    (transcripts_df["global_y_px"] >= y_min) &
                    (transcripts_df["global_y_px"] <= y_max)
            )
            return transcripts_df[mask]

        # Selecting stains to use
        if selected_stains is None:
            selected_stains = [ch["label"] for ch in zarr_attributes["omero"]["channels"]]

        self.selected_stains = selected_stains

        # open once for building tensors and computing stats (skipping for now), then discard
        zarr_array = zarr.open(f"{self.ome_zarr_path}/0/0", mode="r")
        # self.channel_intensity_stats_dict = compute_global_intensity_stats(self.ome_zarr_path, self.channel_names)
        # TODO remove this speed hack
        # self.channel_intensity_stats_dict = {'Cellbound1': {'mean': 623.166358530191, 'std': 240.14081567000878},
        #  'Cellbound2': {'mean': 2855.887921505783, 'std': 1468.8268038004205},
        #  'Cellbound3': {'mean': 2649.0847150285545, 'std': 3170.3506509640947},
        #  'DAPI': {'mean': 5614.163506785238, 'std': 4989.878456710014},
        #  'Ki67': {'mean': 1749.7557510940997, 'std': 1006.7450722766594},
        #  'PolyT': {'mean': 7490.723521784297, 'std': 4988.989565408447},
        #  'WT1': {'mean': 570.6027619398159, 'std': 298.5829817267415}}

        # Packing args to avoid positional dependence
        args = [{
            "cell_indices": cell_index_sublist,
            "detected_transcripts_subset_df": get_transcripts_within_crop_bounds(
                transcripts_df,
                cell_metadata_df,
                cell_index_sublist,
                input_dim,
                zarr_attributes),
            "cell_metadata_subset_df": cell_metadata_df[cell_metadata_df["EntityID"].isin(cell_index_sublist)],
            "cell_boundary_subset_df": cell_boundary_df[cell_boundary_df["EntityID"].isin(cell_index_sublist)],
            "n_genes": n_genes,
            "input_dim": input_dim,
            "selected_stains": selected_stains,
            "n_masks": n_masks,
            "padding": padding,
            "zarr_array": zarr_array,
            "zarr_attributes": zarr_attributes,
            "include_stains": include_stains,
            "representation_type": representation_type,
            "gaussian_patch_sigma": gaussian_patch_sigma,
            "gaussian_kernel_size": gaussian_kernel_size,
            "psf_normalization": psf_normalization,
            "peak_norm_peak_val": peak_norm_peak_val,
            "circular_psf": circular_psf
        } for cell_index_sublist in cell_index_split_list]

        with Pool(processes=n_workers) as pool:
            results = pool.map(unpack_and_run_build_cell_tensor_components, args)

        del zarr_array  # delete the ome-zarr array so the live handle is not stored in self

        # Unpacking results
        cell_indices = []
        cell_transcripts = []
        stain_crop_lists_dict = defaultdict(list) # Generalizing stains instead of hardcoding
        cell_crop_coords = [] # only filled in lazy loading mode (include_stains=False)
        for out in results:
            if include_stains:
                processed_cell_indices, transcripts, stain_dict = out
            else:
                processed_cell_indices, transcripts, stain_dict, crop_coords = out
                cell_crop_coords.extend(crop_coords)

            cell_indices.extend(processed_cell_indices)
            cell_transcripts.extend(transcripts)
            for stain, crops in stain_dict.items():
                stain_crop_lists_dict[stain].extend(crops)
        self.cell_transcripts = cell_transcripts
        self.cell_indices = cell_indices
        self.stain_crop_lists_dict = dict(stain_crop_lists_dict)
        self.stain_order = [stain for stain in zarr_attributes['omero']['channels']]
        self.stain_order = [ch['label'] for ch in self.stain_order if ch['label'] in self.stain_crop_lists_dict]
        if self.n_masks != 0 and "mask" in self.stain_crop_lists_dict and "mask" not in self.stain_order:
            self.stain_order.append("mask")
        self.cell_crop_coords = cell_crop_coords  # can be empty (if include_stains=True)

    def __getitem__(self, index):
        # lazy_loading => when include_stains is False
        if self.lazy_loading and not hasattr(self, "_zarr"):
            # Each worker opens its own handle the first time it needs it
            self._zarr = zarr.open(f"{self.ome_zarr_path}/0/0", mode="r")

        sparse_input_array = self.cell_transcripts[index]
        input_array = sparse_input_array.todense()

        cell_index = self.cell_indices[index]
        transcripts_tensor = torch.tensor(input_array.copy(), dtype=self.tensor_type)

        # Stack all stain crops in a fixed order
        stain_tensors = []
        for stain in self.stain_order:
            crop = self.stain_crop_lists_dict[stain][index]
            # For standarizing across whole WSI in that channel. SKipping for now
            # ch_intensity_mean = self.channel_intensity_stats_dict[stain]['mean']
            # ch_intensity_std = self.channel_intensity_stats_dict[stain]['std']

            # If we're lazy loading, the crop dict keys contain None as value, except the mask.
            # Fetch the actual crops from the OME-Zarr
            if self.lazy_loading and crop is None and stain != "mask":
                # derive crop coords for this cell
                x_min, y_min = self.cell_crop_coords[index]
                y_max = y_min + self.input_dim
                x_max = x_min + self.input_dim

                chan = self.chan_to_idx[stain]
                # Fetch the crop
                crop = self._zarr[chan, y_min:y_max, x_min:x_max]

                #START ###################################### Normalization ############################################
                # Standardize based on entire WSI for that channel. Skipping for now
                # stain_norm = NormalizeIntensity(
                #     subtrahend=ch_intensity_mean,
                #     divisor=ch_intensity_std,
                #     nonzero=True,
                #     dtype=crop.dtype
                # )
                ############################################
                # Min-max per crop, skipping for now
                # rng = crop.max() - crop.min()
                # crop = (crop - crop.min()) / (rng + 1e-6)
                ############################################
                # Min-max per region
                region_size = 4000

                # center of crop
                cx = x_min + (self.input_dim // 2)
                cy = y_min + (self.input_dim // 2)

                # initial region bounds
                region_x_min = max(0, cx - region_size // 2)
                region_y_min = max(0, cy - region_size // 2)
                region_x_max = min(self._zarr.shape[2], region_x_min + region_size)
                region_y_max = min(self._zarr.shape[1], region_y_min + region_size)

                # adjust if truncated at edge
                if region_x_max - region_x_min < region_size:
                    region_x_min = max(0, region_x_max - region_size)
                if region_y_max - region_y_min < region_size:
                    region_y_min = max(0, region_y_max - region_size)

                region = self._zarr[chan, region_y_min:region_y_max, region_x_min:region_x_max]

                # compute min/max
                r = region[region != 0] if region.any() else region
                if r.size == 0:
                    rmin, rmax = 0.0, 1.0
                else:
                    rmin, rmax = float(r.min()), float(r.max())
                    if rmin == rmax:
                        rmax = rmin + 1.0

                # normalize crop to [0,1]
                crop = (crop.astype(np.float32) - rmin) / (rmax - rmin)
                crop = np.clip(crop, 0.0, 1.0)

                # END ####################################### Normalization ############################################

                # cache so future epochs don’t re-read (commented out to avoid RAM blow-up)
                # self.stain_crop_lists_dict[stain][index] = crop

            # mask is already present and heavy mode always has crops, so..
            stain_tensors.append(
                torch.tensor(crop, dtype=self.tensor_type).unsqueeze(0)
            )
        stain_tensor = torch.cat(stain_tensors, dim=0)

        # Concat cloned stain and spot tensors
        model_input_tensor = torch.cat((stain_tensor.clone(), transcripts_tensor.clone()), dim=0)

        # Apply augmentation on stain tensor and then on concatenated tensor
        model_input_tensor_augmented = self.stain_spot_transformations(
            torch.cat((self.stain_transformations(stain_tensor), transcripts_tensor), dim=0)).clamp(0, 1)

        sample = {
            "anchor": model_input_tensor,
            "positive": model_input_tensor_augmented
        }

        return sample

    def __len__(self):
        return len(self.cell_indices)

    def output_dataset(self, output_dir):
        """
        Write the dataset to disk
        Create a folder for each cell containing images of: dapi, membrane and spots
        """
        print(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # If lazy loading, load the zarr array again
        if self.lazy_loading and not hasattr(self, "_zarr"):
            self._zarr = zarr.open(f"{self.ome_zarr_path}/0/0", mode="r")

        for idx, cell_index in enumerate(self.cell_indices):
            cell_dir = os.path.join(output_dir, str(cell_index))
            if not os.path.exists(cell_dir):
                os.makedirs(cell_dir)

            stain_crop_dict = {}
            for stain in self.stain_order:
                crop = self.stain_crop_lists_dict[stain][idx]

                # If this is a lazy-loaded dataset and the crop isn't loaded yet
                if self.lazy_loading and crop is None and stain != "mask":
                    x_min, y_min = self.cell_crop_coords[idx]
                    y_max, x_max = y_min + self.input_dim, x_min + self.input_dim
                    chan = self.chan_to_idx[stain]
                    crop = self._zarr[chan, y_min:y_max, x_min:x_max]
                    rng = crop.max() - crop.min()
                    crop = (crop - crop.min()) / (rng + 1e-6)

                    # cache so subsequent calls don’t re-read
                    self.stain_crop_lists_dict[stain][idx] = crop

                stain_crop_dict[stain] = crop

            # Save overlay image
            sparse_input_array = self.cell_transcripts[idx]
            cell_transcripts = sparse_input_array.todense()
            projected_cell_transcripts = np.argmax(cell_transcripts, axis=0)
            cell_image_array = np.mean(list(stain_crop_dict.values()), axis=0) * 255
            plt.figure()
            plt.imshow(cell_image_array, cmap='gray')
            plt.imshow(projected_cell_transcripts, alpha=0.2)
            plt.savefig(os.path.join(cell_dir, 'cell_overlay.png'))
            plt.close()

            for stain, crop in stain_crop_dict.items():
                imwrite(os.path.join(cell_dir, f"{stain.lower()}.tif"), crop)


def build_cell_tensor_components(
        cell_indices,
        detected_transcripts_subset_df,
        cell_metadata_subset_df,
        cell_boundary_subset_df,
        n_genes,
        input_dim,
        selected_stains,
        n_masks,
        padding,
        zarr_array,
        zarr_attributes,
        include_stains,
        representation_type,
        gaussian_patch_sigma,
        gaussian_kernel_size,
        psf_normalization,
        peak_norm_peak_val,
        circular_psf):

    # Initializing lists and dicts
    cell_transcripts = []
    processed_cell_indices = []
    stain_crop_lists_dict = defaultdict(list)
    cell_crop_coords = [] if not include_stains else None

    # Iterating over cells to collect tensor components
    for cell_idx in cell_indices:
        # Crop each stain
        stain_crop_dict, x_min, y_min = get_cell_crop(
            cell_index=cell_idx,
            cell_metadata_subset_df=cell_metadata_subset_df,
            cell_boundary_subset_df=cell_boundary_subset_df,
            zarr_array=zarr_array,
            zarr_attributes=zarr_attributes,
            selected_stains=selected_stains,
            crop_size=input_dim,
            return_stains=include_stains,
            n_masks=n_masks
        )

        # Append the crop coords if lazy loading
        if cell_crop_coords is not None:
            cell_crop_coords.append((x_min, y_min))

        # Create gene expression multichannel representation (3D binary spot arrays)
        try:
            input_array = (
                create_multichannel_representation(cell_transcripts_df=detected_transcripts_subset_df,
                                                   zarr_attributes=zarr_attributes,
                                                   x_min=x_min,
                                                   y_min=y_min,
                                                   n_genes=n_genes,
                                                   padding=padding,
                                                   crop_size=input_dim,
                                                   representation_type=representation_type,
                                                   gaussian_sigma=gaussian_patch_sigma,
                                                   gaussian_kernel_size=gaussian_kernel_size,
                                                   psf_normalization=psf_normalization,
                                                   peak_norm_peak_val=peak_norm_peak_val,
                                                   circular=circular_psf
                                                   ))

            sparse_input_array = sparse.COO(input_array)
        except Exception as e:
            print("Error processing cell", cell_idx)
            continue

        # Collect common outputs
        for stain, crop in stain_crop_dict.items():
            stain_crop_lists_dict[stain].append(crop)
        cell_transcripts.append(sparse_input_array)
        processed_cell_indices.append(cell_idx)

    if include_stains:
        return (processed_cell_indices,
                cell_transcripts,
                stain_crop_lists_dict) # heavy data
    else:
        return (processed_cell_indices,
                cell_transcripts,
                stain_crop_lists_dict,
                cell_crop_coords)

# Utils
def unpack_and_run_build_cell_tensor_components(d):
    return build_cell_tensor_components(**d)


def get_all_exterior_coords(geom):
    '''
    Returns list of all exterior coordinates of single or multiple polygons
    '''
    if isinstance(geom, Polygon):
        return [list(geom.exterior.coords)]
    elif isinstance(geom, MultiPolygon):
        return [list(p.exterior.coords) for p in geom.geoms]
    else:
        return []

def micron_coords_to_px(coords, scale_vector, translational_vector):
    '''
    Converts coords from micron space to pixel space
    '''
    _, um_per_px_y, um_per_px_x = scale_vector
    _, trans_um_y, trans_um_x = translational_vector

    return [(int(round((x - trans_um_x) / um_per_px_x)),
             int(round((y - trans_um_y) / um_per_px_y)))
            for (x, y) in coords]


def create_gaussian_patch(sigma, kernel_size, normalization, peak_val=0.8, circular=True):
    '''
    Returns a gaussian patch with a specified sigma, kernel size, and pixel value normalization
    peak norm: normalize so the peak of a given patch with no neighbors = 1
    sum norm: normalize so the sum of a given patch with no neighbors = 1
    sum_corrected: sum norm and then scale such that even if a patch has 4 immediate and 4 diagonal neighbors,
        max pixel intensity remains < 1. However, if a patch has additional non-immediate neighbors where their
        >2nd pixel shell overlaps with the peak, it may surpass 1 but that is extremely unlikely. The scaling factor
        for this approach is hardcoded and experimentally derived for sigma=1 and only immediate/diagonal neighbors.
    circular: if True, truncate support to a disk of radius floor(kernel_size/2) (corners set to 0)
    '''
    ax = np.arange(kernel_size) - kernel_size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    if circular:
        r = kernel_size // 2
        d2 = xx**2 + yy**2
        kernel = np.where(d2 <= r*r, kernel, 0.0)

    if normalization == 'peak':
        kernel /= np.max(kernel)  # normalize so gaussian peak = 1
        kernel *= peak_val  # re-normalize so peak = specific value
    elif normalization == 'sum':
        kernel /= np.sum(kernel)  # normalize so sum of patch pixel values = 1
    elif normalization == 'sum_corrected':
        kernel /= np.sum(kernel)
        # scaling factor derived experimentally for sigma = 1 with 8 immediate/diagonal neighbors, on square kernels
        kernel *= 1.2829
    else:
        print("Unknown gaussian patch normalization arg. Aborting.")
        sys.exit()
    return kernel

# Function to apply patch at a single (y, x) location in a 3D array
def apply_gaussian_patch(array, gene_idx, y, x, patch, crop_size):
    '''
    Applies a gaussian patch for a given gene spot
    '''
    h, w = patch.shape
    half_h, half_w = h // 2, w // 2
    y_start = max(y - half_h, 0)
    y_end = min(y + half_h + 1, crop_size)
    x_start = max(x - half_w, 0)
    x_end = min(x + half_w + 1, crop_size)

    gy_start = half_h - (y - y_start)
    gy_end = gy_start + (y_end - y_start)
    gx_start = half_w - (x - x_start)
    gx_end = gx_start + (x_end - x_start)

    array[gene_idx, y_start:y_end, x_start:x_end] += patch[gy_start:gy_end, gx_start:gx_end]


def psf_total_intensity(kernel_size: int,
                        sigma: float,
                        normalization: str = "peak",
                        peak_val: float = 0.8,
                        circular: bool = True) -> float:
    """
    Compute the total intensity (sum of pixel values) of a Gaussian PSF patch,
    using the same normalization conventions as create_gaussian_patch().

    Args:
        kernel_size (int): size of the patch (odd preferred).
        sigma (float): Gaussian sigma in pixels.
        normalization (str): one of "peak", "sum", "sum_corrected".
        peak_val (float): only used for 'peak' normalization (scales the maximum pixel).
        circular (bool): if True, truncate support to a disk of radius floor(kernel_size/2).

    Returns:
        float: total intensity (sum of kernel values after normalization).
    """
    ax = np.arange(kernel_size) - kernel_size // 2
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    if circular:
        r = kernel_size // 2
        d2 = xx**2 + yy**2
        kernel = np.where(d2 <= r * r, kernel, 0.0)

    if normalization == "peak":
        kernel = kernel / np.max(kernel) * peak_val
    elif normalization == "sum":
        kernel = kernel / np.sum(kernel)
    elif normalization == "sum_corrected":
        kernel = kernel / np.sum(kernel)
        # experimentally derived scaling for sigma=1, 8-neighbor overlap (measured with square kernels)
        kernel = kernel * 1.2829
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    return float(kernel.sum())


def get_cell_crop(cell_index, cell_metadata_subset_df, cell_boundary_subset_df, zarr_array, zarr_attributes,
                  selected_stains, crop_size, return_stains=False, n_masks=1):
    # Isolating to current cell
    current_cell_metadata = cell_metadata_subset_df[cell_metadata_subset_df["EntityID"] == cell_index].iloc[0]
    current_cell_boundary_row = cell_boundary_subset_df[cell_boundary_subset_df["EntityID"] == cell_index].iloc[0]

    # Fetch OME-Zarr shape and scaling metadata
    _, img_h, img_w = zarr_array.shape
    transformation_list = zarr_attributes["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    scale_vec = next(t["scale"] for t in transformation_list if t["type"] == "scale")
    translation_vec = next(t["translation"] for t in transformation_list if t["type"] == "translation")
    _, um_per_px_y, um_per_px_x = scale_vec
    _, trans_um_y, trans_um_x = translation_vec

    # Get cell boundaries in WKB geometry, decode, and convert to pixels
    wkb_geom = current_cell_boundary_row["Geometry"]
    geom = wkb.loads(wkb_geom)
    boundary_coords = get_all_exterior_coords(geom)
    boundary_coords_px = [
        micron_coords_to_px(poly, scale_vec, translation_vec)
        for poly in boundary_coords
    ]

    # Computing coords for static cropping around centroids
    # Get raw coords in microns
    x_center, y_center = current_cell_metadata['center_x'], current_cell_metadata['center_y']
    # Convert to pixels
    x_center_px = int(round(((x_center - trans_um_x) / um_per_px_x)))
    y_center_px = int(round(((y_center - trans_um_y) / um_per_px_y)))
    # Computing padding and crop coords
    padding = int(crop_size / 2)
    x_min = max(0, min(x_center_px - padding, img_w - crop_size))
    y_min = max(0, min(y_center_px - padding, img_h - crop_size))
    x_max, y_max = x_min + crop_size, y_min + crop_size

    crop_dict = {}
    # Build channel-name and index lookup from attrs
    channel_names = [c["label"] for c in zarr_attributes["omero"]["channels"]]
    chan_to_idx   = {name: i for i, name in enumerate(channel_names)}

    # Cropping each stain (channel)
    for name in selected_stains:
        idx = chan_to_idx[name]
        if return_stains:
            crop = zarr_array[idx, y_min:y_max, x_min:x_max]
            # Normalize
            rng = crop.max() - crop.min()
            crop_dict[name] = (crop - crop.min()) / (rng + 1e-6)
        else:
            crop_dict[name] = None

    # Creating cell mask
    if n_masks != 0:
        mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
        # mask = np.zeros((crop_size, crop_size), dtype=np.float32) # if mask true pixels are <1
        for poly in boundary_coords_px: # as we often have multi-polygons
            # shift to crop coordinates
            xs = [x - x_min for x, y in poly]
            ys = [y - y_min for x, y in poly]
            rr, cc = polygon(ys, xs, mask.shape)
            mask[rr, cc] = 1

        # Adding mask to crop dict
        crop_dict["mask"] = mask

    return crop_dict, x_min, y_min


def create_multichannel_representation(cell_transcripts_df, zarr_attributes, x_min, y_min, n_genes, padding,
                                       crop_size, representation_type='psf', gaussian_sigma=1,
                                       gaussian_kernel_size=11, psf_normalization='sum_corrected',
                                       peak_norm_peak_val=0.8, circular=True):
    # Fetch OME-Zarr scaling metadata for micron to pixel conversion
    transformation_list = zarr_attributes["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    scale_vec = next(t["scale"] for t in transformation_list if t["type"] == "scale")
    translation_vec = next(t["translation"] for t in transformation_list if t["type"] == "translation")
    _, um_per_px_y, um_per_px_x = scale_vec
    _, trans_um_y, trans_um_x = translation_vec

    # Convert spot coords from microns to pixels
    cell_transcripts_df["global_x_px"] = ((cell_transcripts_df["global_x"] - trans_um_x) / um_per_px_x).round().astype(
        int)
    cell_transcripts_df["global_y_px"] = ((cell_transcripts_df["global_y"] - trans_um_y) / um_per_px_y).round().astype(
        int)

    # Remove transcripts outside the crop
    cell_transcripts_df = cell_transcripts_df[(cell_transcripts_df['global_x_px']
                                               > x_min) & (cell_transcripts_df['global_y_px']
                                                           > y_min)]
    cell_transcripts_df = cell_transcripts_df[(cell_transcripts_df['global_x_px']
                                               < (x_min + crop_size)) & (cell_transcripts_df['global_y_px']
                                                                         < (y_min + crop_size))]

    x_coordinates = cell_transcripts_df['global_x_px'].astype(int) - x_min
    y_coordinates = cell_transcripts_df['global_y_px'].astype(int) - y_min
    barcode_ids = cell_transcripts_df['barcode_id']

    # Binary padded spot representation
    if representation_type == "binary":
    # Construct a binary 3D representation of the transcripts
        binary_transcripts = np.zeros((n_genes, crop_size, crop_size), dtype=np.uint8)
        binary_transcripts[barcode_ids, y_coordinates, x_coordinates] = 1

        # Define the kernel
        kernel = np.ones((padding+2, padding+2))
        # Run 2D convolution on each gene plane
        transcript_array = np.array([signal.convolve2d(gene_transcripts, kernel, boundary='symm', mode='same') for gene_transcripts in binary_transcripts])
        # Binarize
        transcript_array = (transcript_array != 0).astype(np.uint8)
    # Grayscale gaussian point spread function representation
    elif representation_type == "psf":
        # Initialize grayscale transcript array
        transcript_array = np.zeros((n_genes, crop_size, crop_size), dtype=np.float32)
        gaussian_patch = create_gaussian_patch(sigma=gaussian_sigma, kernel_size=gaussian_kernel_size,
                                               normalization=psf_normalization, peak_val=peak_norm_peak_val,
                                               circular=circular)

        for gene_idx, y, x in zip(barcode_ids, y_coordinates, x_coordinates):
            apply_gaussian_patch(array=transcript_array,
                                 gene_idx=gene_idx,
                                 y=y,
                                 x=x,
                                 patch=gaussian_patch,
                                 crop_size=crop_size)
        # Clip to 1 to avoid issues with loss functions. More of a paranoid guard when using corrected_sum since it's
        # incredibly unlikely that a spot has 8 immediate neighbors and then additional enough non-immediate neighbors
        # for its peak to surpass 1. When using 'peak' normalization, this will distort the relationship between
        # total image intensity and its spot count. When using 'sum' normalization, it's difficult to compute but most
        # likely >1 is impossible. This is all assuming sigma=1.
        transcript_array[transcript_array > 1.0] = 1.0 # MKW disable clipping
    else:
        raise ValueError(f"Unknown gene representation type: {representation_type}. Aborting.")
        sys.exit()

    return transcript_array


def compute_global_intensity_stats(ome_zarr_path, channel_names):
    """
    Compute per-channel mean/std ignoring zeros.

    Parameters
    ----------
    ome_zarr_path : str
        Path to OME-Zarr root (directory containing "0/0").
    channel_names : list[str]
        List of channel names (already parsed from zarr_attributes["omero"]["channels"]).
    """
    # open chunks as Dask array
    zarr_chunks = da.from_zarr(f"{ome_zarr_path}/0/0").astype(np.float64)  # (C,Y,X)

    # mask out zeros
    mask = zarr_chunks != 0

    # counts, sums, sums of squares
    count = mask.sum(axis=(1, 2))
    sum_  = (zarr_chunks * mask).sum(axis=(1, 2))
    sumsq = ((zarr_chunks ** 2) * mask).sum(axis=(1, 2))

    # compute in one go
    sum_, sumsq, count = da.compute(sum_, sumsq, count)

    # stats
    mean = sum_ / count
    var  = (sumsq / count) - mean**2
    std  = np.sqrt(np.maximum(var, 0))

    # dict keyed by provided channel names
    channel_stats = {
        name: {"mean": float(m), "std": float(s)}
        for name, m, s in zip(channel_names, mean, std)
    }

    return channel_stats


def get_clipping_stats(ds):
    """
    Diagnostic fxn for when using the PSF spot representation that shows stats for pixels that exceed
    1. Obviously, need to turn off clipping first (manually, in create_multichannel_representation) if
    calling from main

    Returns:
      overall_pos_clip_frac:   (# of values >1) / (# of non-zero values) across the dataset
      per_ch_pos_clip_frac:    array[C], per-channel fraction >1 among that channel's non-zeros
      overflow_frac_pos_mass:  sum(x-1 for x>1) / sum(x for x>0)  (how much mass sits above 1, among positives)
    """
    C = ds.n_genes
    total_nnz = 0
    total_clipped = 0
    pos_mass = 0.0
    overflow_mass = 0.0

    nnz_per_ch = np.zeros(C, dtype=np.int64)
    clipped_per_ch = np.zeros(C, dtype=np.int64)

    clipped_values_all = []
    clipped_values_ch = [[] for _ in range(C)]

    for arr in ds.cell_transcripts:          # arr: sparse.COO with coords (3, nnz) -> (c, y, x)
        data = arr.data                      # (nnz,)
        ch   = arr.coords[0]                 # (nnz,) channel indices

        # counts
        nnz = data.size
        total_nnz += nnz
        nnz_per_ch += np.bincount(ch, minlength=C)

        # clipped among non-zeros
        m = data > 1.0
        if m.any():
            k = int(m.sum())
            total_clipped += k
            clipped_per_ch += np.bincount(ch[m], minlength=C)

            overflow_mass += np.maximum(data[m] - 1.0, 0.0).sum()

            clipped_values_all.extend(data[m])
            for ci in np.unique(ch[m]):
                clipped_values_ch[ci].extend(data[(m) & (ch == ci)])

        # positive mass (all non-zeros)
        pos_mass += float(data.sum())

    overall_pos_clip_frac = (total_clipped / total_nnz) if total_nnz > 0 else 0.0

    # per-channel fraction
    per_ch_pos_clip_frac = np.divide(
        clipped_per_ch, np.maximum(nnz_per_ch, 1), where=(nnz_per_ch > 0)
    )

    overflow_frac_pos_mass = overflow_mass / (pos_mass + 1e-8)

    # if clipped_values_all:
    #     print(f"[Overall >1] avg={np.mean(clipped_values_all):.4f}, "
    #           f"median={np.median(clipped_values_all):.4f}")
    # else:
    #     print("[Overall >1] No clipped values found.")

    for c in range(C):
        if clipped_values_ch[c]:
            print(f"[Channel {c} >1] avg={np.mean(clipped_values_ch[c]):.4f}, "
                  f"median={np.median(clipped_values_ch[c]):.4f}")

    return overall_pos_clip_frac, per_ch_pos_clip_frac, overflow_frac_pos_mass, clipped_values_all


def compute_sparsity_stats(dataset, n_stains, print_stats=True):
    """
    Compute mean, median, and geometric mean of non-zero pixel fraction, globally.

    Args:
        dataset: torch Dataset or iterable of (image, …).
        n_stains: number of stain channels to skip.
        print_stats: whether to print the results.

    Returns:
        dict with keys: 'mean', 'median', 'geom_mean'
    """
    count_pos, count_tot = 0, 0
    per_sample_fracs = []

    for sample in dataset:
        img = sample[0]
        gene_stack = img[n_stains:]      # (C,H,W)
        pos = (gene_stack > 0).sum().item()
        tot = gene_stack.numel()

        count_pos += pos
        count_tot += tot

        per_sample_fracs.append(0.0 if tot == 0 else pos / tot)

    mean_rate = count_pos / count_tot if count_tot > 0 else float('nan')
    median_rate = statistics.median(per_sample_fracs) if per_sample_fracs else float('nan')

    if per_sample_fracs and any(f == 0.0 for f in per_sample_fracs):
        geom_mean_rate = 0.0
    else:
        geom_mean_rate = (
            math.exp(sum(math.log(f) for f in per_sample_fracs) / len(per_sample_fracs))
            if per_sample_fracs else float('nan')
        )

    if print_stats:
        print(f"Dataset global non-zero px fraction (mean)            = {mean_rate:.6g}")
        print(f"Dataset global non-zero px fraction (median)          = {median_rate:.6g}")
        print(f"Dataset global non-zero px fraction (geometric mean)  = {geom_mean_rate:.6g}")

    return {
        "mean": mean_rate,
        "median": median_rate,
        "geom_mean": geom_mean_rate,
    }


def inspect_pt_file(obj, max_items=10, max_repr=5):
    def summarize(v):
        if isinstance(v, torch.Tensor):
            try:
                vmin = v.min().item();
                vmax = v.max().item()
                return f"Tensor(shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}, min={vmin:.4g}, max={vmax:.4g})"
            except Exception:
                return f"Tensor(shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device})"
        if isinstance(v, dict):
            keys = list(v.keys())
            head = keys[:max_items]
            extra = f"... +{len(keys) - max_items} more" if len(keys) > max_items else ""
            return f"dict({len(keys)} keys) preview={head}{extra}"
        if isinstance(v, (list, tuple, set)):
            seq = list(v)
            head = seq[:max_items]
            extra = f"... +{len(seq) - max_items} more" if len(seq) > max_items else ""
            return f"{type(v).__name__}(len={len(seq)}) preview={head}{extra}"
        r = repr(v)
        return r if len(r) <= max_repr else r[:max_repr] + "... (truncated)"

    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f'"{k}": {summarize(v)}')
        return

    if hasattr(obj, "__dict__") or hasattr(obj, "__slots__"):
        # Prefer __dict__; fall back to __slots__
        items = vars(obj).items() if hasattr(obj, "__dict__") else (
            (slot, getattr(obj, slot)) for slot in getattr(obj, "__slots__", []) if hasattr(obj, slot)
        )
        for k, v in items:
            print(f'"{k}": {summarize(v)}')
        return

    # Fallback (tensor / scalar / other)
    print(f'"value": {summarize(obj)}')
