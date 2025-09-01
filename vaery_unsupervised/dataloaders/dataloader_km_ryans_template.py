import logging
import spatialdata as sd
import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from monai.data import set_track_meta
from monai.data.utils import collate_meta_tensor
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    DictTransform,
    MapTransform,
    MultiSampleTrait,
    RandAffined,
)

from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.affinity import translate
from shapely.ops import unary_union
from skimage.draw import polygon as draw_polygon

def polygon_to_centered_mask(geom, size=64):
    """
    Convert a Shapely (Multi)Polygon in image coordinates (x=cols, y=rows)
    into a boolean mask of shape (size, size) centered at the polygon centroid.
    
    Parameters
    ----------
    geom : shapely.geometry.Polygon or MultiPolygon
        Geometry in the same coordinate system as the image (x right, y down).
    size : int
        Output mask width and height.
    
    Returns
    -------
    mask : np.ndarray (bool) of shape (size, size)
    offset_xy : tuple (float, float)
        (x_min, y_min) of the crop window in the original coordinate system.
        Useful if you want to crop the image: img[y_min:y_min+size, x_min:x_min+size]
        (after rounding/clipping to valid indices).
    """
    if geom.is_empty:
        return np.zeros((size, size), dtype=bool), (np.nan, np.nan)

    # centroid (x, y). Note: x ~ columns, y ~ rows
    cx, cy = geom.centroid.x, geom.centroid.y
    half = size / 2.0

    # crop window bounds in original coords (float)
    x_min = cx - half
    y_min = cy - half

    # translate geometry so that (x_min, y_min) maps to (0, 0) in the mask frame
    g = translate(geom, xoff=-x_min, yoff=-y_min)

    # clip to the mask frame to avoid degenerate or out-of-bounds coords
    frame = box(0.0, 0.0, float(size), float(size))
    g_clipped = g.intersection(frame)

    mask = np.zeros((size, size), dtype=bool)
    if g_clipped.is_empty:
        return mask, (x_min, y_min)

    # normalize to iterable of Polygons
    polys = []
    if isinstance(g_clipped, Polygon):
        polys = [g_clipped]
    elif isinstance(g_clipped, MultiPolygon):
        polys = list(g_clipped.geoms)
    else:
        # If intersection yielded a collection, attempt to union polygons
        polys = [p for p in unary_union(g_clipped).geoms] if hasattr(unary_union(g_clipped), "geoms") else []

    for poly in polys:
        if poly.is_empty:
            continue

        # exterior
        ex = np.asarray(poly.exterior.coords)
        rr, cc = draw_polygon(ex[:,1], ex[:,0], shape=mask.shape)  # rows=y, cols=x
        mask[rr, cc] = True

        # holes (interiors): subtract them
        for interior in poly.interiors:
            ih = np.asarray(interior.coords)
            rr_h, cc_h = draw_polygon(ih[:,1], ih[:,0], shape=mask.shape)
            mask[rr_h, cc_h] = False

    return mask, (x_min, y_min)

def extract_image_mask_poly_well(
        sdata,idx,
        name_image,
        name_coordinates,
        name_transformed_poly = 'affine_transformed', 
        size=128
):

    poly = sdata[name_transformed_poly].iloc[idx]['geometry']
    well = sdata[name_transformed_poly].index[idx]
    centroid = poly.centroid.coords[0]

    cropped = sdata.query.bounding_box(
        axes = ['x', 'y'],
        min_coordinate=(np.array(centroid)-size/2),
        max_coordinate=(np.array(centroid)+size/2),
        target_coordinate_system=name_coordinates
    )

    mask, _ = polygon_to_centered_mask(poly,128)

    return cropped[name_image].values, mask, poly, well

def load_sdata_files(file_list,shapes_name = 'affine_transformed'):
    
    sdata_objects = [
        sd.read_zarr(file) for file in file_list   
    ]

    lengths = [
        len(sdata[shapes_name]) for sdata in sdata_objects
    ]
    cumul_length = np.cumsum(lengths)
    total_length = cumul_length[-1]
    mapping = {}
    for i, (length, dataset_idx) in enumerate(
            zip(lengths, range(len(file_list)))
        ):
        for j in range(length):
            if i == 0:
                idx = j
            else:   
                idx = cumul_length[i-1] + j
            mapping[idx] = (dataset_idx, j)

    return (sdata_objects, mapping, total_length)

_logger = logging.getLogger("lightning.pytorch")

class SpatProteomicsDataset(Dataset):
    def __init__(
            self, 
            file_list:list[str],
            patient_id_list: list[str],
            masking_function,
            transform: DictTransform|None=None, 
            train_val_split =0.8,
            polygon_sdata_name = 'affine_transformed',
            seed=42,
            train:bool=True,
    ):
        super().__init__()
        # adding dataset_specific information
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

        # setting transform and train/val split
        self.transform = transform
        self.train = train
        self.masking_function = masking_function

        # get random seed to ensure same train / val split
        np.random.seed(seed)
        indices = np.random.permutation(total_length)
        split = int(train_val_split * total_length)
        self.train_indices = indices[:split]
        self.val_indices = indices[split:]

    def __len__(self):
        if self.train:
            return len(self.train_indices)
        else:
            return len(self.val_indices)

    def __getitem__(self, index):
        # Logic to get a sample from the dataset
        if self.train:
            index = self.train_indices[index]
        else:
            index = self.val_indices[index]

        dataset_idx, row_idx = self.mapping[index]

        image, mask, poly, well = extract_image_mask_poly_well(
            self.datasets[dataset_idx], 
            row_idx,
            name_transformed_poly = self.polygon_sdata_name,
            name_image=self.image_name_list[dataset_idx],
            name_coordinates=self.coordinate_name_list[dataset_idx],
        )
        metadata = {
            "dataset":self.file_list[dataset_idx], 
            "row_idx":row_idx,
            "well_id":well,
            "poly":poly
        }
        sample = self.masking_function(image, mask)
        if self.transform:
            # TODO FIGURE OUT TRANSFORMS
            NotImplementedError("This method is not implemented")
            # self.transform a list of transform (crop, rotation, flip, normalize, etc)
            
            sample = self.transform(sample)

        return sample, metadata


class SpatProteomicDataModule(LightningDataModule):
    def __init__(
            self, 
            data_paths:list[str],
            image_name_list:list[str],
            coordinate_name_list:list[str],
            sdata_polygon_name:str,
            masking_function,
            batch_size:int, 
            num_workers:int,
            transform: DictTransform|None=None,
            prefetch_factor:int=2,
            pin_memory:bool=True,
            persistent_workers:bool=False,
            seed=42,
        ):
        super().__init__()
        # adding dataset_specific information
        self.data_paths = data_paths
        self.image_name_list = image_name_list
        self.coordinate_name_list = coordinate_name_list
        self.sdata_polygon_name = sdata_polygon_name

        # setting masking function and transform
        self.masking_function = masking_function
        self.transform = transform

        # setup lightning datamodule parameters
        self.dataset = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        # NOTE: These parameters are for performance
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
    
    def prepare_data(self):
        # we don't need this
        pass

    def setup(self, stage: str):
        if stage == "fit" or stage == "train":
            self.dataset = SpatProteomicsDataset(
                file_list=self.data_paths, 
                masking_function=self.masking_function, 
                transform=self.transform,
                seed=self.seed,
                image_name_list=self.image_name_list,
                coordinate_name_list=self.coordinate_name_list,
                polygon_sdata_name=self.sdata_polygon_name,
            )

        elif stage == "val" or stage == "validate":
            self.dataset = SpatProteomicsDataset(
                file_list=self.data_paths, 
                train=False,
                masking_function=self.masking_function, 
                transform=self.transform,
                seed=self.seed,
                image_name_list=self.image_name_list,
                coordinate_name_list=self.coordinate_name_list,
                polygon_sdata_name=self.sdata_polygon_name,
            )

        elif stage == "predict":
            NotImplementedError("This method is not implemented")

    def train_dataloader(self):
        return DataLoader(self.dataset,shuffle=True,batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=True,prefetch_factor=self.prefetch_factor)

    def val_dataloader(self):
        return DataLoader(self.dataset,shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=True,prefetch_factor=self.prefetch_factor)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=True,prefetch_factor=self.prefetch_factor)