#%%
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
#    DictTransform, -> doesn't work
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
    sdata = sdata.subset([name_coordinates,name_image,name_transformed_poly])
    poly = sdata[name_transformed_poly].iloc[idx]['geometry']
    well = sdata[name_transformed_poly].index[idx]
    centroid = poly.centroid.coords[0]

    cropped = sdata.query.bounding_box(
        axes = ['x', 'y'],
        min_coordinate=(np.array(centroid)-size/2),
        max_coordinate=(np.array(centroid)+size/2),
        target_coordinate_system=name_coordinates
    )

    mask, _ = polygon_to_centered_mask(poly,size)

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

def simple_masking(image, mask):
    return image * mask

_logger = logging.getLogger("lightning.pytorch")

class SpatProteomicsDataset(Dataset):
    def __init__(
            self, 
            file_list:list[str],
            patient_id_list: list[str],
            masking_function,
            normalise_per_patient = True,
            transform: list|None=None, 
            train_val_split =0.8,
            polygon_sdata_name = 'affine_transformed',
            crop_size = 128,
            seed=42,
            train:bool=True,
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
        self.masking_function = masking_function
        self.normalise_per_patient = normalise_per_patient
        # setting transform and train/val split
        self.transform = transform
        self.train = train

        # get random seed to ensure same train / val split
        np.random.seed(seed)
        indices = np.random.permutation(total_length)
        split = int(train_val_split * total_length)
        self.train_indices = indices[:split]
        self.val_indices = indices[split:]
        
        self.get_means_stddev_per_dataset()

    def get_means_stddev_per_dataset(self):
        dataset_idx_to_images = {
            i:[] for i in range(len(self.datasets))
        }
        for idx in range(self.__len__()):
            img, dataset_idx = self.get_cropped_sample_for_norm(
                idx
            )
            dataset_idx_to_images[dataset_idx].append(img)
        
        print(np.array(dataset_idx_to_images[0]).shape)


    def __len__(self):
        if self.train:
            return len(self.train_indices)
        else:
            return len(self.val_indices)
        
    def get_cropped_sample_for_norm(self, index):
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
            size = self.crop_size
        )
        sample = self.masking_function(image, mask)
        return sample, dataset_idx

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
            size = self.crop_size
        )
        #print(f"image_shape: {image.shape}")
        #print(f"mask_shape: {mask.shape}")
        metadata = {
            "dataset":self.patient_id_list[dataset_idx], 
            "dataset_file":self.file_list[dataset_idx],
            "row_idx":row_idx,
            "well_id":well,
            "poly":poly
        }
        sample = self.masking_function(image, mask)
        if self.normalise_per_patient:
            sample = (
                sample - np.array(self.means[dataset_idx])[:,np.newaxis,np.newaxis]
            ) / np.array(self.std_devs[dataset_idx])[:,np.newaxis,np.newaxis]
        sample = torch.Tensor(sample)
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
            patient_id_list: list[str],
            sdata_polygon_name:str,
            masking_function,
            batch_size:int, 
            num_workers:int,
            crop_size = 128,
            normalise_per_patient = True,
            transform: list|None=None,
            prefetch_factor:int=2,
            pin_memory:bool=True,
            persistent_workers:bool=False,
            seed=42,
        ):
        super().__init__()
        # adding dataset_specific information
        self.data_paths = data_paths
        self.patient_id_list = patient_id_list
        self.sdata_polygon_name = sdata_polygon_name

        # setting masking function and transform
        self.crop_size = crop_size
        self.normalise_per_patient = normalise_per_patient
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
                patient_id_list=self.patient_id_list,
                masking_function=self.masking_function,
                normalise_per_patient=self.normalise_per_patient,
                crop_size=self.crop_size,
                transform=self.transform,
                polygon_sdata_name=self.sdata_polygon_name,
                seed=self.seed,
                train=True,
            )

        elif stage == "val" or stage == "validate":
            self.dataset = SpatProteomicsDataset(
                file_list=self.data_paths, 
                patient_id_list=self.patient_id_list,
                masking_function=self.masking_function,
                normalise_per_patient=self.normalise_per_patient,
                crop_size=self.crop_size,
                transform=self.transform,
                polygon_sdata_name=self.sdata_polygon_name,
                seed=self.seed,
                train=False,

            )

        elif stage == "predict":
            NotImplementedError("This method is not implemented")

    def train_dataloader(self):
        return DataLoader(self.dataset,shuffle=True,batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=True,prefetch_factor=self.prefetch_factor)

    def val_dataloader(self):
        return DataLoader(self.dataset,shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=True,prefetch_factor=self.prefetch_factor)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=True,prefetch_factor=self.prefetch_factor)
    
#%% testing
prefix = "/mnt/efs/aimbl_2025/student_data/S-KM/001_Used_Zarrs/onlybatch1/"
suffix = "_sdata_at"
name_list = ["450","453","456_2","457","493_2", "543_2"] #  something fishy going on with this one
file_list = [prefix + name + suffix for name in name_list]
file_list

#%%
dataset = SpatProteomicsDataset(
    file_list=file_list,
    patient_id_list=name_list,
    masking_function=simple_masking,
    normalise_per_patient=True,
    transform=None,
    train_val_split=0.9,
    polygon_sdata_name='affine_transformed',
    seed=42,
    train=True,
)
#%%
len(dataset)
# %%
def transpose_polygon_to_image(geom, size):
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

    return g
#%%
import matplotlib.pyplot as plt
def plot_single_sample(dataset, idx, channels = [1,2,3]):
    test_sample = dataset[idx]
    test_image = test_sample[0].numpy()
    test_poly = test_sample[1]["poly"]
    translated_poly = transpose_polygon_to_image(test_poly,128)
    x,y = translated_poly.exterior.xy

    plt.imshow(
        np.moveaxis(
            test_image[np.array(channels)]/np.max(
                test_image[np.array(channels)],axis=(1,2)
            )[:,np.newaxis,np.newaxis],
            0,
            -1
        )
    )
    plt.plot(x,y, color='yellow')

plot_single_sample(dataset, 400)
#%%

sdata = dataset.datasets[0]
mean = sdata["czi_450"].mean(dim = ['y','x'])
std = sdata["czi_450"].std(dim = ['y','x'])
# %%
# transforms = [affine, shear, gauss_noise, smoothing, flips, intensity, variation]
# %%
dataset[400][0].std(axis=(1,2))
# %%
