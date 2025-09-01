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
from spatialdata.dataloader import ImageTilesDataset
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.affinity import translate
from shapely.ops import unary_union
from skimage.draw import polygon as draw_polygon

DATASET_NORM_DICT = {
    '450': (
        np.array([8193.28315909,  669.43053592,  702.62678794,  375.91258369]),
        np.array([791.26713011, 417.95098902, 247.56795527, 294.39635837])),
    '453': (
        np.array([8189.4060447 ,  462.4284883 ,  382.43056141,  114.06013627]),
        np.array([1282.18826538,  333.59554946,  149.90218956,   52.23534189])),
    '456_2': (
        np.array([8208.27391122,  713.2599626 ,  457.31603657,  219.30696453]),
        np.array([1252.39658392,  289.24816494,  100.44187674,  100.16392884])),
    '457': (
        np.array([8199.63903064,  678.65686303,  589.10372731,  286.77450606]),
        np.array([1330.71658258,  475.29554154,  220.51604021,  237.50510507])),
    '493_2': (
        np.array([8166.68329065,  851.66964514,  707.97295588,  189.50970736]),
        np.array([1071.74627862,  515.23908128,  194.46742114,   81.4215602 ])),
    '543_2': (
        np.array([8190.31123317,  737.43471155,  590.31072146,  165.17529439]),
        np.array([891.61264098, 544.82783817, 184.27180556,  94.89693314]))
}

def polygon_to_centered_mask(geom, size=64): #This will be applied to the image coords to get an image around the bounding box of the object
    
    """
    Convert a Shapely (Multi)Polygon in image coordinates (x=cols, y=rows)
    into a boolean mask of shape (size, size) centered at the polygon centroid. #This will be applied to the image coords to get an image around the bounding box of the object
    
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
    #This defines the coordinates of the bounding box to 0,0 for the mask
    g = translate(geom, xoff=-x_min, yoff=-y_min)

    # clip to the mask frame to avoid degenerate or out-of-bounds coords
    frame = box(0.0, 0.0, float(size), float(size)) #box(minx, miny, maxx, maxy, ccw=True)
    g_clipped = g.intersection(frame) #This crops the mask frame of the cell to force it to have the size of the frame
    #Cells that are larger than the bounding box will be cropped if they weren't already

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
        rr, cc = draw_polygon(ex[:,1], ex[:,0], shape=mask.shape)  # rows=y, cols=x, the : keeps all channels
        mask[rr, cc] = True #Making them true means that you can map the values of the pixels later when you apply it to the image

        # holes (interiors): subtract them
        for interior in poly.interiors: #poly.interiors returns all dataframes that are within that shape
            ih = np.asarray(interior.coords)
            rr_h, cc_h = draw_polygon(ih[:,1], ih[:,0], shape=mask.shape)
            mask[rr_h, cc_h] = False #Within the interiors (which you draw above) make them false

    return mask, (x_min, y_min) #This will probably be applied within a loop later to apply to each image

def extract_image_mask_poly_well(
        sdata,idx,
        name_image,
        name_coordinates,
        name_transformed_poly = 'affine_transformed', 
        size=128
):
    sdata = sdata.subset([name_coordinates,name_image,name_transformed_poly]) #subset sdata object to keep coordinates, the image, as well as the transformed
    poly = sdata[name_transformed_poly].iloc[idx]['geometry'] #the polygon subsets the gdf and gives you the geometry from that idx
    well = sdata[name_transformed_poly].index[idx] #this gives you the plate id for that cell, ie the actual shape
    centroid = poly.centroid.coords[0] #gives you the centroid of the polygon

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
    ] #how long is the affine_transformed gdf in the xml_list in sdata_objects
    cumul_length = np.cumsum(lengths) ##?
    total_length = cumul_length[-1] ##?
    mapping = {}
    for i, (length, dataset_idx) in enumerate(
            zip(lengths, range(len(file_list))) #this will return i=0, length=100, dataset_idx=0
        ):
        for j in range(length): #This iterates through the length of the xml file to get those indices
            if i == 0:
                idx = j 
            else:   
                idx = cumul_length[i-1] + j
            mapping[idx] = (dataset_idx, j) #This fills the mapping in the dictionary to have the idx,
            #and then that mapped to the dataset_idx (this gets you to sdata) and then the idx of the shape

    return (sdata_objects, mapping, total_length)

def simple_masking(image, mask):
    return image * mask #Where mask is 0 or False, you lose the image information

def get_crop_mean_stdev(sdata,data_name, crop_size =128):
    test_dataset = ImageTilesDataset(
        sdata,
        regions_to_images={'affine_transformed':f'czi_{data_name}'}, 
        regions_to_coordinate_systems={'affine_transformed':f'aligned_{data_name}'},
        tile_scale=4,
        tile_dim_in_units=128,
    )
    all_crops = np.array(
        [
            test_dataset[i][f'czi_{data_name}'] for i in range(len(test_dataset))
        ]
    ) # dimensions are (n_crops, channels, y, x)
    
    return all_crops.mean(axis=(0,2,3)), all_crops.std(axis=(0,2,3))


_logger = logging.getLogger("lightning.pytorch") #????

class SpatProteomicsDataset(Dataset):
    def __init__(
            self, 
            file_list:list[str],
            patient_id_list: list[str],
            masking_function,
            dataset_normalisation_dict = None,
            transform_input: list|None=None, 
            transform_both: list|None=None,
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
        self.dataset_normalisation_dict = dataset_normalisation_dict
        # setting transform and train/val split
        self.transform_both = transform_both
        self.transform_input = transform_input
        self.train = train

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
            size = self.crop_size
        )
        #print(f"image_shape: {image.shape}")
        #print(f"mask_shape: {mask.shape}")
        patient_id_sample = self.patient_id_list[dataset_idx]
        metadata = {
            "dataset":patient_id_sample, 
            "dataset_file":self.file_list[dataset_idx],
            "row_idx":row_idx,
            "well_id":well,
            # "poly":[poly] #TODO deal with this later
        }
        
        if self.dataset_normalisation_dict:
            mean, stddev = self.dataset_normalisation_dict[patient_id_sample]
            image = (
                image - np.array(mean)[:,np.newaxis,np.newaxis]
            ) / np.array(stddev)[:,np.newaxis,np.newaxis]
        input = self.masking_function(image, mask)
        input = torch.Tensor(input)
        raw = input.clone()
        if self.transform_both:
            
            composed_transform = Compose(transforms=self.transform_both)
            
            input = composed_transform(input)
        
        target = input.clone()

        if self.transform_input:
            
            composed_transform_input = Compose(transforms=self.transform_input)

            input = composed_transform_input(input)
            


        return {"raw": raw, "input":input, "target":target, "metadata":metadata}


class SpatProteomicDataModule(LightningDataModule):
    def __init__(
            self, 
            data_paths:list[str],
            patient_id_list: list[str],
            polygon_sdata_name:str,
            masking_function,
            batch_size:int, 
            num_workers:int,
            crop_size = 128,
            dataset_normalisation_dict = None,
            transform_both: list|None=None,
            transform_input: list|None=None,
            prefetch_factor:int=2,
            pin_memory:bool=True,
            persistent_workers:bool=False,
            seed=42,
            train_val_split = 0.9
        ):
        super().__init__()
        # adding dataset_specific information
        self.data_paths = data_paths
        self.patient_id_list = patient_id_list
        self.polygon_sdata_name = polygon_sdata_name

        # setting masking function and transform
        self.crop_size = crop_size
        self.dataset_normalisation_dict = dataset_normalisation_dict
        self.masking_function = masking_function
        self.transform_input = transform_input
        self.transform_both = transform_both
        self.train_val_split = train_val_split

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
                dataset_normalisation_dict=self.dataset_normalisation_dict,
                crop_size=self.crop_size,
                transform_input=self.transform_input,
                transform_both = self.transform_both,
                polygon_sdata_name=self.polygon_sdata_name,
                seed=self.seed,
                train=True,
                train_val_split = self.train_val_split
            )

        elif stage == "val" or stage == "validate":
            self.dataset = SpatProteomicsDataset(
                file_list=self.data_paths, 
                patient_id_list=self.patient_id_list,
                masking_function=self.masking_function,
                dataset_normalisation_dict=self.dataset_normalisation_dict,
                crop_size=self.crop_size,
                transform_input=None,
                transform_both = None,
                polygon_sdata_name=self.polygon_sdata_name,
                seed=self.seed,
                train=False,
                train_val_split = self.train_val_split

            )

        elif stage == "predict":

            self.dataset = SpatProteomicsDataset(
                file_list=self.data_paths, 
                patient_id_list=self.patient_id_list,
                masking_function=self.masking_function,
                dataset_normalisation_dict=self.dataset_normalisation_dict,
                crop_size=self.crop_size,
                transform_input=None,
                transform_both = None,
                polygon_sdata_name=self.sdata_polygon_name,
                seed=self.seed,
                train=True,
                train_val_split=1
                )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size, 
            # num_workers=self.num_workers,
            # pin_memory=self.pin_memory,
            # prefetch_factor=self.prefetch_factor
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            shuffle=False, 
            batch_size=self.batch_size, 
            # num_workers=self.num_workers,
            # pin_memory=self.pin_memory,
            # prefetch_factor=self.prefetch_factor
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset, 
            shuffle=False,
            batch_size=self.batch_size, 
            # num_workers=self.num_workers,
            # pin_memory=self.pin_memory,
            # prefetch_factor=self.prefetch_factor
        )
    
#%% testing
prefix = "/mnt/efs/aimbl_2025/student_data/S-KM/001_Used_Zarrs/onlybatch1/"
suffix = "_sdata_at"
name_list = ["450","453","456_2","457","493_2", "543_2"] 
file_list = [prefix + name + suffix for name in name_list]
file_list

# means_stddev_dict = {}
# for data_name in name_list:
#     file_name = prefix + data_name + suffix 
#     sdata = sd.read_zarr(file_name)
#     mean, stddev = get_crop_mean_stdev(sdata,data_name)
#     means_stddev_dict[data_name] = (mean, stddev)
#%%
means_stddev_dict = DATASET_NORM_DICT
#%%

dataset = SpatProteomicsDataset(
    file_list=file_list,
    patient_id_list=name_list,
    masking_function=simple_masking,
    dataset_normalisation_dict=means_stddev_dict,
    transform_input=None,
    transform_both = None,
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
# %%
def plot_dataloader_output(data_dict, figsize=(20, 10)):
    """Plot images from data loader output, showing last 3 channels as RGB."""
    
    image_keys = [k for k in ["raw", "input", "target"] if k in data_dict]
    fig, axes = plt.subplots(1, len(image_keys), figsize=figsize)
    if len(image_keys) == 1:
        axes = [axes]
    
    for ax, key in zip(axes, image_keys):
        # Convert tensor to numpy and take last 3 channels: (4,y,x) -> (y,x,3)
        img = data_dict[key].detach().cpu().numpy()[-3:].transpose(1, 2, 0)
        
        # Normalize to [0,1] for display
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        ax.imshow(img)
        ax.set_title(key.capitalize())
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

sdata = dataset.datasets[0]
mean = sdata["czi_450"].mean(dim = ['y','x'])
std = sdata["czi_450"].std(dim = ['y','x'])
# %%
# transforms = [affine, flips, shear, gauss_noise, smoothing, intensity, variation]
# %%
for i in range(600)[::20]:
    print(dataset[i]["raw"].mean(axis=(1,2)))
# %%
import monai
import monai.transforms as transforms
# %%
transform_both = [
    transforms.RandAffine(prob=0.5, 
        rotate_range=3.14, 
        shear_range=(0,0,0), 
        translate_range=(0,20,20), 
        scale_range=None,   
        padding_mode="zeros",
        spatial_size=(128,128)),
    transforms.RandFlip(prob = 0.5,
                                spatial_axis = [-1], 
                            ),]

transform_input = [
    transforms.RandGaussianNoise(
        prob = 0.5,
        mean = 0,
        std = 1
    ),
]
# %%
transformed_dataset = SpatProteomicsDataset(
    file_list=file_list,
    patient_id_list=name_list,
    masking_function=simple_masking,
    dataset_normalisation_dict=means_stddev_dict,
    transform_input=transform_input,
    transform_both = transform_both,
    train_val_split=0.9,
    polygon_sdata_name='affine_transformed',
    seed=42,
    train=True,
)
# %%
for i in np.random.randint(0, len(transformed_dataset), 10):
    x = transformed_dataset[i]
    plot_dataloader_output(x)

# %%
lightningmodule = SpatProteomicDataModule(
    data_paths=file_list,
    patient_id_list=name_list,
    masking_function=simple_masking,
    dataset_normalisation_dict=means_stddev_dict,
    transform_input=transform_input,
    transform_both = transform_both,
    polygon_sdata_name='affine_transformed',
#     num_workers=2,
#     batch_size=4,
#     prefetch_factor=None,
#     pin_memory=False,
)
lightningmodule.setup("train")
# %%
loader = DataLoader(
    lightningmodule.dataset,
    batch_size=4,
    shuffle = False
)
for batch in loader:
    tes_batch = batch
    break
#%%
tes_batch["raw"].shape
# %%
loader = lightningmodule.train_dataloader()
#%%
for batch in loader:
    tes_batch = batch
    break
tes_batch

# %%
def plot_batch_sample(batch_dict, figsize=(12, 4)):
    """Plot single sample from batched data loader output."""
    for idx in range(len(batch_dict["raw"])):
        sample = {k: v[idx] for k, v in batch_dict.items() if k != "metadata"}
        plot_dataloader_output(sample, figsize)

plot_batch_sample(tes_batch)
# %%
