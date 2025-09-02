import logging

import lightning as L
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from iohub import open_ome_zarr
import numpy as np
#from monai.transforms import RandSpatialCropSamples

from typing import Tuple, Union, List

_logger = logging.getLogger("lightning.pytorch")

class SalBrainDataModule(L.LightningDataModule):
    """
    Notes:
    - DataModule initializes DataSet depending on the mode (e.g. train, val, predict)
    Goals:
    - Flexible to number of channels: add and exclude in the future
    - Flexible to spatial resolution
    - Try different sampling strategies 
    """
    
    def __init__(self, 
                 data_path: str="/home/jnc2161/mbl/registered_salamander_data.zarr",
                 batch_size: int=9, 
                 patch_size: Union[Tuple, List]=(32, 32, 32), 
                 #normalizations: List=[], 
                 #augmentations: List=[], 
                 pin_memory: bool=False,
                 num_workers: int=32, 
                 prefetch_factor: int=2,
                 persistent_workers: bool=True
                 ):
        
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.patch_size = patch_size

        # self.transformations = Compose([
        #     Normalize(mean=[0.0], std=[1.0])
        # ])
        #self.augmentations = augmentations

        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        #self.dataset = SalBrainDataset(self.data_path, self.patch_size)


    def setup(self, stage: str):
        """
        Initializes the Dataset object based on stage (train, val or predict)
        """

        if stage == "fit" or stage == "train":
            self.dataset = SalBrainDataset(self.data_path, 
                                           self.batch_size, 
                                           self.patch_size) 
                                           #transformations=self.transformations)
            self.channel_names = self.dataset.channel_names
        elif stage == "val" or stage == "validate":
            self.dataset = SalBrainDataset(self.data_path, 
                                           self.batch_size, 
                                           self.patch_size) 
                                           #transformations=self.transformations)
            self.channel_names = self.dataset.channel_names
        elif stage == "predict":
            raise NotImplementedError("Predict stage not implemented yet.")
        else:
            raise ValueError("Stage must be fit/train, val/validate, or predict.")
        

    def train_dataloader(self):
        return DataLoader(self.dataset, 
                          shuffle=True, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          pin_memory=self.pin_memory,
                          prefetch_factor=self.prefetch_factor,
                          persistent_workers=self.persistent_workers)
    
    def validation_dataloader(self):
        return DataLoader(self.dataset, 
                          shuffle=False, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          pin_memory=self.pin_memory,
                          prefetch_factor=self.prefetch_factor,
                          persistent_workers=self.persistent_workers)


class SalBrainDataset(Dataset):
    """
    Handles data IO, prepares batches of data, applies transformations
    Sampling strategy:
    - Because the task is unsupervised, it might be okay to sample patches randomly from the volume without
    explicitly seperating training vs validation patches. The changes of getting the exact same patch is low. 
    Use of mask:
    - Mask (of brain) will be used to restrict patch sampling. Initially, just define a bounding box. Might want to
    try a more sophisticated approach where the centers of each patch are sampled from the mask.
    - What to do about filling the non-brain pixel values? 
    """

    def __init__(self, 
                 data_path: str, 
                 batch_size: int, 
                 patch_size: Union[List, Tuple], 
                 mask_channel: int=1,
                 #normalizations: List=[], 
                 #augmentations: List=[], 
                 ):
        
        super().__init__()

        self.data_path = data_path
        self.patch_size = patch_size
        self.exclude_channels = [4, 5]

        with open_ome_zarr(
            self.data_path,
            mode='r'
        ) as dataset:
            self.volume = dataset["0/0/0/0"] # dims is (T, C, Z, Y, X), (1, 68, 530, 1189, 585)
            self.channel_names = [c for i, c in enumerate(dataset["0/0/0"].channel_names) if i not in self.exclude_channels]
        self.n_channels = len(self.channel_names)
        self.mask = self.volume[0, mask_channel].copy().astype(bool)
        self.bounding_box = get_bounding_box(self.mask)
        self.volume_shape = self.volume.shape

        self.transformations = v2.Compose([v2.Normalize(mean=[0.0 for i in range(self.n_channels)], 
                                                  std=[1.0 for i in range(self.n_channels)])])
        self.indices = random_patch_sampler(self.bounding_box, 
                                            self.patch_size, 
                                            num_samples=batch_size,  # *100??
                                            binary_mask=self.mask)


    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, 
                    index):
        
        vol = self.volume[0, 
                           [c for c in range(self.volume.shape[1]) if c not in self.exclude_channels], 
                           self.indices[index][0]:self.indices[index][1], 
                           self.indices[index][2]:self.indices[index][3], 
                           self.indices[index][4]:self.indices[index][5]]


        mean = np.mean(vol, axis=(1, 2, 3), keepdims=True)
        std = np.std(vol, axis=(1, 2, 3), keepdims=True)
        std = np.where(std == 0, 1, std)


        vol = (vol - mean) / std

        return vol


def get_bounding_box(brainary_mask: np.array, pad: int=0) -> Tuple:
    """
    Use a binary mask to compute a bounding box for the brain in the 
    space of the volume
    """
    volume_shape = brainary_mask.shape
    z_coords, y_coords, x_coords = np.where(brainary_mask > 0)
    z_min, z_max = np.min(z_coords) - pad, np.max(z_coords) + pad
    y_min, y_max = np.min(y_coords) - pad, np.max(y_coords) + pad
    x_min, x_max = np.min(x_coords) - pad, np.max(x_coords) + pad

    bb_size = (z_max - z_min, y_max - y_min, x_max - x_min)
    print(f"Bounded size {bb_size[0], bb_size[1], bb_size[2]}")

    bbox = (z_min, z_max, y_min, y_max, x_min, x_max)
    assert all(b >= 0 for b in bbox), f"Bounding box is out of bounds: {bbox}, padding may be too large"
    assert all(z_max < volume_shape[1] for z_max in bbox[1::2]), f"Bounding box is out of bounds: {bbox}, padding may be too large"

    return (z_min, z_max, y_min, y_max, x_min, x_max)


def random_patch_sampler(bbox: Tuple, 
                        patch_size: Union[List, Tuple, int], 
                        num_samples: int, 
                        min_coverage: float=0.4,
                        binary_mask: np.array=None) -> List[Tuple]:
    """
    Sample random patches from the given bounding box. 
    Parameters
    ----------
    bbox : Tuple
        The bounding box from which to sample patches.
    patch_size : Union[List, Tuple, int]
        The size of the patches to sample.
    num_samples : int
        The number of patches to sample.
    Returns
    -------
    List[Tuple]
        A list of (z_start, z_end, ... x_start, x_end) ranges. Corresponds to the global coordinates 
        of the volume.
    """
    if isinstance(patch_size, int):
        patch_size = (patch_size,) * 3

    patches = []
    i = 0
    while i < num_samples:
        z_start = np.random.randint(bbox[0], bbox[1] - patch_size[0])
        y_start = np.random.randint(bbox[2], bbox[3] - patch_size[1])
        x_start = np.random.randint(bbox[4], bbox[5] - patch_size[2])
        z_end = z_start + patch_size[0]
        y_end = y_start + patch_size[1]
        x_end = x_start + patch_size[2]

        if binary_mask is not None:
            coverage = np.sum(binary_mask[z_start:z_end, y_start:y_end, x_start:x_end])/(patch_size[0] * patch_size[1] * patch_size[2])
            #_db_log.debug(f"Patch {i}: coverage {coverage:.2f}")
            if coverage < min_coverage:
                #_db_log.debug("Skipping patch due to insufficient coverage")
                continue
            else:
                patches.append((z_start, z_end,
                                y_start, y_end,
                                x_start, x_end))
                i += 1
    return patches
