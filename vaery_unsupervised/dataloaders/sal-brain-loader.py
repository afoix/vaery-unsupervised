import lightning as L
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from iohub import open_ome_zarr
import numpy as np
from monai.transforms import RandSpatialCropSamples

from typing import Tuple, Union, List

class SalBrainDataModule(L.LightningDataModule):
    """
    logic regarding interaction of data and training routines 
    """
    
    def __init__(self, 
                 data_path: str="/home/jnc2161/mbl/registered_salamander_data.zarr",
                 batch_size: int=32, 
                 patch_size: Union[Tuple, List]=(32, 32, 32), 
                 normalizations: List=[], 
                 augmentations: List=[], 
                 pin_memory: bool=False,
                 num_workers: int=32
                 ):
        
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.normalizations = normalizations
        self.augmentations = augmentations
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.dataset = SalBrainDataset(self.data_path, self.patch_size)

        def train_val_split():


        def setup(self, stage: str):



            if stage == "fit" or stage == "train":
                
            elif stage == "val" or stage == "validate":
                ...
            elif stage == "predict":
                ...
            else:
                ValueError("Stage must be fit/train, val/validate, or predict.")
        
        def train_dataloader(self, ):
            ...

        def prepare_data(self):
            """
            Get samples from the 
            """
            
class SalBrainDataset(Dataset):
    """
    For grabbing a sample of brain from the OME-ZARR file
    - transforms
    - measure the length of the thing
    - 
    """
    def __init__(self, 
                 data_path: str, 
                 patch_size: Union[List, Tuple],
                #  normalizations: List=[], 
                #  augmentations: List=[], 
                 ):
        
        super().__init__(self)

        self.data_path = data_path
        # self.normalizations = normalizations
        # self.augmentations = augmentations
        #self.position_key = "0/0/0/0"
        #self.volume_shape = (530, 1189, 586)
        #self.num_samples = 
        self.patch_size = patch_size

        with open_ome_zarr(
            self.data_path,
            mode='r'
        ) as dataset:
            self.volume = dataset["0/0/0/0"]
            self.channel_names = dataset["0/0/0"].channel_names

        self.mask_chan_ind = [i for i, chan in enumerate(self.channel_names) if "mask" in chan]
        assert len(self.mask_chan_ind) == 0, f"Multiple channel names contain the word mask: {self.mask_chan_ind}. Binary mask ambiguous"
        self.mask_chan_ind = self.mask_chan_ind[0]
        self.bounding_box = get_bounding_box(self, self.volume[0, self.mask_chan_ind])
        self.volume_shape = self.volume.shape
    
    def __len__(self):
        return self.num_samples


def get_bounding_box(brainary_mask: np.array) -> Tuple:

    z_coords, y_coords, x_coords = np.where(brainary_mask > 0)
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    x_min, x_max = np.min(x_coords), np.max(x_coords)

    return (z_min, z_max, y_min, y_max, x_min, x_max)


