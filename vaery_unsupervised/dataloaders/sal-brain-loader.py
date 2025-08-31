import lightning as L
from torch.utils.data import Dataset

from typing import Tuple, Union, List

class SalBrainDataModule(L.LightningDataModule):
    
    def __init__(self, 
                 batch_size: int=32, 
                 patch_size: Union[Tuple, List]=(32, 32, 32), 
                 normalizations: List=[], 
                 augmentations: List=[], 
                 pin_memory: bool=False 
                 ):
        
        super().__init__()
        self.data_path = "/mnt/data0/home/jnc2161/mbl/registered_salamander_data.zarr"
        self.batch_size = batch_size
        self.patch_size
        self.train_data = 

        def setup(self, stage: str):
            if stage == "fit" or stage == "train":
                ...
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
    """
    def __init__(self, 
                 data_path: str, 
                 normalizations: List=[], 
                 augmentations: List=[]):
        self.data_path = data_path
        self.normalizations = normalizations
        self.augmentations = augmentations

    def __len__(self):
        return len()

