#%%
from iohub import open_ome_zarr
import torch
from pytorch_lightning import seed_everything
import numpy as np
from vaery_unsupervised.dataloaders.hcs_dataloader_ryan import HCSDataModule 
import matplotlib.pyplot as plt

# ome_zarr_path = "/mnt/efs/aimbl_2025/student_data/S-RM/full_dataset/RM_project_ome.zarr"
# plate = open_ome_zarr(ome_zarr_path, mode="r")
# positions = [pos for _, pos in plate.positions()]
# shuffled_indices = torch.randperm(len(positions))
# shuffled_positions = list(positions[i] for i in shuffled_indices)
# # shuffled_indices = self._set_fit_global_state(len(positions))

# # %%
# for idx, pos in plate.positions():
#     print(idx,pos)
#     zyx = pos.data[0,0,:,:,:]
#     print(zyx.shape)
#     break
# %%

if __name__ == "__main__":
    seed_everything(42)
    ome_zarr_path = "/mnt/efs/aimbl_2025/student_data/S-RM/full_dataset/RM_project_ome.zarr"

    dm = HCSDataModule(
        ome_zarr_path=ome_zarr_path,
        source_channel_names=['mito','er','nuclei'],
        weight_channel_name='nuclei',
        crop_size=(256, 256),
        crops_per_position=4,
        batch_size=1,
        num_workers=10,
        split_ratio=0.8,
        normalization_transform=[],
        augmentations=[]
    )
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    # val_loader = dm.val_dataloader()
    #%%

    for i,batch in enumerate(train_loader):
        print(batch['anchor'][0].shape)
        print(batch['positive'][0].shape)

        # Plot the two images
        plt.figure(figsize=(10, 5))
        
        # Plot anchor
        plt.subplot(1, 2, 1)
        plt.imshow(batch['anchor'][0][0,0])  # Change from (C,H,W) to (H,W,C)
        plt.title('Anchor')
        plt.axis('off')
        
        # Plot positive
        plt.subplot(1, 2, 2)
        plt.imshow(batch['positive'][0][0,0])  # Change from (C,H,W) to (H,W,C)
        plt.title('Positive')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

        if i>5:
            break
        

    # %%
