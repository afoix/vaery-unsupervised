#%%
import torch
import torch.nn.functional as F
from torch import nn, optim

class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()
        # print(f'{in_planes}_inplanes_before*stride{stride}')
        planes = in_planes * stride 
        # print(f'{planes}_planes_after*stride')
        # print(f'{in_planes}_inplanes_after*stride')
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # print(f'{planes}_planes_afterconvblock')
        self.planes = planes

        if stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Enc(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3, input_shape =128): #z_dims will go to 1, nc = 3, 
        super().__init__()
        self.in_planes = 64 #64 og, number of kernels
        self.z_dim = z_dim 
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False) #first a normal cnov block w 64 kernel
        self.bn1 = nn.BatchNorm2d(64) #after first conv you have 64 layers, you multiply together the RGB channels
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=2)  #BasicBlockenc takes stride, uses it to determine the output number of 
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.final_conv = nn.Conv2d(512*2, 2 * z_dim, kernel_size=3, stride = 2,padding=1)
        self.final_shape = int(input_shape / 2**6)
        self.final_linear = nn.Linear(
            in_features = 2 * z_dim * self.final_shape**2,
            out_features= 2 * z_dim
        )


    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride): #creates a sequential list of the BasicBlockDec
        strides = [stride] + [1]*(num_Blocks-1) #stride in first layer is 1, second - fourth is strides 2
        layers = []
        for stride in strides: 
            # print(f'{stride}stride_before_addlayer')
            # print(f'{planes}_planes_beforestrideloop')
            # print(f'{planes}_inplanes_beforestrideloop')
            layers += [BasicBlockEnc(self.in_planes, stride)] #Adds a layer for each BasicBlockEncoder
            self.in_planes = self.in_planes * stride #key change
            # print(f'{planes}_planes_afterstrideloop')
            # print(f'{self.in_planes}_inplanes_afterstrideloop')
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        x = torch.relu(self.bn1(self.conv1(x)))
        # print(x.shape) 
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)
        x = torch.relu(self.final_conv(x)) #relus are already in forward loop of make layer
        # print("befor_flattening",x.shape)
        b,c,h,w = x.shape
        x = x.view(b,c*h*w)
        # print(x.shape)
        x = torch.tanh(self.final_linear(x))
        # print(x.shape)
        mu, logvar = torch.chunk(x, 2, dim=1)
        return mu, logvar
    
class ResizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


    
class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)
        # print(f'{stride} stride before convolving')
        # print(f'{planes} planes before convolving')
        # print(f'{in_planes} in planes before convolving')

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, 
        # but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out = out + self.shortcut(x)
        out = torch.relu(out)
        return out

class UpsampleConv(nn.Module):
    def __init__(
            self, 
            in_channels,
            out_channels,
            scale_factor= 2, 
            kernel_size = 3
        ):
        super().__init__()
        self.upsample = ResizeConv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            scale_factor= scale_factor, 
            kernel_size = kernel_size
        )
        self.conv_1 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        self.conv_2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
    def forward(self, x):
        x = torch.relu(self.upsample(x))
        x = torch.relu(self.conv_1(x))
        x = torch.relu(self.conv_2(x))
        return x

class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3, out_features=28):
        super().__init__()
        self.in_planes = 512
        self.nc = nc
        self.out_features = out_features
        self.z_dim = z_dim

        self.linear_outfeatures = 4*4*z_dim
        #self.linear = nn.Conv2d(z_dim, 512, kernel_size=1) #original but we want a conv
        self.linear = nn.Linear(z_dim, self.linear_outfeatures)
        # print('making layer 4')
        
        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2) 
        # print('making layer 3')
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        # print('making layer 2')
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        # print('making layer 1')
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)

        # print('final conv')
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)
        #self.final_linear = torch.nn.Linear(32**2, self.out_features**2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        # print(f'{planes}_planes before stride loop')
        # print(f'{self.in_planes}_inplanes before stride loop')        
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
            # print(f'{self.in_planes} inplanes after stride loop')
            # print(f'{planes} planes after stride loop')            
        self.in_planes = planes
        # print(f'{planes} planes after redefining in_planes')   
        # print(f'{self.in_planes} inplanes after redefining in_planes')  
        return nn.Sequential(*layers) 

    def forward(self, z):
        # print(z.shape)
        x = self.linear(z)
        # print(f'{x.shape}_linearoutputshape')
        b,c = z.shape
        x = x.view(b, self.z_dim, 4, 4)
        # print(x.shape)
        x = self.upsample(x)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))

        b,c,w,h = x.shape
        # print(x.shape)
        #x = self.final_linear(x.view(b,c,w*h))
        return x#.view(b,c,self.out_features, self.out_features)
    
    
class LinearVAEResNet18(nn.Module):
    
    def __init__(self, nc, z_dim):
        super().__init__()
        self.encoder = ResNet18Enc(nc=nc, z_dim=z_dim)
        self.decoder = ResNet18Dec(nc=nc, z_dim=z_dim)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x = self.decoder(z)
        return x, z
    
    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean
#%%
shape = 128
input = torch.zeros(1,3,shape,shape)
encoder = ResNet18Enc(input_shape=shape)
encoder(input)[0].shape
# #%%
# import numpy as np
# from pathlib import Path
# from vaery_unsupervised.dataloaders.dataloader_km_ryans_template import (
#     simple_masking, 
#     DATASET_NORM_DICT, 
#     SpatProteoDatasetZarr,
#     SpatProtoZarrDataModule,
# )
# from vaery_unsupervised.km_utils import plot_batch_sample,plot_dataloader_output
# import monai.transforms as transforms
# from vaery_unsupervised.networks.LitVAE_km import SpatialVAE
# import lightning
# from lightning.pytorch.callbacks import ModelCheckpoint
# from lightning.pytorch.callbacks import Callback
# from lightning.pytorch.loggers import TensorBoardLogger
# from pathlib import Path

# out_path = Path("/mnt/efs/aimbl_2025/student_data/S-KM/")

# transform_both = [
#     transforms.RandAffine(
#         prob=0.5, 
#         rotate_range=3.14, 
#         shear_range=(0,0,0), 
#         translate_range=(0,20,20), 
#         scale_range=None,   
#         padding_mode="zeros",
#         spatial_size=(128,128)),
#     transforms.RandFlip(
#         prob = 0.5,
#         spatial_axis = [-1], 
#     ),
# ]
# transform_input = [
#     transforms.RandGaussianNoise(
#         prob = 0.5,
#         mean = 0,
#         std = 1
#     ),
# ]

# dataset_zarr = SpatProteoDatasetZarr(
#     out_path/"converted_crops_with_metadata.zarr",
#     masking_function=simple_masking,
#     dataset_normalisation_dict=DATASET_NORM_DICT,
#     transform_both=transform_both,
#     transform_input=transform_input
# )
# plot_dataloader_output(dataset_zarr[0])
# # %%
# lightning_module = SpatProtoZarrDataModule(
#     out_path/"converted_crops_with_metadata.zarr",
#     masking_function=simple_masking,
#     dataset_normalisation_dict=DATASET_NORM_DICT,
#     transform_both=transform_both,
#     transform_input=transform_input,
#     num_workers=1,
#     batch_size=1,
# )
# lightning_module.setup("train")

# loader = lightning_module.train_dataloader()
# # %% looking at a batch
# for i,batch in enumerate(loader):
#     plot_batch_sample(batch)    
#     break

# #%%
# encoderblock = ResNet18Enc(num_Blocks=[2,2,2,2], z_dim = 128, nc = 4)
# mu, log_var = encoderblock(batch["input"])

# #%%
# std = torch.exp(0.5 * log_var)
# std.unsqueeze(1).unsqueeze(2).shape
# #%%
# mu = mu.unsqueeze(2).unsqueeze(3)
# log_var = log_var.unsqueeze(2).unsqueeze(3)

# #%%
# def reparameterize(mean, log_var):
#   std = torch.exp(0.5 * log_var)
#   epsilon = torch.randn_like(std)
#   return mean + epsilon * std
# # %%
# z = reparameterize(mu, log_var)
# z = z#.unsqueeze(2).unsqueeze(3)
# # %%
# decoderblock = ResNet18Dec(num_Blocks=[2,2,2,2], z_dim = 128, nc = 4, out_features = 28)
# decoderblock(z).shape

# # %%
# def model_loss(x, recon_x, z_mean, z_log_var, beta = 1e-3):
#   mse = torch.nn.functional.mse_loss(x, recon_x)
#   kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
#   loss = mse + beta * kl_loss
#   return loss, mse, kl_loss
# # %%
# model_loss()
# %%
