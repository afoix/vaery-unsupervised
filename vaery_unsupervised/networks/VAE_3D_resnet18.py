import logging
import torch
import torch.nn.functional as F
from torch import nn, optim

_logger = logging.getLogger("lightning.pytorch")
# _logger.setLevel(logging.DEBUG)

class ResizeConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):

    def __init__(self, in_features, stride=1):
        super().__init__()

        features = in_features*stride

        self.conv1 = nn.Conv3d(in_features, features, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(features)
        self.conv2 = nn.Conv3d(features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(features)

        if stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_features, features, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(features)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):
    def __init__(self, in_features, stride=1):
        super().__init__()
        features = int(in_features/stride)

        self.conv2 = nn.Conv3d(in_features, in_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(in_features)
        # self.bn1 could have been placed here, 
        # but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv3d(in_features, features, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm3d(features)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv3d(in_features, features, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm3d(features)
            self.shortcut = nn.Sequential(
                ResizeConv3d(in_features, features, kernel_size=3, scale_factor=stride),
                nn.BatchNorm3d(features)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out = out + self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Enc(nn.Module):
    """
    INITIAL LAYERS:
    Conv1 nc -> 64 features
    BatchNorm
    BASIC BLOCKS:
    layer1 64 -> 64 
    layer2 64 -> 128
    layer3 128 -> 256
    layer4 256 -> 512
    TO LATENT SPACE:
    linear 512 -> z_dim
    """
    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3, matrix_size=32):
        super().__init__()
        self.in_features = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv3d(in_channels=nc, 
                               out_channels=64, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm3d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1) # 32->16 ds2
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2) # 16->8
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2) # 8->4
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2) # 4->2
        # Issue: z = (B,z_dim*2, 2, 2, 2)
        #self.linear = nn.Conv3d(512, 2 * z_dim, kernel_size=1)
        self.linear = nn.Linear(int(512 * (matrix_size/16)**3), 2 * z_dim)

    def _make_layer(self, BasicBlockEnc, features, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_features, stride)]
            self.in_features = features
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        _logger.debug(f"x after layer 1 {x.shape}")
        x = self.layer2(x)
        _logger.debug(f"x after layer 2 {x.shape}")
        x = self.layer3(x)
        _logger.debug(f"x after layer 3 {x.shape}")
        x = self.layer4(x)
        _logger.debug(f"x after layer 4 {x.shape}")

        x = torch.flatten(x, start_dim=1)
        _logger.debug(f"x after flatten {x.shape}")
        x = self.linear(x)

        _logger.debug(f"z before chunk: {x.shape}")
        mu, logvar = torch.chunk(x, 2, dim=1)
        _logger.debug(f"mu logvar: {mu.shape} {logvar.shape}")
        return mu, logvar

class ResNet18Dec(nn.Module):
    """
    INITIAL LAYERS:
    Linear z_dim -> 512
    BASIC BLOCKS:
    layer4 512 -> 256 
    layer3 256 -> 128
    layer2 128 -> 64
    layer1 64 -> 64
    SIGMOID(Conv1 64 -> nc)
    """

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3, out_features=32, matrix_size=32):
        super().__init__()
        self.in_features = 512
        self.z_dim = z_dim
        self.nc = nc
        self.out_features = out_features
        self.matrix_size = matrix_size

        # self.linear = nn.Conv3d(in_channels=z_dim, 
        #                         out_channels=512, 
        #                         kernel_size=1)
        #in_size = int(64 / matrix_size)
        #self.linear = nn.Linear(z_dim, 512) #* int(64/matrix_size)**3)
        self.upsample = ResizeConv3d(in_channels=z_dim, out_channels=self.in_features, scale_factor=2, kernel_size=3)
        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)

        self.conv1 = ResizeConv3d(64, nc, kernel_size=3, scale_factor=2)
        #self.final_linear = torch.nn.Linear(32**3, self.out_features**3)

    def _make_layer(self, BasicBlockDec, features, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_features, stride)]
        self.in_features = features
        return nn.Sequential(*layers)

    def forward(self, z):

        #in_mat_sz = int(64/self.matrix_size)

        # x = self.linear(z)
        # _logger.debug(f"x after linear {x.shape}")
        # b, _ = z.shape
        # x = x.view(b, self.z_dim, 2, 2, 2)
        # _logger.debug(f"x after view {x.shape}")
        x = z.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = self.upsample(x)
        _logger.debug(f"x after upsample {x.shape}")
        # x = x.view(-1, self.zdim, in_mat_sz, in_mat_sz, in_mat_sz)
        # _logger.debug(f"x after view zdim x 2 x 2 x 2 {x.shape}")

        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))

        return x
    
    
class VAEResNet18(nn.Module):
    
    def __init__(self, nc, z_dim, matrix_size):
        super().__init__()
        self.encoder = ResNet18Enc(nc=nc, z_dim=z_dim)
        self.decoder = ResNet18Dec(nc=nc, z_dim=z_dim)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        _logger.debug(f"Z after reparam: {z.shape}")
        x = self.decoder(z)
        return x, z
    
    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean
