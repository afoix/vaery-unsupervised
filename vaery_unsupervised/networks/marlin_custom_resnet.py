from typing import Literal
import torch
import torch.nn as nn
from torch import Tensor

def projection_mlp(
        in_dims: int,
        hidden_dims: int,
        out_dims: int
) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dims, hidden_dims),
        nn.BatchNorm1d(hidden_dims),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dims, out_dims),
        nn.BatchNorm1d(out_dims),
    )

from typing import Literal, Tuple
import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------
# Utilities
# ---------------------------

def _norm2d(num_channels: int, kind: Literal["group", "batch"] = "group", groups: int = 8):
    if kind == "batch":
        return nn.BatchNorm2d(num_channels)
    # clamp groups to channels
    return nn.GroupNorm(min(groups, num_channels), num_channels)


class DropPath(nn.Module):
    """Stochastic depth. Drop residual path at rate p (per sample)."""
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        mask = torch.empty((x.shape[0],) + (1,) * (x.ndim - 1), dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x / keep * mask


class BasicBlock(nn.Module):
    """Standard 3×3 + 3×3 ResNet block with configurable norm, stride, dilation, and drop-path."""
    expansion = 1

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        dilation: int = 1,
        norm: Literal["group", "batch"] = "batch",
        gn_groups: int = 1,
        drop_path: float = 0.0,
        activation: nn.Module = nn.SiLU(inplace=True),
    ):
        super().__init__()
        self.act = activation if isinstance(activation, nn.Module) else nn.SiLU(inplace=True)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn1 = _norm2d(out_ch, norm, gn_groups)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn2 = _norm2d(out_ch, norm, gn_groups)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                _norm2d(out_ch, norm, gn_groups),
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.drop_path(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.act(out + identity)
        return out


# ---------------------------
# Small-object-friendly ResNet-10
# ---------------------------

class SmallObjectResNet10Encoder(nn.Module):
    """
    A ResNet-10 (layers=[1,1,1,1]) tailored for tiny features
    - 3×3, stride=1 stem (no 7×7, no early downsample)
    - No maxpool
    - Downsample only in layer2 & layer3 (stride=2)
    - layer4 keeps stride=1 and uses dilation (default=2) for bigger receptive field
    - Default stage widths are slim: (32, 64, 128, 256) to reduce overfitting
    - GroupNorm by default (better for small batches)
    Returns (embedding, projection), where embedding is pooled features from `feature_stage`.
    """

    def __init__(
        self,
        in_channels: int = 1,
        widths: Tuple[int, int] = (32, 64),
        norm: Literal["group", "batch"] = "group",
        stride: int | Tuple[int, int] = 1,
        gn_groups: int = 8,
        drop_path_rate: float = 0.05,  # mild stochastic depth across 4 blocks
        mlp_hidden_dims: int = 64,
        projection_dim: int = 32,
        activation: nn.Module = nn.SiLU(inplace=True),
    ):
        super().__init__()
        self.act = activation if isinstance(activation, nn.Module) else nn.SiLU(inplace=True)
        self.stride = stride
        # Stem: 3×3, stride=1, no maxpool
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], kernel_size=3, stride=1, padding=1, bias=False),
            _norm2d(widths[0], norm, gn_groups),
            self.act,
        )

        # ResNet-10 uses 1 basic block per stage
        # depths = [1, 1, 1, 1]
        depths = [1, 1]
        total_blocks = sum(depths)
        # linearly increase drop-path over blocks
        dp_rates = [drop_path_rate * i / max(1, (total_blocks - 1)) for i in range(total_blocks)]

        c1, c2 = widths
        b = 0  # block index for dp rate schedule

        # layer1 (no downsample, no dilation)
        self.layer1 = self._make_layer(
            in_ch=c1, out_ch=c1, blocks=depths[0], stride=self.stride, dilation=1,
            norm=norm, gn_groups=gn_groups, dp_rates=dp_rates[b:b + depths[0]],
        )
        b += depths[0]

        self.inter_layer_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)        

        # layer2 (downsample to 75×75 for 150×150 input)
        self.layer2 = self._make_layer(
            in_ch=c1, out_ch=c2, blocks=depths[1], stride=self.stride, dilation=1,
            norm=norm, gn_groups=gn_groups, dp_rates=dp_rates[b:b + depths[1]],
        )
        b += depths[1]


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Embedding dimension depends on which stage we pool from
        self.embedding_dim = c2

        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, mlp_hidden_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(mlp_hidden_dims, projection_dim),
        )

    def _make_layer(
        self,
        in_ch: int,
        out_ch: int,
        blocks: int,
        stride: int,
        dilation: int,
        norm: Literal["group", "batch"],
        gn_groups: int,
        dp_rates,
    ) -> nn.Sequential:
        layers = []
        for i in range(blocks):
            s = stride if i == 0 else 1
            layers.append(
                BasicBlock(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    stride=s,
                    dilation=dilation,
                    norm=norm,
                    gn_groups=gn_groups,
                    drop_path=float(dp_rates[i]) if isinstance(dp_rates, (list, tuple)) else float(dp_rates),
                    activation=self.act,
                )
            )
        return nn.Sequential(*layers)

    @staticmethod
    def _unwrap_x(x):
        # Accept dict/list/tuple batches transparently
        if isinstance(x, dict):
            for k in ("image", "images", "img", "x"):
                if k in x:
                    return x[k]
        if isinstance(x, (list, tuple)):
            return x[0]
        return x

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self._unwrap_x(x)

        x = self.stem(x)
        x = self.layer1(x)       # ~150×150
        x = self.inter_layer_maxpool(x)
        x = self.layer2(x)       # ~75×75

        emb = self.avgpool(x).flatten(1)
        proj = self.projection(emb)
        return emb, proj
