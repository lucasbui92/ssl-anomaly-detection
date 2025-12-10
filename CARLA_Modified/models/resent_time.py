# Adapted from: CARLA (https://github.com/zamanzadeh/CARLA/tree/main).
# Modifications by Thuan Anh Bui, 2025.
# Changes: Mainly debugging to verify feature domains

import torch
from torch import nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.layers(x)


class ResNetBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ) -> None:
        super().__init__()

        channels = [in_channels, out_channels, out_channels, out_channels]
        kernel_sizes = [8, 5, 3]

        self.layers = nn.Sequential(*[
            ConvBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=kernel_sizes[i],
                stride=1
            ) for i in range(len(kernel_sizes))
        ])

        self.match_channels = False
        if in_channels != out_channels:
            self.match_channels = True
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1
                ),
                nn.BatchNorm1d(num_features=out_channels)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.match_channels:
            return self.layers(x) + self.residual(x)
        return self.layers(x)


class ResNetRepresentation(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int = 4, feature_idx=None) -> None:
        super().__init__()
        self.input_args = {
            'in_channels': in_channels,
            'mid_channels': mid_channels,
            'feature_idx': feature_idx,
        }

        # store indices as buffer so they move with the model device
        if feature_idx is not None:
            self.register_buffer(
                "feature_idx",
                torch.tensor(feature_idx, dtype=torch.long)
            )
        else:
            self.feature_idx = None

        self.layers = nn.Sequential(
            ResNetBlock(in_channels=in_channels, out_channels=mid_channels),
            # ResNetBlock(in_channels=mid_channels, out_channels=mid_channels)
            # ResNetBlock(in_channels=mid_channels, out_channels=mid_channels * 2),
            # ResNetBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 2),
        )

    def forward(self, x: torch.Tensor):
        # x: (B, C_total, T)
        if self.feature_idx is not None:
            x = x[:, self.feature_idx, :]  # select subset (B, in_channels, T)

        z = self.layers(x)               # convs see only selected channels
        z = z.mean(dim=-1)               # global average pooling over time
        return z

def resnet_ts(**kwargs):
    # kwargs must include 'in_channels', 'mid_channels', and optionally 'feature_idx'
    return {
        'backbone': ResNetRepresentation(**kwargs),
        'dim': kwargs['mid_channels'],
    }
