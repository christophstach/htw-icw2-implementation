import math

from torch import nn
from torch.nn.utils import spectral_norm as sn


class Discriminator(nn.Module):
    def __init__(self, d_depth: int, image_size: int, image_channels: int, spectral_norm: bool = False) -> None:
        super().__init__()

        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (4, 4), (2, 2), (1, 1), bias=False),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(out_channels)
            )

        def block_sn(in_channels, out_channels):
            return nn.Sequential(
                sn(nn.Conv2d(in_channels, out_channels, (4, 4), (2, 2), (1, 1), bias=False)),
                nn.LeakyReLU(0.2)
            )

        def weights_init(m):
            class_name = m.__class__.__name__

            if class_name.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif class_name.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        self.blocks = nn.ModuleList()

        if spectral_norm:
            self.blocks.append(block_sn(image_channels, d_depth))
        else:
            self.blocks.append(block(image_channels, d_depth))

        multiplier = 1
        for i in range(int(math.log2(image_size) - 3)):
            multiplier = 2 ** i
            if spectral_norm:
                self.blocks.append(block_sn(multiplier * d_depth, 2 * multiplier * d_depth))
            else:
                self.blocks.append(block(multiplier * d_depth, 2 * multiplier * d_depth))

        multiplier *= 2

        if spectral_norm:
            self.blocks.append(sn(nn.Conv2d(multiplier * d_depth, 1, (4, 4), (1, 1), (0, 0), bias=False)))
        else:
            self.blocks.append(nn.Conv2d(multiplier * d_depth, 1, (4, 4), (1, 1), (0, 0), bias=False))

        self.apply(weights_init)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)

        return x
