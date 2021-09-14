import math

from torch import nn


class Generator(nn.Module):
    def __init__(self, g_depth: int, image_size: int, image_channels: int, latent_dim: int) -> None:
        super().__init__()

        def block_bn(in_channels, out_channels, size):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, (4, 4), (2, 2), (1, 1), bias=False),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            )

        def weights_init(m):
            class_name = m.__class__.__name__

            if class_name.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif class_name.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        multiplier = 2 ** (int(math.log2(image_size)) - 3)
        size = 4

        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Sequential(
            nn.ConvTranspose2d(latent_dim, multiplier * g_depth, (4, 4), (1, 1), (0, 0), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(multiplier * g_depth)
        ))

        for i in reversed(range(int(math.log2(image_size) - 3))):
            multiplier = 2 ** i
            size *= 2

            self.blocks.append(block_bn(2 * multiplier * g_depth, multiplier * g_depth, size))

        self.blocks.append(nn.Sequential(
            nn.ConvTranspose2d(g_depth, image_channels, (4, 4), (2, 2), (1, 1), bias=False),
            nn.Tanh()
        ))

        self.apply(weights_init)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)

        return x
