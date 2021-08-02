import torch

from discrimnator import Discriminator
from generator import Generator

z = torch.randn(4, 128, 1, 1)

d = Discriminator(1, 128, 3)
g = Generator(1, 128, 3, 128)

image = g(z)


print(image.shape)

score= d(image)
print(score.shape)