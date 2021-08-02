from discrimnator import Discriminator
from generator import Generator

latent_dimension = 128
image_size = 64
image_channels = 3

generator = Generator(1, image_channels, latent_dimension)
discriminator = Discriminator(1, image_channels)

print(generator)
print(discriminator)