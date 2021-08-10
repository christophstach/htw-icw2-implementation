import torch
from determined.pytorch import PyTorchTrialContext
from torch import Tensor, autograd, nn


class GradientPenalty:
    def __init__(self, context: PyTorchTrialContext, discriminator: nn.Module) -> None:
        super().__init__()

        self.context = context
        self.discriminator = discriminator

        self.coefficient = 100.0  # Lambda
        self.center = 0.0
        # self.coefficient = 10.0
        # self.center = 1.0

    def interpolate_images(self, real_images: Tensor, fake_images: Tensor):
        alpha = self.context.to_device(torch.rand(real_images.shape[0], 1, 1, 1))
        images = real_images + (1 - alpha) * fake_images
        images.requires_grad_(True)

        return images

    def mix_images(self, real_images: Tensor, fake_images: Tensor):
        half_batch_size = real_images.shape[0] // 2
        images = torch.cat((real_images[:half_batch_size], fake_images[:half_batch_size]))
        images.requires_grad_(True)

        return images

    def __call__(self, real_images: Tensor, fake_images: Tensor):
        # images = self.interpolate_images(real_images, fake_images)
        images = self.mix_images(real_images, fake_images)
        scores = self.discriminator(images)

        ones = self.context.to_device(torch.ones_like(scores))
        gradients = autograd.grad(outputs=scores, inputs=images, grad_outputs=ones, create_graph=True)[0]
        penalties = (gradients.norm(2, dim=1) - self.center) ** 2
        gradient_penalty = self.coefficient * penalties.mean()

        return gradient_penalty
