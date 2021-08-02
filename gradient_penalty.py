import torch
from determined.pytorch import PyTorchTrialContext
from torch import Tensor, autograd, nn


class GradientPenalty:
    def __init__(self, context: PyTorchTrialContext, discriminator: nn.Module) -> None:
        super().__init__()

        self.context = context
        self.discriminator = discriminator

    def __call__(self, real_images: Tensor, fake_images: Tensor):
        batch_size = real_images.shape[0]

        alpha = self.context.to_device(torch.rand(batch_size, 1, 1, 1))
        interpolated_images = real_images + (1 - alpha) * fake_images
        interpolated_images.requires_grad_(True)

        scores = self.discriminator(interpolated_images)

        ones = self.context.to_device(torch.ones_like(scores))
        gradients = autograd.grad(outputs=scores, inputs=interpolated_images, grad_outputs=ones, create_graph=True)[0]
        penalties = (gradients.norm(2, dim=1) - 1.0) ** 2
        gradient_penalty = 10.0 * penalties.mean()

        return gradient_penalty
