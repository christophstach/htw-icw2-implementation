from typing import Union, Dict, Any

import torch
from determined.pytorch import PyTorchTrial, PyTorchTrialContext, DataLoader, TorchData
from determined.tensorboard.metric_writers.pytorch import TorchWriter
from torch import Tensor
from torch.optim import Adam
from torchvision.utils import make_grid

from classifier_score import ClassifierScore
from create_evaluator import create_evaluator
from discrimnator import Discriminator
from ema import ExponentialMovingAverage
from generator import Generator
from gradient_penalty import GradientPenalty
from loss import WGAN
from utils import shift_image_range, create_dataset


class GANTrail(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        super().__init__(context)

        self.context = context
        self.logger = TorchWriter()
        self.num_log_images = 25

        self.dataset = self.context.get_hparam('dataset')
        self.image_size = self.context.get_hparam('image_size')
        self.image_channels = self.context.get_hparam('image_channels')
        self.latent_dimension = self.context.get_hparam('latent_dimension')
        self.spectral_norm = self.context.get_hparam('spectral_norm')
        self.ema = self.context.get_hparam('ema')

        self.g_depth = self.context.get_hparam('g_depth')
        self.d_depth = self.context.get_hparam('d_depth')

        self.g_lr = self.context.get_hparam('g_lr')
        self.g_b1 = self.context.get_hparam('g_b1')
        self.g_b2 = self.context.get_hparam('g_b2')

        self.d_lr = self.context.get_hparam('d_lr')
        self.d_b1 = self.context.get_hparam('d_b1')
        self.d_b2 = self.context.get_hparam('d_b2')

        self.generator = Generator(self.g_depth, self.image_size, self.image_channels, self.latent_dimension)
        self.discriminator = Discriminator(self.d_depth, self.image_size, self.image_channels, self.spectral_norm)
        self.evaluator, resize_to, num_classes = create_evaluator('vggface2')
        self.evaluator.eval()

        if self.ema:
            self.generator = ExponentialMovingAverage(self.generator)
        else:
            self.generator = self.generator

        self.opt_g = Adam(self.generator.parameters(), self.g_lr, (self.g_b1, self.g_b2))
        self.opt_d = Adam(self.discriminator.parameters(), self.d_lr, (self.d_b1, self.d_b2))

        self.generator = self.context.wrap_model(self.generator)
        self.discriminator = self.context.wrap_model(self.discriminator)
        self.evaluator = self.context.wrap_model(self.evaluator)

        self.opt_g = self.context.wrap_optimizer(self.opt_g)
        self.opt_d = self.context.wrap_optimizer(self.opt_d)

        self.loss = WGAN()
        self.gradient_penalty = GradientPenalty(self.context, self.discriminator)

        self.fixed_z = torch.randn(self.num_log_images, self.latent_dimension, 1, 1)

        self.classifier_score = ClassifierScore(
            classifier=self.evaluator,
            resize_to=resize_to
        )

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Union[Tensor, Dict[str, Any]]:
        real_images, _ = batch
        batch_size = real_images.shape[0]

        # optimize discriminator
        self.discriminator.zero_grad()
        z = torch.randn(batch_size, self.latent_dimension, 1, 1)
        z = self.context.to_device(z)

        with torch.no_grad():
            fake_images = self.generator(z)

        real_scores = self.discriminator(real_images)
        fake_scores = self.discriminator(fake_images)

        d_loss = self.loss.discriminator_loss(real_scores, fake_scores)
        gp = self.gradient_penalty(real_images, fake_images)
        d_total_loss = d_loss + gp

        self.context.backward(d_total_loss)
        self.context.step_optimizer(self.opt_d)

        # optimize generator
        self.generator.zero_grad()
        z = torch.randn(batch_size, self.latent_dimension, 1, 1)
        z = self.context.to_device(z)

        fake_images = self.generator(z)
        fake_scores = self.discriminator(fake_images)
        g_loss = self.loss.generator_loss(fake_scores)

        self.context.backward(g_loss)
        self.context.step_optimizer(self.opt_g)
        if self.ema:
            self.generator.update()

        self.generator.eval()
        z = torch.randn(batch_size, self.latent_dimension, 1, 1)
        z = self.context.to_device(z)
        fake_images = self.generator(z)
        classifier_score = self.classifier_score(fake_images)
        self.generator.train()

        return {
            'd_total_loss': d_total_loss,
            'd_loss': d_loss,
            'g_loss': g_loss,
            'abs_d_loss': torch.abs(d_loss),
            'gp': gp,
            'classifier_score': classifier_score
        }

    def evaluate_batch(self, batch: TorchData, batch_idx: int) -> Dict[str, Any]:
        real_images, _ = batch
        batch_size = real_images.shape[0]

        self.generator.eval()
        self.discriminator.eval()

        z = torch.randn(batch_size, self.latent_dimension, 1, 1)
        z = self.context.to_device(z)

        fake_images = self.generator(z)

        real_scores = self.discriminator(real_images)
        fake_scores = self.discriminator(fake_images)

        abs_d_loss = torch.abs(self.loss.discriminator_loss(real_scores, fake_scores))
        classifier_score = self.classifier_score(fake_images)

        if batch_idx == 0:
            # log sample images
            z = torch.randn(self.num_log_images, self.latent_dimension, 1, 1)
            z = self.context.to_device(z)

            sample_images = self.generator(z)
            sample_images = shift_image_range(sample_images)
            sample_grid = make_grid(sample_images, nrow=5)

            self.logger.writer.add_image(f'generated_sample_images', sample_grid, self.context.current_train_batch())

            # log fixed images
            z = self.context.to_device(self.fixed_z)

            fixed_images = self.generator(z)
            fixed_images = shift_image_range(fixed_images)
            fixed_grid = make_grid(fixed_images, nrow=5)

            self.logger.writer.add_image(f'generated_fixed_images', fixed_grid, self.context.current_train_batch())

        self.generator.train()
        self.discriminator.train()

        return {
            'val_abs_d_loss': abs_d_loss,
            'val_classifier_score': classifier_score
        }

    def build_training_data_loader(self) -> DataLoader:
        train_data = create_dataset(dataset=self.dataset, size=self.image_size, channels=self.image_channels)

        return DataLoader(
            train_data,
            batch_size=self.context.get_per_slot_batch_size(),
            drop_last=True
        )

    def build_validation_data_loader(self) -> DataLoader:
        validation_data = create_dataset(dataset=self.dataset, size=self.image_size, channels=self.image_channels)

        return DataLoader(
            validation_data,
            batch_size=self.context.get_per_slot_batch_size(),
            drop_last=True
        )
