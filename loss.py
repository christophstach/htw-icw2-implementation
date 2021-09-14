import torch
from numpy import histogram, random
from scipy.stats import skewnorm
from torch import Tensor, from_numpy
from torch.nn.functional import softmax


class WGAN:
    def __init__(self) -> None:
        super().__init__()

    def discriminator_loss(self, real_scores: Tensor, fake_scores: Tensor) -> Tensor:
        real_loss = -real_scores
        fake_loss = fake_scores

        loss = real_loss.mean() + fake_loss.mean()

        return loss

    def generator_loss(self, fake_scores: Tensor) -> Tensor:
        fake_loss = -fake_scores
        loss = fake_loss.mean()

        return loss


class RaHinge:
    def __init__(self) -> None:
        super().__init__()

    def discriminator_loss(self, real_scores: Tensor, fake_scores: Tensor) -> Tensor:
        relativistic_real_validity = real_scores - fake_scores.mean()
        relativistic_fake_validity = fake_scores - real_scores.mean()

        real_loss = torch.relu(1.0 - relativistic_real_validity)
        fake_loss = torch.relu(1.0 + relativistic_fake_validity)

        loss = (real_loss.mean() + fake_loss.mean()) / 2

        return loss

    def generator_loss(self, real_scores: Tensor, fake_scores: Tensor) -> Tensor:
        relativistic_real_validity = real_scores - fake_scores.mean()
        relativistic_fake_validity = fake_scores - real_scores.mean()

        real_loss = torch.relu(1.0 - relativistic_fake_validity)
        fake_loss = torch.relu(1.0 + relativistic_real_validity)

        loss = (fake_loss.mean() + real_loss.mean()) / 2

        return loss


class RaLSGAN:
    def __init__(self) -> None:
        super().__init__()

    def discriminator_loss(self, real_scores: Tensor, fake_scores: Tensor) -> Tensor:
        relativistic_real_scores = real_scores - fake_scores.mean()
        relativistic_fake_scores = fake_scores - real_scores.mean()

        real_loss = (relativistic_real_scores - 1.0) ** 2
        fake_loss = (relativistic_fake_scores + 1.0) ** 2

        loss = (fake_loss.mean() + real_loss.mean()) / 2

        return loss.unsqueeze(0)

    def generator_loss(self, real_scores: Tensor, fake_scores: Tensor) -> Tensor:
        relativistic_real_scores = real_scores - fake_scores.mean()
        relativistic_fake_scores = fake_scores - real_scores.mean()

        real_loss = (relativistic_real_scores + 1.0) ** 2
        fake_loss = (relativistic_fake_scores - 1.0) ** 2

        loss = (fake_loss.mean() + real_loss.mean()) / 2

        return loss


def js_div(p, q, reduce=True):
    m = 0.5 * (p + q)
    jsd = 0.5 * (kl_div(p, m, reduce=False) + kl_div(q, m, reduce=False))

    return torch.mean(jsd) if reduce else jsd


def kl_div(p, q, epsilon=1e-12, reduce=True):
    kld = torch.sum(
        p * (p / (q + epsilon)).log(),
        dim=1
    )

    return torch.mean(kld) if reduce else kld


class Realness:
    def __init__(self, score_dim) -> None:
        super().__init__()

        self.score_dim = score_dim
        self.gauss_uniform = True
        self.measure = 'kl'

        if self.measure == 'js':
            self.distance = js_div
        elif self.measure == 'kl':
            self.distance = kl_div
        else:
            raise NotImplementedError()

        if self.gauss_uniform:
            gauss = random.normal(0.0, 0.1, size=1000)
            count, _ = histogram(gauss, self.score_dim)
            self.anchor0 = from_numpy(count / sum(count)).float()

            uniform = random.uniform(-1.0, 1.0, size=1000)
            count, _ = histogram(uniform, self.score_dim)
            self.anchor1 = from_numpy(count / sum(count)).float()
        else:
            skew_left = skewnorm.rvs(-5.0, size=1000)
            count, _ = histogram(skew_left, self.score_dim)
            self.anchor0 = from_numpy(count / sum(count)).float()

            skew_right = skewnorm.rvs(5.0, size=1000)
            count, _ = histogram(skew_right, self.score_dim)
            self.anchor1 = from_numpy(count / sum(count)).float()

    def discriminator_loss(self, real_scores: Tensor, fake_scores: Tensor) -> Tensor:
        self.anchor0 = self.anchor0.to(real_scores)
        self.anchor1 = self.anchor1.to(real_scores)

        real_probabilities = softmax(real_scores, dim=1)
        fake_probabilities = softmax(fake_scores, dim=1)

        loss = self.distance(self.anchor1, real_probabilities) + self.distance(self.anchor0, fake_probabilities)
        # loss -= self.div(self.anchor1, fake_probabilities) + self.div(self.anchor0, real_probabilities)

        return loss

    def generator_loss(self, real_scores: Tensor, fake_scores: Tensor) -> Tensor:
        self.anchor0 = self.anchor0.to(real_scores)
        self.anchor1 = self.anchor1.to(real_scores)

        real_probabilities = softmax(real_scores, dim=1)
        fake_probabilities = softmax(fake_scores, dim=1)

        # No relativism
        # loss = self.distance(self.anchor0, fake_probabilities)

        # EQ19 (default)
        loss = self.distance(real_probabilities, fake_probabilities) - self.distance(self.anchor0, fake_probabilities)

        # EQ20
        # loss = self.distance(self.anchor1, fake_probabilities) - self.distance(self.anchor0, fake_probabilities)

        return loss
