import torch

from torch import nn
import torch.nn.functional as F


# https://github.com/torchgan/torchgan/blob/master/torchgan/metrics/classifierscore.py
class ClassifierScore:
    def __init__(self, classifier: nn.Module, resize_to: int) -> None:
        super().__init__()

        self.classifier = classifier
        self.resize_to = resize_to

    def resize(self, x):
        return F.interpolate(
            input=x,
            size=(self.resize_to, self.resize_to),
            mode='bilinear',
            align_corners=False
        )

    def __call__(self, x):
        x = self.resize(x) if x.shape[2] != self.resize_to or x.shape[3] != self.resize_to else x
        x = self.classifier(x)

        p = F.softmax(x, dim=1)
        q = torch.mean(p, dim=0)

        kl_div = torch.sum(p * (F.log_softmax(x, dim=1) - torch.log(q)), dim=1)
        return torch.exp(torch.mean(kl_div)).item()
