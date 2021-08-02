import os

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.folder import is_image_file


class FlatImageFolder(Dataset):
    def __init__(self, root, transform=None, loader=default_loader):
        self.root = root + "/"
        self.transform = transform
        self.loader = loader

        self.files = sorted([file for file in os.listdir(self.root) if is_image_file(file)])

    def __getitem__(self, index: int):
        path = self.root + self.files[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, 0

    def __len__(self):
        return len(self.files)


def celeba_hq(size=128, channels=3, root="/datasets"):
    assert channels == 1 or channels == 3

    available_sizes = [128, 256, 512, 1024]
    needs_resize = False

    if size not in available_sizes:
        needs_resize = True
        nearest_size = min(filter(lambda s: s > size, available_sizes))
        root += "/celebAHQ/data" + str(nearest_size) + "x" + str(nearest_size)
    else:
        root += "/celebAHQ/data" + str(size) + "x" + str(size)

    transform_ops = [
        transforms.Resize(size) if needs_resize else None,
        transforms.Grayscale() if channels == 1 else None,
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if channels == 1
        else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    transform = transforms.Compose([op for op in transform_ops if op is not None])

    return FlatImageFolder(
        root,
        transform=transform
    )


def anime_face(size=128, channels=3, root="/datasets"):
    assert channels == 1 or channels == 3

    root += "/anime-face/cropped"

    transform_ops = [
        transforms.Resize(size),
        transforms.RandomCrop(size),
        None if channels == 3 else transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if channels == 1
        else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    transform = transforms.Compose([op for op in transform_ops if op is not None])

    return FlatImageFolder(
        root,
        transform=transform
    )

