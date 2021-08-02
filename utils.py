import torch

from data import celeba_hq, anime_face


def create_dataset(dataset: str, size: int, channels: int = None):
    dataset_dict = {
        "celeba-hq": lambda: celeba_hq(size, channels) if channels else celeba_hq(size),
        "anime-face": lambda: anime_face(size, channels) if channels else anime_face(size),
    }

    return dataset_dict[dataset]()


def shift_image_range(images: torch.Tensor, range_in=(-1, 1), range_out=(0, 1)):
    images = images.clone()
    images.detach_()

    scale = (range_out[1] - range_out[0]) / (range_in[1] - range_in[0])
    bias = range_out[0] - range_in[0] * scale
    images = images * scale + bias
    images.clamp_(min=range_out[0], max=range_out[1])

    return images
