import os
from enum import Enum
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image

import wandb


class Mode(Enum):
    TRAIN = "train"
    TEST = "test"


def logits_2_prob(logits: torch.Tensor):
    return torch.mean(torch.sigmoid(logits))


def image_save_to_logdir(
    logdir: str, image: Union[torch.Tensor, np.ndarray], gs
):
    path = os.path.join(logdir, "image", f"{str(gs).zfill(8)}.jpg")
    save_image(image, path)


def build_log_image(
    images: List[torch.Tensor], max_image_count=8
) -> torch.Tensor:
    images = [
        image[:max_image_count] if len(image) > max_image_count else image
        for image in images
    ]
    images = [
        make_grid(image, max_image_count, 2, range=(-1, 1), normalize=True)
        for image in images
    ]
    image = make_grid(images, 1, 2)
    return image


def create_checkpoint(
    D: nn.Module,
    G_f: nn.Module,
    G_g1: nn.Module,
    G_g2: nn.Module,
):
    ckpt = {
        "D": D.state_dict(),
        "G_f": G_f.state_dict(),
        "G_g1": G_g1.state_dict(),
        "G_g2": G_g2.state_dict(),
    }
    return ckpt
