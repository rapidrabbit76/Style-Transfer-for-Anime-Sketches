import os
from glob import glob
from typing import Callable, List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2 as ToTensor
from torch.utils import data
from utils import Mode


class Transforms:
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    def __init__(
        self,
        image_size: int,
        train: bool = True,
        mean: List[float] = None,
        std: List[float] = None,
    ) -> None:
        mean = self.mean if mean is None else mean
        std = self.std if std is None else std

        self.transforms = A.Compose(
            [
                A.Resize(image_size, image_size, cv2.INTER_AREA),
                A.HorizontalFlip(p=1.0 if train else 0.0),
            ],
            additional_targets={
                "image0": "image",
                "image1": "image",
            },
        )
        # TODO: color rotation affine transformation
        self.color_normalize = A.Compose(
            [
                A.Normalize(
                    mean=mean,
                    std=std,
                    max_pixel_value=255.0,
                ),
                ToTensor(),
            ]
        )
        self.gray_normalize = A.Compose(
            [
                A.Normalize(
                    mean=mean[:1],
                    std=std[:1],
                    max_pixel_value=255.0,
                ),
                ToTensor(),
            ],
            additional_targets={"image0": "image"},
        )

    def __call__(
        self, line: np.ndarray, gray: np.ndarray, color: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        augmentations = self.transforms(image=line, image0=gray, image1=color)
        line = augmentations["image"]
        gray = augmentations["image0"]
        color = augmentations["image1"]

        augmentations = self.gray_normalize(image=line, image0=gray)
        line = augmentations["image"]
        gray = augmentations["image0"]
        color = self.color_normalize(image=color)["image"]
        return line, gray, color

