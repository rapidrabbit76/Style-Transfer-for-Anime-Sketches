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


class Dataset(data.Dataset):
    def __init__(
        self,
        line_paths: List[str],
        color_paths: List[str],
        transforms: Callable,
    ) -> None:
        super().__init__()
        self.__total_len = len(line_paths)
        self.samples = list(zip(line_paths, color_paths))
        self.transforms = transforms

    def __len__(self):
        return self.__total_len

    @staticmethod
    def _loadder(path: str, code: int):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, code)
        return image

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        line_path, color_path = self.samples[index]
        line = self._loadder(line_path, cv2.COLOR_BGR2GRAY)
        color = self._loadder(color_path, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(np.copy(color), cv2.COLOR_BGR2GRAY)

        if self.transforms is not None:
            line, gray, color = self.transforms(line, gray, color)
        return line, gray, color


def build_dataloader(
    args,
    transforms: Callable,
    mode: Mode,
):
    line_paths = sorted(
        glob(os.path.join(args.root_dir, mode.value, "line", "*"))
    )
    color_paths = sorted(
        glob(os.path.join(args.root_dir, mode.value, "image", "*"))
    )
    assert len(line_paths) != 0
    assert len(color_paths) != 0
    assert len(line_paths) == len(color_paths)
    dataset = Dataset(line_paths, color_paths, transforms)
    return data.DataLoader(
        dataset,
        batch_size=args.batch_size if mode is Mode.TRAIN else 8,
        shuffle=True if mode is Mode.TRAIN else False,
        num_workers=args.num_workers,
    )
