import numpy as np
import pytest
import torch
from dataset import Transforms
from easydict import EasyDict
from models import VGG19, Discriminator, GuideDecoder, UnetGenerator


@pytest.fixture(scope="session")
def args():
    return EasyDict(
        {
            "image_size": 256,
            "batch_size": 2,
        }
    )


@pytest.fixture(scope="session")
def unet_gen():
    return UnetGenerator(1, 3)


@pytest.fixture(scope="session")
def disc():
    return Discriminator(3)


@pytest.fixture(scope="session")
def guide_decoder_1():
    return GuideDecoder(256, 1)


@pytest.fixture(scope="session")
def guide_decoder_2():
    return GuideDecoder(512, 3)


@pytest.fixture(scope="session")
def vgg():
    return VGG19()


def build_tensor(args, channels):
    size = args.image_size
    b = args.batch_size
    return torch.zeros([b, channels, size, size])


@pytest.fixture(scope="session")
def input_line(args):
    return build_tensor(args, 1)


@pytest.fixture(scope="session")
def output_color(args):
    return build_tensor(args, 3)


@pytest.fixture(scope="session")
def train_batch(input_line, output_color):
    return (input_line, input_line, output_color)


def build_image(size, channels):
    shape = [size, size]
    if channels > 1:
        shape.append(channels)
    return np.zeros(shape, dtype=np.uint8)


@pytest.fixture(scope="session")
def image_batch(args):
    size = args.image_size
    line = build_image(size, 1)
    gray = build_image(size, 1)
    color = build_image(size, 3)
    return line, gray, color


@pytest.fixture(scope="session", params=[True, False])
def transforms(request, args):
    return Transforms(args.image_size, request.param)
