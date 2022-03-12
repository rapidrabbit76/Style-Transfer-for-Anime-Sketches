from joblib import parallel_backend
from torchvision.models import vgg19
import torch
import torch.nn as nn


def VGG19() -> nn.Module:
    model = vgg19(pretrained=True)
    head = list(model.classifier.children())[0]  # get vgg19 fc1
    model.classifier = head
    model = model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model
