import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary
from glob import glob
import numpy as np
import torch


class FaceRecog(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.resnet34 = models.resnet34(True)
        self.features = nn.Sequential(*list(self.resnet34.children())[:-1])
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(self.resnet34.fc.in_features, num_classes))

    def forward(self, x):
        x = self.features(x)
        y = self.classifier(x)
        return y

    def summary(self, input_size):
        return summary(self, input_size)


def Model():
    model = FaceRecog(num_classes=6)
    model.load_state_dict(torch.load("./model/demo.pth"))
    model.eval()
    return model
