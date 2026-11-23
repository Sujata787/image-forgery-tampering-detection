from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_DIR / "artifacts"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_resnet18.pth"

NUM_CLASSES = 2


class ForgeryResNet18(nn.Module):

    def __init__(self, pretrained=True):

        super().__init__()

        weights = (
            models.ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = models.resnet18(weights=weights)

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        input_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(
            input_features,
            NUM_CLASSES
        )

        for parameter in self.backbone.fc.parameters():
            parameter.requires_grad = True

    def forward(self, x):

        return self.backbone(x)


def build_model(pretrained=True):

    return ForgeryResNet18(pretrained=pretrained)


if __name__ == "__main__":

    model = build_model(pretrained=False)

    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():

        output = model(dummy_input)

    print("Input shape :", dummy_input.shape)

    print("Output shape:", output.shape)

    print("Model test completed.")