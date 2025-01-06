import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def forward(self, inputs, targets):
        smooth = 1.
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice