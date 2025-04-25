import torch.nn as nn
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torch.nn import functional as F
import sys
import copy

# This is not a general multitask model, 
# it is a specific implementation for the EuroSAT and LandCoverAI datasets.
class MultitaskModel(nn.Module):
    def __init__(self, backbone_model='b0', classification_classes=10, segmentation_classes=5):
        super().__init__()

        if backbone_model == 'b0':
            model = efficientnet_b0(weights="DEFAULT")
        elif backbone_model == 'b1':
            model = efficientnet_b1(weights="DEFAULT")
        elif backbone_model == 'b2':
            model = efficientnet_b2(weights="DEFAULT")
        elif backbone_model == 'b3':
            model = efficientnet_b3(weights="DEFAULT")
        elif backbone_model == 'b4':
            model = efficientnet_b4(weights="DEFAULT")
        elif backbone_model == 'b5':
            model = efficientnet_b5(weights="DEFAULT")
        elif backbone_model == 'b6':
            model = efficientnet_b6(weights="DEFAULT")
        elif backbone_model == 'b7':
            model = efficientnet_b7(weights="DEFAULT")
        else:
            raise ValueError(f"Unsupported backbone model: {backbone_model}")
        out_channel = model.features[-1].out_channels
        self.backbone = model.features[:-2]

        segmentation_shared = copy.deepcopy(model.features[-2:])
        classification_shared = copy.deepcopy(model.features[-2:])

        self.heads = nn.ModuleDict({
            "segmentation": 
                nn.Sequential(
                    segmentation_shared,
                    DeepLabHead(out_channel, segmentation_classes)),
            "classification": 
                nn.Sequential(
                    classification_shared,
                    nn.AdaptiveAvgPool2d(1), 
                    nn.Flatten(),  
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(out_channel, 256),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(256, classification_classes))})

    def forward(self, x, task=None):
        input_shape = x.shape[-2:]
        outputs = self.backbone(x)
        if task is not None:
            # If a specific task is provided, return only that task's head output
            shared_feat = self.backbone(x)
            if task == "segmentation":
                x = self.heads[task](shared_feat)
                x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
                return x
            elif task == "classification":
                return self.heads[task](shared_feat)
        shared_feat = self.backbone(x)
        outputs = {
            task: (
                head(shared_feat) if task == "classification"
                else F.interpolate(head(shared_feat), size=input_shape, mode="bilinear", align_corners=False)
            )
            for task, head in self.heads.items()
        }
        return outputs 