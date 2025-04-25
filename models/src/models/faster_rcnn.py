import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN as Faster_RCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import efficientnet_b0, efficientnet_b3, efficientnet_b4
from collections import OrderedDict
import torch
import sys

class FasterRCNN():
    def __init__(self, num_classes, backbone_model):
        if backbone_model == 'b0':
            backbone = efficientnet_b0(weights="DEFAULT").features
        elif backbone_model == 'b3':
            backbone = efficientnet_b3(weights="DEFAULT").features
        elif backbone_model == 'b4':
            backbone = efficientnet_b4(weights="DEFAULT").features
        else:
            raise ValueError(f"Unsupported backbone model: {backbone_model}")
        
        backbone.out_channels = backbone[-1].out_channels

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )       

        self.model = Faster_RCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )        
