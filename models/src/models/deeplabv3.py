from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.deeplabv3 import DeepLabV3 as DeepLab_V3

class DeepLabV3():
    def __init__(self, num_classes, backbone_model):
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
        return_layers = {"features": "out"}
        backbone = IntermediateLayerGetter(model, return_layers=return_layers)
        aux_classifier = None
        classifier = DeepLabHead(out_channel, num_classes)
        self.model = DeepLab_V3(backbone, classifier, aux_classifier)