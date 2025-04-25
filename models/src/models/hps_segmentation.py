from torchvision.models import efficientnet_b3
import torch.nn as nn
import torch
#TODO
class HPS_Classification(nn.Module):
    def __init__(self, num_classes_dict):
        """
        Args:
            num_classes_dict (dict): Dictionary where key is task name (str), value is num_classes (int)
        """
        super().__init__()
        
        # Shared backbone (EfficientNet-B3 feature extractor)
        self.backbone = efficientnet_b3(weights='DEFAULT').features
        backbone_output_dim = self.backbone[-1].out_channels 
        
        # Task-specific heads using ModuleDict
        self.heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(backbone_output_dim, 256),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(256, num_classes)
            ) for task, num_classes in num_classes_dict.items()
        })

    def forward(self, x, task=None):
        if task is not None:
            # If a specific task is provided, return only that task's head output
            shared_feat = self.backbone(x)
            return self.heads[task](shared_feat)
        shared_feat = self.backbone(x)  # Shape: (B, C, H, W)
        outputs = {
            task: head(shared_feat) for task, head in self.heads.items()
        }
        return outputs 