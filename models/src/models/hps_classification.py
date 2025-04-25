from torchvision.models import efficientnet_b3
import torch.nn as nn
import torch
    
class HPS_Classification(nn.Module):
    def __init__(self, num_classes_dict):
        super().__init__()
        
        self.backbone = efficientnet_b3(weights='DEFAULT').features
        backbone_output_dim = self.backbone[-1].out_channels 
        
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
            shared_feat = self.backbone(x)
            return self.heads[task](shared_feat)
        shared_feat = self.backbone(x) 
        outputs = {
            task: head(shared_feat) for task, head in self.heads.items()
        }
        return outputs 