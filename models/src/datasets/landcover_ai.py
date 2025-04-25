from torchgeo.datasets import LandCoverAI as LandCover_AI
from torch.utils.data import random_split
from torchvision.transforms import v2
import src.detection.utils as utils
import torch
import os
import sys
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torch.utils.data import default_collate

class LandCoverAI():
    def __init__(self, batch_size):
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'datasets', 'landcoverai')
        data_transforms = {
            'train': v2.Compose([
                v2.ToPureTensor(),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
            ]),
            'val': v2.Compose([
                v2.ToPureTensor(),
                v2.ToDtype(torch.float32, scale=True),
            ])
        }
        train_dataset = LandCover_AI(root=dataset_dir, split='train', download=True, transforms=data_transforms['train'])
        val_dataset = LandCover_AI(root=dataset_dir, split='val', download=True, transforms=data_transforms['val'])       
        self.num_classes = 5

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size = batch_size, 
            shuffle=True, 
            num_workers = 8,
            collate_fn=default_collate,
            drop_last=True)
        
        self.valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size = batch_size,
            shuffle=False,
            collate_fn=default_collate,
            num_workers= 8)