from torchgeo.datasets import LoveDA as _LoveDA
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

class SegmentationTransform:
    def __init__(self, image_only_transform, paired_transform):
        self.image_only_transform = image_only_transform
        self.paired_transform = paired_transform

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']

        image = self.image_only_transform(image)

        image = tv_tensors.Image(image)
        mask = tv_tensors.Mask(mask)

        image, mask = self.paired_transform(image, mask)

        return {'image': image, 'mask': mask}

class LoveDA():
    def __init__(self, batch_size):
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'datasets', 'loveda')

        image_only = v2.Compose([
            v2.ToPureTensor(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        paired_train = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomCrop((512, 512)),
        ])
        paired_val = v2.Compose([
            v2.CenterCrop((512, 512)),
        ])
        data_transforms = {
            'train': SegmentationTransform(image_only, paired_train),
            'val': SegmentationTransform(image_only, paired_val),
        }

        train_dataset = _LoveDA(root=dataset_dir, split='train', download=True, transforms=data_transforms['train'])
        val_dataset = _LoveDA(root=dataset_dir, split='val', download=True, transforms=data_transforms['val'])       
        self.num_classes = 8 

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