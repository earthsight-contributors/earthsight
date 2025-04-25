import os
import sys
import torch
from torchgeo.datasets import ADVANCE as _Advance
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, Subset
from torchvision.transforms import v2
from torchvision.tv_tensors import Image as TVImage
from torchvision.io import read_image

def tv_tensor_loader(path):
    img = read_image(path)  
    return TVImage(img)    

class TransformWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.subset[idx]
        return self.transform(sample['image']), sample['label']

    def __len__(self):
        return len(self.subset)

class Advance():
    def __init__(self, batch_size):
        full_dataset = _Advance(root=os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'datasets'), download=True)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])    
        data_transforms = {
            'train': v2.Compose([
                v2.ToImage(),
                v2.RandomResizedCrop(224, antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             ]),
            'val': v2.Compose([
                v2.ToImage(),
                v2.Resize(256, antialias=True),
                v2.CenterCrop(224),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
             ])
        }

        train_dataset = TransformWrapper(train_dataset, data_transforms['train'])
        val_dataset = TransformWrapper(val_dataset, data_transforms['val'])         

        self.num_classes = len(full_dataset.classes)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size = batch_size, 
            shuffle=True, 
            num_workers = 8, 
            drop_last=True)
        self.valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size= batch_size,
            shuffle=False,
            num_workers=8)
