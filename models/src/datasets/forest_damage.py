from torchgeo.datasets import ForestDamage as Forest_Damage
from torch.utils.data import random_split
from torchvision.transforms import v2
import src.detection.utils as utils
import torch
import os
import sys
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

class TransformWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.subset[idx]
        img = sample['image']
        boxes = sample['boxes']
        labels = sample['label']
        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        img = tv_tensors.Image(img)
        
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd     

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.subset)

class ForestDamage():
    def __init__(self, batch_size):
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'datasets')
        full_dataset = [data for data in Forest_Damage(root=dataset_dir, download=False)
                        if data['boxes'].shape[-1] == 4]
        for data in full_dataset:
            valid = (data["boxes"][:, 2] - data["boxes"][:, 0]) != 0
            data["boxes"], data["label"] = data["boxes"][valid], data["label"][valid]
            data["label"][data["label"] == 0] = 4
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])    
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

        train_dataset = TransformWrapper(train_dataset, data_transforms['train'])
        val_dataset = TransformWrapper(val_dataset, data_transforms['val'])   
        # self.num_classes = len(full_dataset.classes)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size = batch_size, 
            shuffle=True, 
            num_workers = 8,
            collate_fn=utils.collate_fn, 
            drop_last=True)
        
        self.valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size= batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=utils.collate_fn)
        
        self.num_classes = 5