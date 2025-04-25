from src.models.multitask_model import MultitaskModel
from src.datasets.multitask_dataset import MultitaskDataset
import torch

def load_mt_model(checkpoint_path=None):
    model = MultitaskModel('b0')
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
    return model

def load_dataloaders(batch_size=32):
    dataloaders = MultitaskDataset(batch_size).data_loaders
    return dataloaders