from .landcover_ai import LandCoverAI
from .eurosat import EuroSAT

class MultitaskDataset():
    def __init__(self, batch_size):
        self.datasets = {
            'landcoverai': LandCoverAI(batch_size),
            'eurosat': EuroSAT(batch_size)
        }
        self.data_loaders = {
            'classification_train': self.datasets['eurosat'].train_loader,
            'classification_val': self.datasets['eurosat'].valid_loader,
            'segmentation_train': self.datasets['landcoverai'].train_loader,
            'segmentation_val': self.datasets['landcoverai'].valid_loader
        }