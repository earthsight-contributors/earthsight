from .eurosat import EuroSAT
from .patternnet import PatternNet
from .advance import Advance

class Hps_Classification_Dataset():
    def __init__(self, batch_size):
        self.eurosat = EuroSAT(batch_size)
        self.advance = Advance(batch_size)
        self.patternnet = PatternNet(batch_size)