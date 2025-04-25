from src.image import Image
metr = None

class Metrics(object):
    """
    Class that holds the metrics for the simulation
    """
    metr = None

    def metr():
        """
        Returns the metrics object
        """
        global metr
        if not metr:
            metr = Metrics()
        return metr

    def __init__(self) -> None:
        self.images_captured = 0
        self.pri_captured = 0
        self.hipri_captured = 0
        self.hipri_computed = 0
        self.hipri_sent = 0
        self.cmpt_delay = [0,1E-32]
        self.transmit_delay = [0,1E-32]


    def print(self) -> None:
        """
        Prints the metrics
        """
        print("Images Captured: ", self.images_captured)
        print("Compute requiring images Captured: ", self.hipri_captured)
        print("Compute requiring image is hi pri: ", self.hipri_computed)
        print("High Priority Images Sent: ", self.hipri_sent)


