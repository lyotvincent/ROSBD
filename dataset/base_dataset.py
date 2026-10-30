from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """
    """

    def __init__(self, slide_root=None, coord_root=None):
        pass

    @abstractmethod
    def get_single_multichannel(self, sample_id):
        pass

    @abstractmethod
    def get_coord_path(self, sample_id, tile_size=256):
        pass
