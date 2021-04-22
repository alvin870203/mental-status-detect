from .CAER_S import CAER_S
from .CAER import CAER
from .dataset_loader import ImageDataset

__factory = {
    'CAER_S': CAER_S,
    'CAER': CAER
}

def get_names():
    return __factory.keys()

def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
