import torchvision.transforms as T

from .transforms import RandomErasing
from .transforms import RandomPatch
from .transforms import Cutout

def build_transforms(cfg, is_drop=False, is_train=True, is_face=False):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = [
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.Pad(10),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.ToTensor(),
            normalize_transform
        ]
        if is_drop:
            transform.append(RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN))
        transform = T.Compose(transform)
    else:
        if is_face:
            transform = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TEST),
                T.ToTensor(),
                normalize_transform])
        else:
            transform = T.Compose([
                T.Resize((128, 171)),
                T.CenterCrop(cfg.INPUT.SIZE_TEST),
                T.ToTensor(),
                normalize_transform])
    return transform
