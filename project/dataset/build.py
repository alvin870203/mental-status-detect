from torch.utils.data import DataLoader

from .collate_batch import collate_fn
from .datasets import init_dataset, ImageDataset
from .samplers import RandomIdentitySampler
from .transforms import build_transforms


def make_data_loader(cfg):
    train_transforms_f = build_transforms(cfg, is_train=True)
    train_transforms_c = build_transforms(cfg, is_drop=True, is_train=True)
    val_transforms   = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    label_template = dataset.label_template
    num_classes = dataset.num_train_eids

    train_set = ImageDataset(dataset.train, train_transforms_f, train_transforms_c)
    train_loader = DataLoader(
        train_set, batch_size=cfg.DATALOADER.IMS_PER_BATCH,
        sampler=RandomIdentitySampler(dataset.train, cfg.DATALOADER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
        num_workers=num_workers, collate_fn=collate_fn
        )

    test_set = ImageDataset(dataset.test, val_transforms, val_transforms)
    test_loader = DataLoader(
        test_set, batch_size=cfg.DATALOADER.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader, test_loader, num_classes, label_template
