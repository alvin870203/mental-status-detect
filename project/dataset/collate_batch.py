import torch


def collate_fn(batch):
    c_imgs, f_imgs, eids = zip(*batch)
    eids = torch.tensor(eids, dtype=torch.int64)
    return torch.stack(c_imgs, dim=0), torch.stack(f_imgs, dim=0), eids