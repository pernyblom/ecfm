"""Persistent data workers with epoch-aware crop seeds, including Windows spawn."""
import torch
from torch.utils.data import DataLoader, Sampler


class EpochSampler(Sampler):
    def __init__(self, dataset, shuffle, seed):
        self.dataset, self.shuffle, self.seed = dataset, shuffle, seed

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        epoch = self.dataset.epoch
        order = (torch.randperm(len(self), generator=torch.Generator().manual_seed(self.seed+epoch)).tolist()
                 if self.shuffle else range(len(self)))
        return iter((index, epoch) for index in order)


def make_loader(dataset, batch_size, workers=0, shuffle=False, seed=0):
    settings = getattr(dataset, 'cfg', {}).get('train', {})
    sampler = EpochSampler(dataset, shuffle, seed) if getattr(dataset, 'training', False) else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
        shuffle=shuffle if sampler is None else False, num_workers=workers,
        persistent_workers=workers > 0 and settings.get('persistent_workers', True),
        pin_memory=settings.get('pin_memory', str(settings.get('device', 'cpu')).startswith('cuda')),
        generator=torch.Generator().manual_seed(seed))


def to_device(value, device):
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
