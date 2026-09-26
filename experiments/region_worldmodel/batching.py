"""Group independent recordings under identical sensing conditions."""
import math
import torch
from torch.utils.data import Sampler


class SharedSensingBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, seed=0):
        if batch_size < 2 or len(dataset.entries) < 2:
            raise ValueError('Shared sensing requires >=2 recordings and batch_size >=2')
        self.dataset, self.batch_size, self.seed = dataset, batch_size, seed

    def _chunks(self, indices):
        chunks = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.dataset.training:
            return [chunk for chunk in chunks if len(chunk) == self.batch_size]
        # Keep every validation recording exactly once per action. Merge a
        # singleton remainder rather than estimating regularization from B=1.
        if len(chunks) > 1 and len(chunks[-1]) == 1:
            chunks[-2].extend(chunks.pop())
        return chunks

    def __iter__(self):
        ds = self.dataset
        generator = torch.Generator().manual_seed(self.seed)
        if ds.training:
            indices = torch.randperm(len(ds.entries), generator=generator).tolist()
            for batch in self._chunks(indices):
                layout_seed = int(torch.randint(0, 2**31, (), generator=generator))
                action = int(torch.randint(len(ds.actions), (), generator=generator))
                yield [(index, layout_seed, action) for index in batch]
        else:
            # Fixed layouts and membership across epochs; action boundaries
            # never mix. Use the same source layout for every action.
            for action in range(len(ds.actions)):
                for group, batch in enumerate(self._chunks(list(range(len(ds.entries))))):
                    yield [(index, group, action) for index in batch]

    def __len__(self):
        n = len(self.dataset.entries)
        if self.dataset.training:
            return n // self.batch_size
        count = math.ceil(n / self.batch_size)
        if count > 1 and n % self.batch_size == 1:
            count -= 1
        return count * len(self.dataset.actions)
