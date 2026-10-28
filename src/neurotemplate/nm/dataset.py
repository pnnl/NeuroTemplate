"""
DictDataset: a dict of equal-length tensors, batched with its split name.

From neuromancer/dataset.py; the rest of that module (Lightning data module,
sequence/static/graph datasets, file readers) is not used.
"""
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class DictDataset(Dataset):
    """
    Basic dataset compatible with the Trainer
    """

    def __init__(self, datadict, name='train'):
        """

        :rtype: object
        :param datadict: (dict {str: Tensor})
        :param name: (str) Name of dataset
        """
        super().__init__()
        self.datadict = datadict
        lens = [v.shape[0] for v in datadict.values()]
        assert len(set(lens)) == 1, 'Mismatched number of samples in dataset tensors'
        self.length = lens[0]
        self.name = name

    def __getitem__(self, i):
        """Fetch a single item from the dataset."""
        return {k: v[i] for k, v in self.datadict.items()}

    def __len__(self):
        return self.length

    def collate_fn(self, batch):
        """Wraps the default PyTorch batch collation function and adds a name field.

        :param batch: (dict str: torch.Tensor) dataset sample.
        """
        batch = default_collate(batch)
        batch['name'] = self.name
        return batch
