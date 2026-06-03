import numpy as np
from ..autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any



class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))
            
        self._iterable = None

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_dict_indices = indices[i : i + self.batch_size]
            # 获取数据并转换为符合要求的 Tensor 格式
            batch_samples = [self.dataset[idx] for idx in batch_dict_indices]
            
            return_data = []
            for j in range(len(batch_samples[0])):
                # 把样本汇聚成大数组
                data_batch = np.stack([sample[j] for sample in batch_samples])
                return_data.append(Tensor(data_batch))
            
            yield tuple(return_data)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __repr__(self):
        return f"DataLoader({self.dataset}, batch_size={self.batch_size}, shuffle={self.shuffle})"

    def __str__(self):
        return f"DataLoader({self.dataset}, batch_size={self.batch_size}, shuffle={self.shuffle})"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self

    def __next__(self):
        if self._iterable is None:
            self._iterable = iter(self)
        
        try:
            return next(self._iterable)
        except StopIteration:
            self._iterable = None
            raise StopIteration