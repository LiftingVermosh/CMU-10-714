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
        # __init__ 应该只做简单的赋值
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        # 移除这里所有的 self.ordering, self.indices 的计算

    def __iter__(self):
        # 所有的逻辑都应该在这里
        # 第一步：根据 shuffle 标志生成索引顺序
        if self.shuffle:
            indices = np.random.permutation(len(self.dataset))
        else:
            indices = np.arange(len(self.dataset))

        # 第二步：将索引序列分割成批次
        if self.batch_size is None:
            # 如果 batch_size 是 None，则整个数据集是一个批次
            self.ordering = [indices]
        else:
            # 使用 np.array_split 来处理不能整除的情况
            # 注意 np.array_split 的第二个参数是分割点的数量或分割点的位置
            # 为了按 batch_size 分割，我们需要计算分割点的索引
            num_samples = len(self.dataset)
            split_points = range(self.batch_size, num_samples, self.batch_size)
            self.ordering = np.array_split(indices, split_points)

        # 第三步：初始化当前批次的计数器
        self.index = 0

        # 第四步：返回迭代器自身
        return self

    def __next__(self):
        # __next__ 方法的逻辑是正确的，无需修改
        if self.index >= len(self.ordering):
            raise StopIteration
        
        indices = self.ordering[self.index]
        batch_samples = [self.dataset[i] for i in indices]

        self.index += 1

        # Collate (整理) 数据的部分也是正确的
        if isinstance(batch_samples[0], tuple):
            collated_columns = zip(*batch_samples)
            return tuple([Tensor(np.stack(column)) for column in collated_columns])
        else:
            # 注意：MNISTDataset 返回的是 (X, y)，是元组，会走上面的 if 分支
            # NDArrayDataset 返回的是单个 ndarray，会走这个 else 分支
            # PyTorch 的 DataLoader 返回的是一个列表 [Tensor(batch_stacked)]
            # 我们的测试用例中batch[0]来解包，所以返回元组 (Tensor(batch_stacked),) 是正确的
            batch_stacked = np.stack(batch_samples)
            return (Tensor(batch_stacked),)
