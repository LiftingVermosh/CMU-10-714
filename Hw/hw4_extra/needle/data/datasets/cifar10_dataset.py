import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        self.__init__(transforms=transforms)
        self.transforms = transforms
        if train:
            file_name = [f'data_batch_{i}' for i in range(1, 6)]
        else:
            file_name = ['test_batch']

        self.X, self.y = self.parse_cifar10(base_folder, file_name)

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        if isinstance(index, int):
            img, label = self.X[index], self.y[index]
            if self.transforms is not None:
                img = self.transforms(img)
            return img, label
        elif isinstance(index, slice):
            return self.X[index], label
        else:
            raise RuntimeError(f'Unsupport index:{index}')            

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        return self.X.shape[0]

    def parse_cifar10(dataset_dir:str, file_names: list[str]):
        """
        解析 cifar10 数据集文件
        """
        X_list = []
        y_list = []
        for file_name in file_names:
            file = os.join(dataset_dir, file_name)
            with open(file, 'rb') as f:
                dict = pickle.load(f, encoding='utf-8')
                X_list.append(dict[b'data'])
                y_list.append(dict[b'label'])
        X = np.concatenate(X_list, axis=0).astype(np.float32)
        y = np.concatenate(y_list, axis=0).astype(np.uint8)

        X = X / 255.0
        X = X.reshape(-1, 3, 32, 32)

        return X, y