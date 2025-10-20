from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        self.transforms = transforms
        self.X, self.y = self.parse_mnist(image_filename, label_filename)
        
    def __getitem__(self, index) -> object:
        if isinstance(index, (int, np.integer)):
            features = self.X[index].reshape(28, 28, 1)
            features = self.apply_transforms(features)
            return features.flatten(), self.y[index]
        elif isinstance(index, slice):
            return self.X[index], self.y[index]
        else:
            raise TypeError(f"Invalid argument type: {type(index)}")
    
    def __len__(self) -> int:
        return self.X.shape[0]


    def parse_mnist(self, image_filename: str, label_filename: str) -> np.ndarray:
        """ 读取 MNIST 格式的图像和标签文件。 请参见此页：
        http://yann.lecun.com/exdb/mnist/ 获取文件格式说明。

        参数：
            image_filename (str)：以 MNIST 格式压缩的图像文件的名称
            label_filename (str)：以 MNIST 格式压缩的标签文件名称

        返回值
            元组 (X,y)：
                X（numpy.ndarray[np.float32]）：包含加载数据的 2D numpy 数组。 
                    数据的二维 numpy 数组。 数据的维度应为 
                    (num_examples x input_dim)，其中 "input_dim "是数据的全维度，例如 
                    例如，由于 MNIST 图像的尺寸是 28x28，所以它 
                    为 784。 值的类型应为 np.float32，数据 
                    的最小值为 0.0，最大值为 1.0。 
                    最大值为 1.0（即，将原始值 0 缩放为 0.0，将 255 缩放为 1.0 
                    和 255 分辨率为 1.0）。

                y（numpy.ndarray[dtype=np.uint8]）： 1D numpy 数组，包含
                    数组。 值的类型应为 np.uint8，且
                    对于 MNIST 将包含 0-9 的值。
            """
        import struct
        import numpy as np
        import gzip

        # Load the images and labels
        with gzip.open(label_filename, 'rb') as f:
            magic, num_images = struct.unpack('>II', f.read(8))     # 当前架构为 x86 64 大端序 且前8个字节为魔数和图像数量
            if magic != 2049:
                raise ValueError('非法的标签魔数: {}'.format(magic))
            labels = np.frombuffer(f.read(), dtype=np.uint8)

        with gzip.open(image_filename, 'rb') as f:
            magic, num_images, num_rows, num_cols = struct.unpack('>IIII', f.read(16))
            if magic != 2051:
                raise ValueError('非法的图像魔数: {}'.format(magic))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows*num_cols)

        # Normalize the images to [0,1]
        images = images.astype(np.float32) / 255.0
        return images, labels