"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
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


def softmax_loss(Z, y_one_hot):
    """
    返回Softmax loss。 请注意，就本作业而言、
    你不需要担心如何 "很好地 "缩放数值属性
    的数值特性，而只需直接计算即可。

    参数
        Z（ndl.Tensor[np.float32]）：形状为
            (batch_size，num_classes)的二维张量，包含每个类别的 logit 预测值。
            每个类别的 logit 预测值。
        y_one_hot（ndl.Tensor[np.int8]）：形状（batch_size、num_classes）的二维张量，包含每个类别的对数预测。
            在每个示例的真实标签索引处包含一个 1，其他地方为 0。
            其他地方为 0。

    返回值
        样本的Average Softmax Loss(ndl.张量[np.float32])
    """
    batch_size = Z.shape[0]
    # log_sum_exp 部分
    # Z(batch, classes), max_Z(batch, 1)
    max_Z = ndl.max(Z, axes=1, keepdims=True)
    # log_sum_exp(batch, 1)
    log_sum_exp = ndl.log(ndl.summation(ndl.exp(Z - max_Z.broadcast_to(Z.shape)), axes=1)).reshape((batch_size, 1)) + max_Z
    
    # 提取真实类别对应的 logits (z_y)
    # Z(batch, classes), y_one_hot(batch, classes) -> z_y(batch, 1)
    z_y = ndl.summation(Z * y_one_hot, axes=1).reshape((batch_size, 1))
    # 计算每个样本的损失 (log_sum_exp - z_y)
    # loss_per_sample (batch, 1)
    loss_per_sample = log_sum_exp - z_y
    
    # 计算平均损失
    # 使用 sum / batch_size, 这与 mean() 等价但更明确
    avg_loss = ndl.summation(loss_per_sample) / batch_size
    
    return avg_loss


    # # Softmax loss 的计算公式：
    # # loss = -log(softmax(z_i)) for correct class i
    # # 其中 softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)
    
    # # 步骤1: 计算 log-softmax
    # # log_softmax(z_i) = z_i - log(sum(exp(z_j)))
    
    # # 为了数值稳定性，我们使用 log-sum-exp 技巧
    # # log(sum(exp(z_j))) = max(z) + log(sum(exp(z_j - max(z))))
    
    # batch_size = Z.shape[0]
    
    # # 计算每个样本的最大logit值（用于数值稳定性）
    # z_max = ndl.Tensor(np.max(Z.numpy(), axis=1, keepdims=True))
    
    # # 计算稳定的指数项 exp(z - z_max)
    # exp_z = ndl.exp(Z - z_max.broadcast_to(Z.shape))
    
    # # 计算每个样本的指数和
    # sum_exp_z = ndl.summation(exp_z, axes=(1,))
    
    # # 计算 log_softmax = z - z_max - log(sum_exp_z)
    # log_sum_exp_z = ndl.log(sum_exp_z)
    # log_softmax = Z - z_max.broadcast_to(Z.shape) - log_sum_exp_z.reshape((batch_size, 1)).broadcast_to(Z.shape)
    
    # # 计算交叉熵损失：-sum(y_true * log_softmax)
    # # 由于 y_one_hot 在正确类别处为1，其他地方为0，所以这等价于选择正确类别的log_softmax值
    # loss_per_sample = -ndl.summation(y_one_hot * log_softmax, axes=(1,))
    
    # # 返回平均损失
    # avg_loss = ndl.summation(loss_per_sample) / batch_size
    
    # return avg_loss



def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """
    为两层神经网络运行一轮SGD
    
    参数:
    X (np.ndarray[np.float32]): 2D输入数组，形状为 (num_examples x input_dim)
    y (np.ndarray[np.uint8]): 1D类别标签数组，形状为 (num_examples,)
    W1 (ndl.Tensor[np.float32]): 第一层权重，形状为 (input_dim, hidden_dim)
    W2 (ndl.Tensor[np.float32]): 第二层权重，形状为 (hidden_dim, num_classes)
    lr (float): SGD的学习率
    batch (int): SGD小批量大小
    返回:
    Tuple: (W1, W2)
        W1: ndl.Tensor[np.float32]
        W2: ndl.Tensor[np.float32]
    """
    num_examples = X.shape[0]
    for start in range(0, num_examples, batch):
        end = start + batch
        X_batch = ndl.Tensor(X[start:end])   # (B, in_dim)
        y_batch = y[start:end]               # (B,)
        # Forward
        h = ndl.relu(ndl.matmul(X_batch, W1))                  # (B, hid_dim)
        z = ndl.matmul(h, W2)                                  # (B, num_classes)
        # Softmax
        max_z = ndl.max(z, axes=1, keepdims=True)
        exp_z = ndl.exp(z - max_z.broadcast_to(z.shape))
        sum_exp_z = exp_z.sum(axes=1, keepdims=True)
        probs = exp_z / sum_exp_z
        # One-hot
        y_one_hot = np.zeros((end-start, W2.shape[1]), dtype=np.float32)
        y_one_hot[np.arange(end-start), y_batch] = 1
        y_one_hot = ndl.Tensor(y_one_hot)

        # Backward
        B = end - start
        dZ = (probs - y_one_hot) / B                            # (B, num_classes)
        dW2 = ndl.matmul(h.transpose(), dZ)                     # (hid_dim, num_classes)
        dH = ndl.matmul(dZ, W2.transpose())                     # (B, hid_dim)
        relu_mask = (h.numpy() > 0).astype(np.float32)           # 取 mask
        dH_relu = ndl.Tensor(dH.numpy() * relu_mask)             # 屏蔽负区梯度
        dW1 = ndl.matmul(X_batch.transpose(), dH_relu)           # (in_dim, hid_dim)
        
        
        # 框架中的完整实现
        # Softmax loss
        # loss = softmax_loss(z, y_one_hot)

        # # Backward
        # loss.backward()

        # dW1 = W1.grad
        # dW2 = W2.grad
        
        # 参数更新
        W1 = W1 - lr * dW1
        W2 = W2 - lr * dW2
    return W1, W2

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
