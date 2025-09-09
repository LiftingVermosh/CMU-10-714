import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass



def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    return x + y


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


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    # 装逼用一行流
    # return -np.mean(    # 平均值    
    #     np.log(         # 对数
    #         np.exp(     # 指数
    #             (Z-np.max(Z,axis=1,keepdims=True))[np.arange(Z.shape[0]),y]      # 取最大值后再取对应标签的logit
    #             )/np.sum(    # 求和
    #                 np.exp(Z-np.max(Z,axis=1,keepdims=True)),axis=1,keepdims=True
    #                 )    # 求softmax
    #             )
    #         )

    # 数值稳定性：减去每行的最大值
    max_Z = np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z - max_Z)  # 形状: (batch_size, num_classes)
    
    # 计算softmax分母
    sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)  # 形状: (batch_size, 1)
    
    # 获取每个样本真实类别的概率
    n = Z.shape[0]
    true_class_probs = exp_Z[np.arange(n), y]  # 形状: (batch_size,)
    
    # 计算log softmax概率 for true class
    log_probs = np.log(true_class_probs / sum_exp_Z.flatten())  # 除法后形状匹配
    
    # 返回平均损失
    return -np.mean(log_probs)


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ 
    对数据执行单轮SGD训练实现softmax回归，使用步长lr和指定的批处理大小。该函数应原地修改theta矩阵，并按顺序遍历X中的批次（不进行随机打乱）。

    参数说明：
            X (np.ndarray[np.float32]): 二维输入数组，尺寸为
                (样本数 x 输入维度)
            y (np.ndarray[np.uint8]): 一维类别标签数组，尺寸为 (样本数,)
            theta (np.ndarray[np.float32]): softmax回归参数的二维数组，形状为
                (输入维度, 类别数)
            lr (float): SGD的步长（学习率）
            batch (int): SGD小批量样本数量

    返回值：
            无
    """
    n_samples = X.shape[0]
    n_classes = theta.shape[1]

    for i in range(0, n_samples, batch):
        X_batch= X[i:i+batch]
        y_batch= y[i:i+batch]
        batch_size = X_batch.shape[0]

        # 计算softmax概率
        Z = X_batch @ theta
        max_Z = np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z - max_Z)  # 形状: (batch_size, num_classes)
        sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)  # 形状: (batch_size, 1)
        probs = exp_Z / sum_exp_Z  # 形状: (batch_size, num_classes)

        # 热编码 y
        y_one_hot = np.eye(n_classes)[y_batch]  # 形状: (batch_size, num_classes)

        # 计算梯度
        dZ = (probs - y_one_hot) / batch_size  # 形状: (batch_size, num_classes)
        dW = X_batch.T @ dZ  # 形状: (input_dim, num_classes)

        # 更新参数
        theta -= lr * dW


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ 
    对由权重矩阵W1和W2（无偏置项）定义的双层神经网络执行单轮SGD训练：
        logits = ReLU(X * W1) * W2
    该函数应使用指定的学习率lr和批处理大小batch（且不随机打乱X的顺序），并原地修改W1和W2矩阵。

    参数说明：
        X (np.ndarray[np.float32]): 二维输入数组，尺寸为
            (样本数 x 输入维度)
        y (np.ndarray[np.uint8]): 一维类别标签数组，尺寸为 (样本数,)
        W1 (np.ndarray[np.float32]): 第一层权重二维数组，形状为
            (输入维度, 隐藏层维度)
        W2 (np.ndarray[np.float32]): 第二层权重二维数组，形状为
            (隐藏层维度, 类别数)
        lr (float): SGD的步长（学习率）
        batch (int): SGD小批量样本数量

    返回值：
            无
    """
    def ReLU(x):
        return np.maximum(x, 0)
    
    n_samples = X.shape[0]
    n_classes = W2.shape[1]

    # 批次处理
    for i in range(0, n_samples, batch):
        X_batch= X[i:i+batch]
        y_batch= y[i:i+batch]

        # 前向传播
        H1 = X_batch @ W1   # 隐藏层输入
        A1 = ReLU(H1)       # ReLU激活
        logits = A1 @ W2    # 输出层

        # 计算softmax概率
        max_logits = np.max(logits, axis=1, keepdims=True)      # 取最大
        exp_logits = np.exp(logits - max_logits)                # 取指
        sum_exp_logits = np.sum(exp_logits, axis=1, keepdims=True)  # 求和
        probs = exp_logits / sum_exp_logits                       # 概率

        # y 转换为 one-hot 编码
        y_onehot = np.eye(n_classes)[y_batch]

        # 反向传播
        dZ = (probs - y_onehot) / batch         # 计算 logits 梯度
        dW2 = A1.T @ dZ                         # 计算 W2 梯度
        dA1 = dZ @ W2.T                         # 计算 A1 梯度
        dH1 = dA1 * (H1 > 0)                    # 计算 H1 梯度  
        dW1 = X_batch.T @ dH1                   # 计算 W1 梯度

        # 更新参数
        W1 -= lr * dW1
        W2 -= lr * dW2

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
