"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
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
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    batch_size, _ = Z.shape

    Z_y = Z * y_one_hot
    Z_y = ndl.summation(Z_y, axes=1)    # (B, 1)

    exp_Z = ndl.exp(Z)
    sum_exp_Z = ndl.summation(exp_Z, axes=1)    # (B, 1)
    log_sum_exp_Z = ndl.log(sum_exp_Z)

    loss_softmax = log_sum_exp_Z - Z_y
    loss_softmax = ndl.summation(loss_softmax) / ndl.Tensor(batch_size)
    return loss_softmax


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    num_samples = X.shape[0]
    for i in range(0, num_samples, batch):
        X_batch = X[i:i+batch]
        y_batch = y[i:i+batch]
        tensor_X = ndl.Tensor(X_batch)

        # Forward pass
        H = ndl.relu(ndl.matmul(tensor_X, W1))    # (B, H)
        Z = ndl.matmul(H, W2)    # (B, C)

        # loss
        y_one_hot = np.zeros((X_batch.shape[0], Z.shape[-1]))
        y_one_hot[np.arange(y_batch.size), y_batch] = 1
        tensor_y = ndl.Tensor(y_one_hot)

        loss = softmax_loss(Z, tensor_y)

        # backward
        loss.backward()

        dW1 = W1.grad
        dW2 = W2.grad

        # update parameters
        W1 = (W1 - lr * dW1).detach()
        W2 = (W2 - lr * dW2).detach()

        W1.grad = None
        W2.grad = None

    return W1, W2

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
