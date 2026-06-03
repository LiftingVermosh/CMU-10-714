"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from .....tmp.python.needle.autograd import NDArray
from .....tmp.python.needle.autograd import Op, Tensor, Value, TensorOp
from .....tmp.python.needle.autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from .....tmp.python.needle.backend_selection import array_api, BACKEND
from .....tmp.python.needle.ops.ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return array_api.power(a, b)
        
    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        # f'(a) = b * a^(b-1), f'(b) = ln(a) * a^b
        return out_grad * rhs * array_api.power(lhs, rhs - 1), out_grad * log(lhs) * array_api.power(lhs, rhs)

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad, node):
        lhs = node.inputs[0]
        # Turn to tuple
        return (out_grad * self.scalar * array_api.power(lhs, self.scalar - 1), )


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return array_api.divide(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        # f'(a) = 1/b, f'(b) = -a/b^2
        return out_grad / rhs, -out_grad * lhs / array_api.power(rhs, 2)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return array_api.divide(a, self.scalar)

    def gradient(self, out_grad, node):
        return (out_grad / self.scalar, )


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        N = a.ndim
        ori_axes = list(range(N))
        # By default, reverse the last two dimensions of the tensor
        if self.axes is None:
            ori_axes[-1], ori_axes[-2] = ori_axes[-2], ori_axes[-1]
            return array_api.transpose(a, axes=ori_axes)
        # The whole axes tuple is provided
        if len(self.axes) != 2 and len(self.axes) == len(ori_axes):
            return array_api.transpose(a, axes=self.axes)
        # Specify two special axes 
        else:
            ori_axes[self.axes[0]], ori_axes[self.axes[1]] = ori_axes[self.axes[1]], ori_axes[self.axes[0]]
            return array_api.transpose(a, axes=ori_axes)


    def gradient(self, out_grad, node):
        # A^T' = A'^T
        return (out_grad.transpose(self.axes), )


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        if self.shape is None:
            raise ValueError("Reshape shape cannot be None")
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        return (out_grad.reshape(node.inputs[0].shape), )


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        if self.shape is None:
            raise ValueError("BroadcastTo shape cannot be None")
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        input_tensor = node.inputs[0]
        input_shape = input_tensor.shape
        output_shape = self.shape # out_grad.shape
        # 从右往左确定需要求和的轴 
        axes_to_sum = []
        
        # 处理由于输入维度少而导致的广播（例如(10,) -> (5, 10)）
        # 需要对前缀轴进行求和
        diff = len(output_shape) - len(input_shape)
        if diff > 0:
            axes_to_sum.extend(range(diff))
        # 处理由于维度为1而导致的广播
        for i in range(len(input_shape)):
            out_dim = output_shape[i + diff]
            in_dim = input_shape[i]
            
            if in_dim == 1 and out_dim > 1:
                axes_to_sum.append(i + diff)
        
        # 执行求和
        if axes_to_sum:
            grad = out_grad.sum(axes=tuple(axes_to_sum))
        else:
            grad = out_grad
        
        # 确保结果形状严格匹配输入形状
        return (grad.reshape(input_shape),)

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        if self.axes is None:
            target_shape = [1] * len(input_shape)
        else:
            target_shape = list(input_shape)
            axes_list = self.axes if isinstance(self.axes, tuple) else (self.axes,)
            for axis in axes_list:
                target_shape[axis] = 1
                
        # Reshape 并广播后再相乘
        return (out_grad.reshape(tuple(target_shape)).broadcast_to(input_shape), )

def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        # f'(A) = B^T, f'(B) = A^T
        grad_A = out_grad.matmul(rhs.transpose())
        grad_B = lhs.transpose().matmul(out_grad)
        
        if len(grad_A.shape) > len(lhs.shape):
            axes_to_sum = tuple(range(len(grad_A.shape) - len(lhs.shape)))
            grad_A = grad_A.sum(axes=axes_to_sum)

        if len(grad_B.shape) > len(rhs.shape):
            axes_to_sum = tuple(range(len(grad_B.shape) - len(rhs.shape)))
            grad_B = grad_B.sum(axes=axes_to_sum)

        grad_A = grad_A.reshape(lhs.shape)
        grad_B = grad_B.reshape(rhs.shape)

        return grad_A, grad_B


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return array_api.negative(a)

    def gradient(self, out_grad, node):
        return (-out_grad, )


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        lhs = node.inputs[0]
        # f'(a) = 1/a
        return (out_grad / lhs, )


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        lhs = node.inputs[0]
        # f'(a) = e^a
        return (out_grad * exp(lhs), )


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        input_data = node.inputs[0].realize_cached_data()
        mask = Tensor(input_data > 0, device=node.device, dtype="float32") 
        return (out_grad * mask,)


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        return array_api.tanh(a)

    def gradient(self, out_grad, node):
        return (out_grad * (add_scalar(negate(node**2), 1)), )


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        n = len(args)
        ndim = args[0].ndim
        axis = self.axis % (ndim + 1)
        
        # 计算新 shape
        old_shape = args[0].shape
        new_shape = list(old_shape)
        new_shape.insert(axis, n)
        
        # 分配空间
        out = array_api.NDArray.make(tuple(new_shape), device=args[0].device)
        
        # 准备 reshape 后的子块形状，用于匹配切片
        slice_shape = list(old_shape)
        slice_shape.insert(axis, 1)
        
        # 搬运数据
        for i in range(n):
            # 构造切片索引，类似于 out[:, :, i:i+1, :]
            indices = [slice(0, s) for s in new_shape]
            indices[axis] = slice(i, i + 1)
            
            temp_shape = list(args[i].shape)
            temp_shape.insert(self.axis, 1)
            out[tuple(indices)] = args[i].compact().reshape(tuple(temp_shape))
        return out


    def gradient(self, out_grad, node):
        return split(out_grad, axis=self.axis)


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        n = A.shape[self.axis]
        new_shape = list(A.shape)
        new_shape.pop(self.axis)

        results = []

        for i in range(n):
            indices = [slice(0, s) for s in A.shape]
            indices[self.axis] = slice(i, i + 1)

            sub_array = A[tuple(indices)]

            sub_array = A[tuple(indices)].compact().reshape(tuple(new_shape))
            results.append(sub_array)
        return tuple(results)


    def gradient(self, out_grad, node):
        return stack(out_grad, axis=self.axis)


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return a.flip(self.axes)

    def gradient(self, out_grad, node):
        return flip(out_grad, self.axes)


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        new_shape = list(a.shape)
        for axis in self.axes:
            # 2 Failed Realization
            # new_shape[axis] = new_shape[axis] + (new_shape[axis] - 1) * self.dilation
            new_shape[axis] = a.shape[axis] * (self.dilation + 1)
        
        new_shape = tuple(new_shape)
        
        out = a.device.full(new_shape, 0.0)
        
        # 构造切片 (Slices) 以便将 a 的数据填入 out
        # 在被 dilate 的轴上每隔 (dilation + 1) 个位置填入一个原数据
        slices = []
        for i in range(len(a.shape)):
            if i in self.axes:
                # slice(start, stop, step)
                slices.append(slice(0, new_shape[i], self.dilation + 1))
            else:
                slices.append(slice(None)) # 相当于 :
        
        out[tuple(slices)] = a
        
        return out

    def gradient(self, out_grad, node):
        return undilate(out_grad, self.axes, self.dilation)


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        slices = []
        for i in range(len(a.shape)):
            if i in self.axes:
                # 每隔 dilation + 1 采样一个点
                slices.append(slice(0, a.shape[i], self.dilation + 1))
            else:
                slices.append(slice(None))
        
        return a[tuple(slices)].compact()

    def gradient(self, out_grad, node):
        return dilate(out_grad, self.axes, self.dilation)


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        # A:NDArray (N, H, W, C_in)
        # B:NDArray (K, K, C_in, C_out)

        padded_A = A.pad((
            (0, 0),
            (self.padding, self.padding),
            (self.padding, self.padding),
            (0, 0)
        ))
        
        padded_A = padded_A.compact()
        
        N, H_padded, W_padded, C_in = padded_A.shape
        K, _, _, C_out = B.shape # B: (K, K, C_in, C_out)
        
        # 获取 padded 后的 strides
        p_sn, p_sh, p_sw, p_sc = padded_A.strides
        
        # 计算输出尺寸
        # 注意：这里的 H 是原始输入的尺寸，或者由 H_padded 倒推
        # H = H_padded - 2 * self.padding
        H_out = (H_padded - K) // self.stride + 1
        W_out = (W_padded - K) // self.stride + 1
        
        L = self.stride

        # 滑动窗口视图
        new_shape  = (N, H_out, W_out, K, K, C_in)
        # 步幅逻辑：
        # Batch 跳 p_sn
        # H_out 每次跳 L 行，即 p_sh * L
        # W_out 每次跳 L 列，即 p_sw * L
        # 内层 K, K, C_in 维持原来的 p_sh, p_sw, p_sc
        new_strides = (p_sn, p_sh * L, p_sw * L, p_sh, p_sw, p_sc) 
        
        A_window = padded_A.as_strided(new_shape, new_strides)

        # 矩阵乘法
        A_matrix = A_window.compact().reshape((N * H_out * W_out, K * K * C_in))
        B_matrix = B.compact().reshape((K * K * C_in, C_out))
        
        out = A_matrix @ B_matrix
        return out.reshape((N, H_out, W_out, C_out))



    def gradient(self, out_grad, node):
        A, B = node.inputs
        K = B.shape[0]
        p = self.padding
        s = self.stride
        
        if self.stride > 1:
            out_grad = dilate(out_grad, axes=(1, 2), dilation=self.stride - 1)
        
        # grad_A (对输入的梯度) 
        # 翻转权重 B 的空间维度 (K, K)
        B_flip = flip(B, axes=(0, 1))
        
        # 交换权重 B 的输入输出通道 -> 形状变成 (K, K, C_out, C_in)
        B_trans = transpose(B_flip, axes=(2, 3))
        
        # 用扩张后的 out_grad 和变换后的 B 做卷积，步长恒为 1，计算特殊的 Padding
        grad_A_pad = K - 1 - self.padding
        grad_A = conv(out_grad, B_trans, stride=1, padding=grad_A_pad)
        
        # grad_B (对权重的梯度)
        # 将 A 的 Batch(N) 和 Channel(C_in) 互换 -> 形状 (C_in, H, W, N)
        A_trans = transpose(A, axes=(0, 3))
        
        # 将 out_grad 转为假权重，形状从 (N, H, W, C_out) 变为 (H, W, N, C_out)
        # 先互换 (0, 1) 变成 (H, N, W, C_out)，再互换 (1, 2) 变成 (H, W, N, C_out)
        out_grad_trans = transpose(out_grad, axes=(0, 1))
        out_grad_trans = transpose(out_grad_trans, axes=(1, 2))
        
        # 使用假输入和假权重的卷积
        grad_B_tmp = conv(A_trans, out_grad_trans, stride=1, padding=self.padding)
        # 此时 grad_B_tmp 形状为 (C_in, K, K, C_out)
        
        # 把形状变换回标准权重维度 (K, K, C_in, C_out)
        # 先互换 (0, 1) 变 (K, C_in, K, C_out)，再互换 (1, 2) 边 (K, K, C_in, C_out)
        grad_B = transpose(grad_B_tmp, axes=(0, 1))
        grad_B = transpose(grad_B, axes=(1, 2))
        
        return grad_A, grad_B


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


