"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api

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
            return (out_grad * array_api.ones(input_shape), )   
        
        # 确定目标形状
        target_shape = list(input_shape)

        # 确定需要求和的轴
        axes_to_reshape = self.axes if isinstance(self.axes, tuple) else (self.axes,)
        for axis in axes_to_reshape:
            target_shape[axis] = 1
        
        reshaped_out_grad = out_grad.reshape(tuple(target_shape))

        return (reshaped_out_grad.broadcast_to(input_shape), )

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
        mask = (input_data > 0).astype(array_api.float32)
        mask = Tensor(mask)
        return (out_grad * mask,)


def relu(a):
    return ReLU()(a)

