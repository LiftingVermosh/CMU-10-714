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

def sum_gradient_for_broadcasting(input_tensor: Tensor, out_grad: Tensor) -> Tensor:
    """
    处理广播反向传播的核心辅助函数。
    
    当一个 `input_tensor` 在前向传播中被广播以匹配 `output_grad` 的形状时，
    这个函数通过对 `output_grad` 进行求和，将梯度还原到 `input_tensor` 的原始形状。
    Args:
        input_tensor (Tensor): 前向传播时的原始输入张量。
        output_grad (Tensor): 从计算图中传回的、具有广播后形状的梯度。
    Returns:
        Tensor: 一个形状与 `input_tensor` 完全相同的梯度张量。
    """
    # 如果形状相同，直接返回
    if input_tensor.shape == out_grad.shape:
        return out_grad
    
    axes_to_sum = []    

    shrink_dims = len(out_grad.shape) - len(input_tensor.shape)
    if shrink_dims > 0:
        axes_to_sum.extend(range(shrink_dims))

    for i, dim in enumerate(input_tensor.shape):
        if dim == 1 and out_grad.shape[i + shrink_dims] > 1:
            axes_to_sum.append(i + shrink_dims)

    if axes_to_sum:
        summed_grad = summation(out_grad, axes=tuple(axes_to_sum))
    else:
        summed_grad = out_grad

    return reshape(summed_grad, input_tensor.shape)

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        # NOTE: SGD DEBUG
        lhs, rhs = node.inputs
        grad_a = sum_gradient_for_broadcasting(lhs, out_grad)
        grad_b = sum_gradient_for_broadcasting(rhs, out_grad)

        return grad_a, grad_b

def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad,)


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        # 应用链式法则
        grad_a_unsummed = out_grad * rhs
        grad_b_unsummed = out_grad * lhs
        
        # 将梯度还原到原始形状
        grad_a = sum_gradient_for_broadcasting(lhs, grad_a_unsummed)
        grad_b = sum_gradient_for_broadcasting(rhs, grad_b_unsummed)
        
        return grad_a, grad_b


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
        """
        本质上是双输入运算符，需要分别返回每个输入张量的梯度
        对于pow(a, b)，a的梯度是b * pow(a, b-1)，b的梯度是pow(a, b) * log(a)
        注意：
        - 考虑指数为负数的情况，返回错误
        """
        lhs, rhs = node.inputs
        if (rhs.realize_cached_data() < 0).any():
            raise ValueError("不支持负指数")
        # 应用链式法则
        grad_a_unsummed = out_grad * rhs * power(lhs, rhs - 1)
        grad_b_unsummed = out_grad * power(lhs, rhs) * log(lhs)
        
        # 将梯度还原到原始形状
        grad_a = sum_gradient_for_broadcasting(lhs, grad_a_unsummed)
        grad_b = sum_gradient_for_broadcasting(rhs, grad_b_unsummed)
        
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad, node):
        """
        此处相当于是 pow(a, scalar)，a的梯度是scalar * pow(a, scalar-1)
        注意:
        - 返回必须是张量元组形式
        - 考虑指数为 0 的情况，返回 (0,)
        """
        epsilon = 1e-12
        x = node.inputs[0]
        if self.scalar != 0:
            return (out_grad * self.scalar * power_scalar(x, self.scalar - 1),)
        else:
            return (Tensor(array_api.zeros_like(x.realize_cached_data())),)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        """
        本质上是双输入运算符，需要分别返回每个输入张量的梯度
        对于div(a, b)，a的梯度是1/b，b的梯度是-a/b^2
        """
        lhs , rhs = node.inputs
        grad_a_unsummed = out_grad / rhs
        grad_b_unsummed = -out_grad * lhs / (rhs ** 2)

        grad_a = sum_gradient_for_broadcasting(lhs, grad_a_unsummed)
        grad_b = sum_gradient_for_broadcasting(rhs, grad_b_unsummed) 

        return grad_a, grad_b


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        if self.scalar == 0:
            raise ZeroDivisionError(f"division by zero:self.scalar({self.scalar}) cannot be used as a divisor.")
        return a / self.scalar 

    def gradient(self, out_grad, node):
        """
        此处相当于是 div(a, scalar)，a的梯度是1/scalar
        注意:
        - 返回必须是张量元组形式
        - 考虑分母为 0 的情况，返回 (0,)
        """
        if self.scalar == 0:
            # 梯度数学上未定义，直接返回 0 或报错
            return (array_api.zeros_like(node.inputs[0]),)
        return (out_grad / self.scalar,)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)

class Sqrt(TensorOp):
    def compute(self, a):
        return array_api.sqrt(a)

    def gradient(self, out_grad, node):
        """
        对于sqrt(a)，a的梯度是0.5 * out_grad / sqrt(a)
        """
        return (0.5 * out_grad / node,)

def sqrt(a):
    return Sqrt()(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            """ 
            [注]：这块和numpy的transpose有些区别，numpy的transpose默认是默认是转置所有维度，而这里转置最后两个维度 
            """
            # 批量矩阵转置：交换最后两个维度
            return array_api.swapaxes(a, -1, -2)
        elif len(self.axes) == 2 and not all(isinstance(x,int) for x in self.axes):
            raise ValueError("axes 元素必须是整数")
        elif len(self.axes) == 2:
            # 只交换两个指定的轴
            return array_api.swapaxes(a, self.axes[0], self.axes[1])
        elif len(self.axes) == a.ndim:
            # 完整置换
            return array_api.transpose(a, axes=self.axes)
        else:
            raise ValueError("axes 参数不合法")

    def gradient(self, out_grad, node):
        if self.axes is None:
            # 默认批矩阵转置就是自反
            return (transpose(out_grad),)
        elif len(self.axes) == 2:
            return (transpose(out_grad, self.axes),)
        else:
            inv_axes = array_api.argsort(self.axes)
            return (transpose(out_grad, tuple(inv_axes)),)

def transpose(a, axes=None):
    return Transpose(axes)(a)

class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        """
        对于reshape(a, shape)，a的梯度是reshape(out_grad, a.shape)
        """
        return (reshape(out_grad, node.inputs[0].shape),)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        """
        此处相当于是 broadcast_to(a, shape)，a的梯度是out_grad
        注意:
        - 返回必须是张量元组形式
        - 考虑输入张量形状与目标形状不一致的情况，返回 (0,)
        """
        input_shape = node.inputs[0].shape
        output_shape = self.shape
        
        # 找出需要求和的轴
        # 如果输出的维度比输入多，那么需要对这些多出来的维度求和
        shrink_dims = len(output_shape) - len(input_shape)
        axes = list(range(shrink_dims))
        # 找出因为 size=1 而被广播的轴
        for i in range(len(input_shape)):
          if input_shape[i] == 1 and output_shape[i + shrink_dims] > 1:
            axes.append(i + shrink_dims)
        
        # 执行求和
        grad = summation(out_grad, axes=tuple(axes))
        
        # Reshape 回原始输入形状
        grad = reshape(grad, input_shape)
        
        return (grad,)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None, keepdims: bool = False):
        self.axes = axes
        self.keepdims = keepdims # 添加 keepdims 属性
    def compute(self, a):
        # 将 keepdims 参数传递给 numpy.sum
        return array_api.sum(a, axis=self.axes, keepdims=self.keepdims)
    def gradient(self, out_grad, node):
        """
        此处对原代码做出了一些修改，主要是为了处理 keepdims=False 的情况。
        对于 summation(a, axes=None, keepdims=False)，a的梯度是reshape(out_grad, a.shape)
        对于 summation(a, axes=None, keepdims=True)，a的梯度是broadcast_to(out_grad, a.shape)
        """
        input_shape = node.inputs[0].shape
        
        if not self.keepdims:
            if self.axes is None:
                axes = range(len(input_shape))
            elif isinstance(self.axes, int):
                axes = (self.axes,)
            else:
                axes = self.axes
            
            grad_shape = list(out_grad.shape)
            for ax in sorted(axes):
                grad_shape.insert(ax, 1)
            grad = reshape(out_grad, tuple(grad_shape))
        else:
            grad = out_grad
        return (broadcast_to(grad, input_shape),)
    
def summation(a, axes=None, keepdims=False): # 添加 keepdims 参数
    return Summation(axes=axes, keepdims=keepdims)(a) # 传递 keepdims


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        """
        本质上是双输入运算符，需要分别返回每个输入张量的梯度
        对于matmul(a, b)，a的梯度是matmul(out_grad, b.T)，b的梯度是matmul(a.T, out_grad)
        """
        lhs, rhs = node.inputs
        # return (matmul(out_grad, transpose(rhs, axes=(-1, -2))),
        #         matmul(transpose(lhs, axes=(-1, -2)), out_grad))
        lhs_T = transpose(lhs, axes=(-1, -2))
        rhs_T = transpose(rhs, axes=(-1, -2))
        grad_lhs = matmul(out_grad, rhs_T)
        grad_rhs = matmul(lhs_T, out_grad)
        # 处理广播还原成 lhs/rhs 的原始形状
        while len(grad_lhs.shape) > len(lhs.shape):
            grad_lhs = summation(grad_lhs, axes=0)
        while len(grad_rhs.shape) > len(rhs.shape):
            grad_rhs = summation(grad_rhs, axes=0)
        return grad_lhs, grad_rhs

def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        """
        对于negate(a)，a的梯度是-out_grad
        """
        return (-out_grad,)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        """
        对于log(a)，a的梯度是out_grad / a
        """
        return (out_grad / node.inputs[0],)


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        """
        对于exp(a)，a的梯度是out_grad * exp(a)
        """
        return (out_grad * exp(node.inputs[0]),)


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        """
        对于 ReLU 函数，当输入大于 0 时，输出等于输入，反之输出等于 0。
        因此，ReLU 的梯度是当输入大于 0 时，梯度等于 out_grad，反之梯度等于 0。
        注意:
        - 考虑输入张量中存在负数的情况，返回 (0,)
        """
        # 反向传播：梯度为1 where input > 0, 0 otherwise
        input_data = node.inputs[0].realize_cached_data()
        mask = (input_data > 0).astype(array_api.float32)
        mask = Tensor(mask)
        return (out_grad * mask,)


def relu(a):
    return ReLU()(a)

class Max(TensorOp):
    def __init__(self, axes: Optional[tuple] = None, keepdims=False):
        self.axes = axes
        self.keepdims = keepdims
    def compute(self, a):
        return array_api.max(a, axis=self.axes, keepdims=self.keepdims)
    def gradient(self, out_grad, node):
        """
        最大值的梯度计算，注意需要使用 needle 的 Op 来构建计算图。
        """
        a = node.inputs[0]
        # 直接使用前向计算得到的 Tensor 结果 (node 本身)
        # 注意 node.realize_cached_data() 才是 NDArray, node 是 Value/Tensor
        max_val = node 
        
        # 恢复维度和形状
        # 如果 keepdims=False, 需要恢复被压缩的维度
        if not self.keepdims and self.axes is not None:
            out_shape = list(a.shape)
            if isinstance(self.axes, int): axes = (self.axes,)
            else: axes = self.axes
            
            for ax in axes:
                out_shape[ax] = 1
            max_val = reshape(max_val, out_shape)
        
        # 将最大值广播到原始输入的形状
        max_val_broadcasted = broadcast_to(max_val, a.shape)
        
        # 创建掩码
        mask = (a == max_val_broadcasted)
        
        # 处理多个最大值的情况
        # 使用 summation 和 broadcast_to 来计算每个区域最大值的数量
        div_factor = summation(mask, axes=self.axes, keepdims=True)
        mask = mask / broadcast_to(div_factor, a.shape)
        
        # 应用链式法则 (反向传播梯度)
        # 如果 keepdims=False, out_grad 也需要恢复维度并广播
        if not self.keepdims and self.axes is not None:
             # out_grad 的 shape 就是 max_val 在 keepdims=False 时的 shape, 
             # 所以可以用上面相同的逻辑恢复
            out_shape = list(a.shape)
            if isinstance(self.axes, int): axes = (self.axes,)
            else: axes = self.axes
            for ax in axes:
                out_shape[ax] = 1
            out_grad = reshape(out_grad, out_shape)
        return broadcast_to(out_grad, a.shape) * mask
def max(a, axes=None, keepdims=False):
    return Max(axes, keepdims)(a)


class Mean(TensorOp):
    def __init__(self, axes: Optional[tuple] = None, keepdims=False):
        self.axes = axes
        self.keepdims = keepdims
    def compute(self, a):
        return array_api.mean(a, axis=self.axes, keepdims=self.keepdims)
    def gradient(self, out_grad, node):
        a = node.inputs[0]
        input_shape = a.shape
        
        # 计算参与平均的元素数量
        if self.axes is None:
            count = 1
            for dim in input_shape:
                count *= dim
        else:
            count = 1
            axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
            for ax in axes:
                count *= input_shape[ax]
        # 如果 keepdims=False, out_grad 的形状少了维度，需要先恢复
        if not self.keepdims and self.axes is not None:
            new_out_grad_shape = list(out_grad.shape)
            _axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
            for ax in sorted(_axes):
                new_out_grad_shape.insert(ax, 1) # 在被求和的轴上插入维度1
            out_grad = reshape(out_grad, tuple(new_out_grad_shape))
        
        # 广播到原始输入形状
        grad = broadcast_to(out_grad, input_shape)
        # 梯度均匀分配
        return grad / count # 这里的 / count 也会触发 DivScalar Op
    
def mean(a, axes=None, keepdims=False):
    return Mean(axes, keepdims)(a)

class EWiseCompare(TensorOp):
    """
    比较操作的基类，用于实现其他比较操作。
    """
    def gradient(self, out_grad:Tensor, node:Tensor):
        # 比较操作不可导，梯度恒为 0
        return (out_grad * 0, out_grad * 0)

class EWiseLessThan(EWiseCompare):
    def compute(self, a: NDArray, b: NDArray):
        # 返回 float32 的 0. 或 1. 标志，方便后续梯度运算
        return (a < b).astype(array_api.float32)
    
def less_than(a, b):
    return EWiseLessThan()(a, b)

class EWiseLessEqual(EWiseCompare):
    def compute(self, a: NDArray, b: NDArray):
        # 返回 float32 的 0. 或 1. 标志，方便后续梯度运算
        return (a <= b).astype(array_api.float32)

def less_equal(a, b):
    return EWiseLessEqual()(a, b)
    
class EWiseGreaterThan(EWiseCompare):
    def compute(self, a: NDArray, b: NDArray):
        # 返回 float32 的 0. 或 1. 标志，方便后续梯度运算
        return (a > b).astype(array_api.float32)

def greater_than(a, b):
    return EWiseGreaterThan()(a, b)
    
class EWiseGreaterEqual(EWiseCompare):
    def compute(self, a: NDArray, b: NDArray):
        # 返回 float32 的 0. 或 1. 标志，方便后续梯度运算
        return (a >= b).astype(array_api.float32)

def greater_equal(a, b):
    return EWiseGreaterEqual()(a, b)

class EWiseEqual(EWiseCompare):
    def compute(self, a: NDArray, b: NDArray):
        # 返回 float32 的 0. 或 1. 标志，方便后续梯度运算
        return array_api.equal(a, b).astype(array_api.float32)

def equal(a, b):
    return EWiseEqual()(a, b)

class EWiseNotEqual(EWiseCompare):
    def compute(self, a: NDArray, b: NDArray):
        # 返回 float32 的 0. 或 1. 标志，方便后续梯度运算
        return array_api.not_equal(a, b).astype(array_api.float32)

def not_equal(a, b):
    return EWiseNotEqual()(a, b)
