from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    """LogSoftmax 实现，数值稳定"""

    def compute(self, Z: NDArray) -> NDArray:
        # LogSoftmax = Z - LogSumExp(Z) 按行（或指定轴）计算
        max_Z = array_api.max(Z, axis=1, keepdims=True)  # 减去最大值提升数值稳定性
        log_sum_exp = array_api.log(array_api.sum(array_api.exp(Z - max_Z), axis=1, keepdims=True)) + max_Z
        return Z - log_sum_exp

    def gradient(self, out_grad: Tensor, node: Tensor):
        Z = node.inputs[0] 
        axis = 1 
        
        max_Z = max(Z, axes=axis, keepdims=True)
        Z_exp = exp(Z - max_Z)
        sum_Z_exp = summation(Z_exp, axes=axis, keepdims=True)
        grad_factor = Z_exp / sum_Z_exp
        sum_out_grad = summation(out_grad, axes=axis, keepdims=True)
        grad_factor = grad_factor * sum_out_grad
        
        return out_grad - grad_factor

def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    """LogSumExp 实现，数值稳定"""

    def __init__(self, axes: Optional[Union[int, tuple[int]]] = None, keepdims: bool = False) -> None:
        self.axes = axes
        self.keepdims = keepdims

    def compute(self, Z: NDArray) -> NDArray:
        # 取最大值提升数值稳定性
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        sum_exp = array_api.sum(array_api.exp(Z - max_Z), axis=self.axes, keepdims=self.keepdims)

        out = array_api.log(sum_exp)
        # 把 max 加回来。如果 keepdims=False, max_Z也需要被squeeze/re-compute
        if not self.keepdims:
            # 重新计算不带 keepdims 的 max
            max_Z_for_add = array_api.max(Z, axis=self.axes, keepdims=False)
            out += max_Z_for_add
        else:
            # keepdims=True, max_Z 形状正确可以直接加
            out += max_Z
            
        return out

    def gradient(self, out_grad: Tensor, node: Tensor) -> TensorTuple:
        Z = node.inputs[0]

        # 1. 恢复 out_grad 被压缩的维度 (如果 keepdims=False)
        grad_reshaped = out_grad
        if not self.keepdims and self.axes is not None:
            new_shape = list(Z.shape)
            axes_to_squeeze = self.axes if isinstance(self.axes, tuple) else (self.axes,)
            # 将被压缩的维度大小设为1
            for axis in axes_to_squeeze:
                new_shape[axis] = 1
            
            # 找到 out_grad 中对应的非1维度，填充到 new_shape
            out_grad_shape_iter = iter(out_grad.shape)
            final_shape = [next(out_grad_shape_iter) if s != 1 else 1 for s in new_shape]
            
            grad_reshaped = reshape(out_grad, tuple(final_shape))

        # 2. 广播梯度到 Z 的形状
        grad_bcast = broadcast_to(grad_reshaped, Z.shape)

        # 3. 完全用 needle.ops 从 Z 计算 Softmax(Z)
        #    Softmax(Z) = exp(Z - max(Z)) / sum(exp(Z-max(Z)))
        #    注意：这里的 max, summation, broadcast_to 都是 needle.ops
        max_Z = max(Z, axes=self.axes, keepdims=True)
        max_Z_bcast = broadcast_to(max_Z, Z.shape)
        Z_shifted = Z - max_Z_bcast
        
        exp_Z_shifted = exp(Z_shifted)
        
        sum_exp_Z = summation(exp_Z_shifted, axes=self.axes, keepdims=True)
        sum_exp_Z_bcast = broadcast_to(sum_exp_Z, Z.shape)
        
        softmax_Z = exp_Z_shifted / sum_exp_Z_bcast

        return (grad_bcast * softmax_Z,)



def logsumexp(a: Tensor, axes: Optional[Union[int, tuple[int]]] = None, keepdims: bool = False) -> Tensor:
    return LogSumExp(axes=axes, keepdims=keepdims)(a)
