from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        Z_max = array_api.max(Z, axis=1, keepdims=True)
        Z_exp = array_api.exp(Z - Z_max)
        Z_exp_sum = array_api.sum(Z_exp, axis=1, keepdims=True) 
        Z_log_sum_exp = array_api.log(Z_exp_sum) + Z_max
        return Z - Z_log_sum_exp

    def gradient(self, out_grad: Tensor, node: Tensor):
        axes = (-1,) 
        
        Y = node 
        S = exp(Y) 
        
        G_sum_val = summation(out_grad, axes=axes)
        
        input_shape = out_grad.shape
        ndim = len(input_shape)
        
        # 计算 target_shape
        target_shape = list(input_shape)
        for axis in axes:
            target_shape[axis] = 1 
        target_shape = tuple(target_shape)
        
        G_sum_reshape = G_sum_val.reshape(target_shape)
        G_sum_broadcast = G_sum_reshape.broadcast_to(input_shape)
        
        # 计算最终梯度
        grad_Z = out_grad - G_sum_broadcast * S
        return (grad_Z,)


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        Z_max = Z.max(axis=self.axes, keepdims=True)
        Z_max_broadcast = Z_max.broadcast_to(Z.shape)
        
        Z_exp = (Z - Z_max_broadcast).exp() 
        Z_exp_sum = Z_exp.sum(axis=self.axes, keepdims=True)
        
        # 同样，最后的 Z_max 需要广播
        res = Z_exp_sum.log() + Z_max
        
        # 注意：Needle 的 compute 必须返回 NDArray，不能返回 .item()
        if self.axes is None:
            return res.reshape((1,))
        out_shape = tuple([s for i, s in enumerate(Z.shape) 
                          if self.axes is not None and i not in (self.axes if isinstance(self.axes, tuple) else (self.axes,))])
        if not out_shape: out_shape = (1,)
        return res.reshape(out_shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        Z = node.inputs[0] 
        Y = node           

        input_shape = Z.shape
        ndim = len(input_shape)
        
        # 被求和的轴
        axes = self.axes
        if axes is None:
            axes_to_reduce = tuple(range(ndim))
        elif isinstance(axes, int):
            axes_to_reduce = (axes,)
        else:
            axes_to_reduce = axes
            
        target_shape = list(input_shape)
        for axis in axes_to_reduce:
            if axis < len(target_shape):
                target_shape[axis] = 1
        target_shape = tuple(target_shape) 
        
        # LSE(Z) 扩展后的形状
        Y_reshape = Y.reshape(target_shape) 
        Y_broadcast = Y_reshape.broadcast_to(input_shape) 
        
        softmax_S = exp(Z - Y_broadcast) 
        
        # 外部梯度 G 扩展后的形状
        reshaped_out_grad = out_grad.reshape(target_shape)
        out_grad_broadcast = reshaped_out_grad.broadcast_to(input_shape)
        
        # 计算最终梯度
        grad_Z = out_grad_broadcast * softmax_S

        return (grad_Z,)


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)