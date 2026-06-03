"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.padding = (self.kernel_size - 1) // 2

        weight_shape = [kernel_size, kernel_size, in_channels, out_channels] 
        bias_shape   = [out_channels, ] if bias else None
        
        self.weight = Parameter(init.kaiming_uniform(fan_in=None, fan_out=None, shape=weight_shape, device=device, dtype=dtype))

        if bias:
            bound = 1 / (in_channels * kernel_size * kernel_size) ** 0.5
            self.bias   = Parameter(init.rand(*bias_shape, low=-bound, high=bound, device=device, dtype=dtype)) if bias else None


    def forward(self, x: Tensor) -> Tensor:
        
        # BCHW \to BHWC
        x = x.transpose([1, 2])
        x = x.transpose([2, 3])

        # Conv
        x = ops.conv(x, self.weight, self.stride, self.padding)

        if self.bias is not None:
            bias = self.bias.broadcast_to(x.shape)
            x += bias
        
        # inverse T
        x = x.transpose([2, 3])
        x = x.transpose([1, 2])

        return x




