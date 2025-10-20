"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    """ 线性层 """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 权重
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype, requires_grad=True))
        # 偏置
        if bias:
            fan_in_bias = out_features
            
            bias_tensor = init.kaiming_uniform(fan_in_bias, 1, device=device, dtype=dtype, requires_grad=True)
            
            bias_tensor_reshaped = bias_tensor.reshape((1, out_features))
            
            self.bias = Parameter(bias_tensor_reshaped)
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        # 前向传播，线性层的前向传播为其权重与输入的乘积 + 偏置
        out = X @ self.weight
        if self.bias is not None:
            out += self.bias
        return out


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        return X.reshape([X.shape[0], -1])

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.ReLU()(x)

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        # 顺序遍历调用 forward 方法即可
        for module in self.modules:
            x = module.forward(x)
        return x



# nn_basic.py -> class SoftmaxLoss (修正后)
class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        y_one_hot = init.one_hot(logits.shape[1], y, device=logits.device, dtype=logits.dtype)
        lse = ops.logsumexp(logits, axes=(1,), keepdims=True) 
        z_minus_lse = logits - lse.broadcast_to(logits.shape)
        selected_log_probs = z_minus_lse * y_one_hot
        total_loss = -ops.summation(selected_log_probs)
        return total_loss / logits.shape[0]



class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype), requires_grad=True)
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype), requires_grad=True)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        """ 
        """
        if not self.training:
            # 预测时，直接使用 running_mean 和 running_var 进行归一化
            mean = self.running_mean.reshape((1, self.dim))
            variance = self.running_var.reshape((1, self.dim))

            std = ops.sqrt(variance + self.eps)
            std_broadcasted = std.broadcast_to(x.shape)
            mean_broadcasted = mean.broadcast_to(x.shape)

            x_hat = (x - mean_broadcasted) / std_broadcasted

            weight_broadcasted = self.weight.broadcast_to(x.shape)
            bias_broadcasted = self.bias.broadcast_to(x.shape)

            return x_hat * weight_broadcasted + bias_broadcasted
        else:
            # 训练时，更新 running_mean 和 running_var，并进行归一化
            mean_x = ops.mean(x, axes=0, keepdims=True)
            mean_x_broadcasted = mean_x.broadcast_to(x.shape)
            
            # x - mean_x
            x_minus_mean = x - mean_x_broadcasted
            
            # variance_x 形状 (batch_size, 1)
            variance_x = ops.mean(x_minus_mean ** 2, axes=0, keepdims=True)
            
            # std_x 形状 (batch_size, 1)，需要广播
            std_x = ops.sqrt(variance_x + self.eps)
            std_x_broadcasted = std_x.broadcast_to(x.shape)
            
            # 归一化
            x_hat = x_minus_mean / std_x_broadcasted
            
            # 应用 weight 和 bias
            weight_broadcasted = self.weight.broadcast_to(x.shape)
            bias_broadcasted = self.bias.broadcast_to(x.shape)

            # 更新 running_mean 和 running_var
            # 获取底层 numpy 数组进行计算
            current_mean_data = mean_x.realize_cached_data().reshape((self.dim,))
            current_var_data = variance_x.realize_cached_data().reshape((self.dim,))
            # 更新滑动平均值
            self.running_mean.data = Tensor((1 - self.momentum) * self.running_mean.realize_cached_data() + \
                                    self.momentum * current_mean_data)
            self.running_var.data = Tensor((1 - self.momentum) * self.running_var.realize_cached_data() + \
                                    self.momentum * current_var_data)
            
            return x_hat * weight_broadcasted + bias_broadcasted

class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        # weight 和 bias 初始化为 (1, dim)，方便广播
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype), requires_grad=True)
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # Op: ops.mean -> Mean
        mean_x = ops.mean(x, axes=1, keepdims=True)
        
        # Op: x - mean_x -> EWiseAdd, Negate 
        square_diff = (x - mean_x) ** 2
        
        # Op: ops.mean -> Mean
        variance_x = ops.mean(square_diff, axes=1, keepdims=True)
        
        # Op: variance_x + self.eps -> AddScalar
        # Op: ops.sqrt -> Sqrt 
        std_x = ops.sqrt(variance_x + self.eps)
        
        # Op: (x - mean_x) / std_x -> EWiseDiv 
        x_hat = (x - mean_x) / std_x

        weight_broadcasted = self.weight.broadcast_to(x.shape)
        bias_broadcasted = self.bias.broadcast_to(x.shape)
        return x_hat * weight_broadcasted + bias_broadcasted
    
class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        
        keep_prob = 1 - self.p
        # 随机生成 mask
        mask_data = (np.random.rand(*x.shape) < keep_prob).astype(x.dtype)
        # 注意此处将 mask_data 数组包装成一个不需要梯度的 Tensor（叶子节点
        mask = Tensor(mask_data, device=x.device, dtype=x.dtype, requires_grad=False)
        if keep_prob > 0 :
            return x * mask / (1 - self.p)
        else:
            return x * 0



class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)
