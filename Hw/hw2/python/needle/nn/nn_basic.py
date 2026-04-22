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
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).reshape((1, out_features)))
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:

        H = X @ self.weight

        if self.bias is not None:
            B = self.bias.broadcast_to(H.shape)

        return H + B if self.bias is not None else H


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        return X.reshape((X.shape[0], -1))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        x_data = x.realize_cached_data()
        mask = (x_data > 0)
        mask = Tensor(mask, device=x.device, dtype=x.dtype)
        return mask * x

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        Z = logits
        batch_size = Z.shape[0]
        y_one_hot = init.one_hot(Z.shape[1], y, device=Z.device, dtype=Z.dtype)
        log_sum_Z = ops.logsumexp(Z, axes=(1,))
        log_sum_Z = log_sum_Z.reshape((batch_size,))

        Z_y = Z * y_one_hot
        Z_y = ops.summation(Z_y, axes=(1,))

        loss = ops.summation(-Z_y + log_sum_Z)
        return loss / batch_size

class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        
        self.running_mean = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_var = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        if self.training:
            original_mean = ops.summation(x, axes=(0,)) / B
            original_var = ops.summation((x - original_mean.reshape((1, -1)).broadcast_to(x.shape)) ** 2, axes=(0,)) / B 

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * original_mean.reshape((self.dim,)) # Ensure it stays (dim,)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * original_var.reshape((self.dim,)) # Ensure it stays (dim,)
            
            mean_broadcasted = original_mean.reshape((1, -1)).broadcast_to(x.shape)
            var_broadcasted = original_var.reshape((1, -1)).broadcast_to(x.shape)

            x_hat = (x - mean_broadcasted) / (var_broadcasted + self.eps) ** 0.5
        else:
            mean_broadcasted = self.running_mean.reshape((1, -1)).broadcast_to(x.shape)
            var_broadcasted = self.running_var.reshape((1, -1)).broadcast_to(x.shape)

            x_hat = (x - mean_broadcasted) / (var_broadcasted + self.eps) ** 0.5

        assert x_hat.shape == x.shape, f'Excepted {x.shape}, got {x_hat.shape}'
        y = self.weight.reshape((1, -1)).broadcast_to(x.shape) * x_hat + self.bias.reshape((1, -1)).broadcast_to(x.shape)
        return y



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        N = x.shape[1]
        E_x = ops.summation(x, axes=(1,)) / N
        E_x = E_x.reshape((x.shape[0], 1)).broadcast_to(x.shape)

        diff = x - E_x

        Var_x = ops.summation(diff ** 2, axes=(1,)) / N
        Var_x = Var_x.reshape((x.shape[0], 1)).broadcast_to(x.shape)

        x_hat = diff / (Var_x + self.eps) ** 0.5

        return x_hat * self.weight.reshape((1, self.dim)).broadcast_to(x.shape) + self.bias.reshape((1, self.dim)).broadcast_to(x.shape)


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        assert 0 <= self.p <= 1, "p must be between 0 and 1"
        if self.training:
            mask_data = init.rand(*x.shape, device=x.device, dtype=x.dtype).realize_cached_data()
            mask_data = (mask_data < self.p)
            mask = Tensor(mask_data, device=x.device, dtype=x.dtype)
            return mask * x / (1 - self.p)
        else:
            return x


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        x_res = self.fn(x)
        return x + x_res
