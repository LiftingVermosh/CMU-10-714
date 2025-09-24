import math
from .init_basic import *
from typing import Any


def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    # 计算 a 值
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    # 在 [-a, a] 间随机采样以填充张量
    return rand(fan_in, fan_out, low=-a, high=a, device=kwargs.get("device"), dtype=kwargs.get("dtype"), requires_grad=kwargs.get("requires_grad"))



def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    # 计算 std 值
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    # 在 N(0, a^2) 间随机采样以填充张量
    return randn(fan_in, fan_out,mean=0.0, std=std, device=kwargs.get("device"), dtype=kwargs.get("dtype"), requires_grad=kwargs.get("requires_grad"))

def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    # 确认 gain 值
    if nonlinearity == 'relu':
        gain = math.sqrt(2.0)
    
    # 计算 bound 值
    bound = gain * math.sqrt(3.0 / fan_in)
    # 在 [-bound, bound] 间随机采样以填充张量
    return rand(fan_in, fan_out, low=-bound, high=bound, device=kwargs.get("device"), dtype=kwargs.get("dtype"), requires_grad=kwargs.get("requires_grad"))


def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    # 确认 gain 值
    if nonlinearity == 'relu':
        gain = math.sqrt(2.0)
    
    # 计算 std 值
    std = gain / math.sqrt(fan_in)
    # 在 N(0, std^2) 间随机采样以填充张量
    return randn(fan_in, fan_out, mean=0.0, std=std, device=kwargs.get("device"), dtype=kwargs.get("dtype"), requires_grad=kwargs.get("requires_grad"))