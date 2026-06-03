import math
from .....tmp.python.needle.init.init_basic import *
from typing import Any

def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, shape = None, **kwargs: Any) -> "Tensor":

    out_shape = shape if shape is not None else (fan_in, fan_out)
    if shape is not None:
        K, _, C_in, C_out = shape  
        fan_in  = K * K * C_in
        fan_out = K * K * C_out
        
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return rand(*out_shape, low=-a, high=a, **kwargs)

def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, shape = None, **kwargs: Any) -> "Tensor":
    out_shape = shape if shape is not None else (fan_in, fan_out)
    if shape is not None:
        K, _, C_in, C_out = shape  
        fan_in  = K * K * C_in
        fan_out = K * K * C_out
        

    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return randn(*out_shape, mean=0.0, std=std, **kwargs)

def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    out_shape = shape if shape is not None else (fan_in, fan_out)
    
    if shape is not None:
        K, _, C_in, C_out = shape  
        fan_in  = K * K * C_in
        fan_out = K * K * C_out
        
    gain = math.sqrt(2.0)
    bound = gain * math.sqrt(3.0 / fan_in)

    return rand(*out_shape, low=-bound, high=bound, **kwargs)

def kaiming_normal(fan_in: int, fan_out: int, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    
    out_shape = shape if shape is not None else (fan_in, fan_out)
    if shape is not None:
        K, _, C_in, C_out = shape  
        fan_in  = K * K * C_in
        fan_out = K * K * C_out
        
    gain = math.sqrt(2.0)
    std = gain / math.sqrt(fan_in)
    
    return randn(*out_shape, mean=0.0, std=std, **kwargs)