"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        
        for param in self.params:
                
            if param.grad is not None:
                dtype = param.data.dtype
                device = param.data.device
                lr_t = ndl.Tensor(self.lr, dtype=dtype, device=device)
                weight_decay_t = ndl.Tensor(self.weight_decay, dtype=dtype, device=device)
                momentum_t = ndl.Tensor(self.momentum, dtype=dtype, device=device)
                one_t = ndl.Tensor(1, dtype=dtype, device=device)
            
                grad_with_decay = ndl.Tensor(param.grad, dtype=dtype, device=device)


                if self.weight_decay != 0.0:
                    grad_with_decay.data = grad_with_decay.data + weight_decay_t.data * param.data

                if param not in self.u:
                    self.u[param] = ndl.zeros_like(param.data, requires_grad=False)

                self.u[param].data = (momentum_t.data * self.u[param].data + (one_t.data - momentum_t.data) * grad_with_decay.data)

                param.data = (param.data - lr_t.data * self.u[param].data)

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for param in self.params:
            if param.grad is not None:
                dtype = param.data.dtype
                device = param.data.device
                lr_t = ndl.Tensor(self.lr, dtype=dtype, device=device)
                beta1_t = ndl.Tensor(self.beta1, dtype=dtype, device=device)
                beta2_t = ndl.Tensor(self.beta2, dtype=dtype, device=device)
                eps_t = ndl.Tensor(self.eps, dtype=dtype, device=device)
                weight_decay_t = ndl.Tensor(self.weight_decay, dtype=dtype, device=device)
                one_t = ndl.Tensor(1, dtype=dtype, device=device)
                t_t = ndl.Tensor(self.t, dtype=dtype, device=device)

                # p_grad = ndl.Tensor(param.grad, dtype=dtype, device=device) # Mem Leak
                p_grad = ndl.Tensor(param.grad, dtype=dtype, device=device).detach()

                if self.weight_decay != 0.0:
                    p_grad.data = p_grad.data.detach() + weight_decay_t.data.detach() * param.data.detach()

                if param not in self.m:
                    self.m[param] = ndl.zeros_like(param.data.detach(), requires_grad=False)
                
                self.m[param].data = (beta1_t.data.detach() * self.m[param].data.detach() + (one_t.data.detach() - beta1_t.data.detach()) * p_grad)

                if param not in self.v:
                    self.v[param] = ndl.zeros_like(param.data.detach(), requires_grad=False)
                
                self.v[param].data = (beta2_t.data.detach() * self.v[param].data.detach() + (one_t.data.detach() - beta2_t.data.detach()) * (p_grad * p_grad))

                bias_m = self.m[param].data.detach() / (one_t.data.detach() - beta1_t.data.detach() ** t_t.data.detach())
                bias_v = self.v[param].data.detach() / (one_t.data.detach() - beta2_t.data.detach() ** t_t.data.detach())

                assert bias_m.dtype == param.data.detach().dtype

                param.data = param.data.detach() - lr_t.data.detach() * bias_m / (bias_v.data.detach() ** 0.5 + eps_t.data.detach())