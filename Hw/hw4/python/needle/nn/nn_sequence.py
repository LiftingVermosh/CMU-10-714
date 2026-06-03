"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        exp_x = ops.exp(-x)
        return (1 + exp_x) ** (-1)

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        k = 1 / (hidden_size ** 0.5)
        self.hidden_size = hidden_size

        W_ih_shape   = [input_size, hidden_size]
        W_hh_shape   = [hidden_size, hidden_size]
        bias_ih_size = [hidden_size, ] if bias else None
        bias_hh_size = [hidden_size, ] if bias else None

        self.W_ih = Parameter(init.rand(*W_ih_shape, low=-k, high=k, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(*W_hh_shape, low=-k, high=k, device=device, dtype=dtype))
        self.bias_ih = Parameter(init.rand(*bias_ih_size, low=-k, high=k, device=device, dtype=dtype)) if bias else None
        self.bias_hh = Parameter(init.rand(*bias_hh_size, low=-k, high=k, device=device, dtype=dtype)) if bias else None
        self.nonlinearity = ops.tanh if nonlinearity == 'tanh' else ops.relu

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        if h is None:
            h = init.zeros(*(X.shape[0], self.hidden_size), device=X.device, dtype=X.dtype)
    
        out = X @ self.W_ih + h @ self.W_hh

        if self.bias_ih is not None:
            out = out + self.bias_ih.broadcast_to(out.shape)
        if self.bias_hh is not None:
            out = out + self.bias_hh.broadcast_to(out.shape)
             
        return self.nonlinearity(out)


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.hidden_size = hidden_size

        self.device = device
        self.dtype  = dtype

        self.rnn_cells = []
        self.rnn_cells.append(
            RNNCell(
                input_size=input_size, 
                hidden_size=hidden_size, 
                bias=bias,
                nonlinearity=nonlinearity,
                device=device,
                dtype=dtype
            )
        )
        for i in range(num_layers - 1):
            cur_cells = RNNCell(
                input_size=hidden_size,
                hidden_size=hidden_size,
                bias=bias,
                nonlinearity=nonlinearity,
                device=device,
                dtype=dtype
            )
            self.rnn_cells.append(cur_cells)

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        seq_len, bs, _ = X.shape
        
        if h0 is None:
            h0 = init.zeros(*(self.num_layers, bs, self.hidden_size), device=self.device, dtype=self.dtype)
        
        # 拆分 h0
        h_layer_states = ops.split(h0, axis=0) # 得到 num_layers 个 (bs, hs)
        
        current_layer_input = ops.split(X, axis=0) # 得到 seq_len 个 (bs, input_size)
        
        final_h_n = [] # 记录每层最后一个时步的 h_t
        
        for l in range(self.num_layers):
            cell = self.rnn_cells[l]
            h_t = h_layer_states[l]
            layer_outputs = []
            for t in range(seq_len):
                h_t = cell(current_layer_input[t], h_t)
                layer_outputs.append(h_t)
            
            current_layer_input = layer_outputs # 当前层的输出序列作为下一层的输入
            final_h_n.append(h_t)
            
        # 组装返回
        output = ops.stack(current_layer_input, axis=0) # (seq_len, bs, hs)
        h_n = ops.stack(final_h_n, axis=0)             # (num_layers, bs, hs)
        return output, h_n

class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        k = 1 / (hidden_size ** 0.5)
        W_ih_shape   = [input_size, 4 * hidden_size]
        W_hh_shape   = [hidden_size, 4 * hidden_size]
        bias_ih_size = [4 * hidden_size, ] if bias else None
        bias_hh_size = [4 * hidden_size, ] if bias else None

        self.W_ih = Parameter(init.rand(*W_ih_shape, low=-k, high=k, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(*W_hh_shape, low=-k, high=k, device=device, dtype=dtype))
        self.bias_ih = Parameter(init.rand(*bias_ih_size, low=-k, high=k, device=device, dtype=dtype)) if bias else None
        self.bias_hh = Parameter(init.rand(*bias_hh_size, low=-k, high=k, device=device, dtype=dtype)) if bias else None

        self.hidden_size = hidden_size

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        if h is None:
            h0 = init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype=X.dtype)
            c0 = init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype=X.dtype)
        else:
            h0, c0 = h
        # 计算线性部分：X * W_ih + h * W_hh + biases
        gates = X @ self.W_ih + h0 @ self.W_hh
        if self.bias_ih is not None:
            gates = gates + self.bias_ih.reshape((1, 4 * self.hidden_size)).broadcast_to(gates.shape)
        if self.bias_hh is not None:
            gates = gates + self.bias_hh.reshape((1, 4 * self.hidden_size)).broadcast_to(gates.shape)
        # 利用 reshape + split 拆分四个门
        bs = X.shape[0]
        gates = gates.reshape((bs, 4, self.hidden_size))
        i, f, g, o = ops.split(gates, axis=1)
        # 应用激活函数
        i = Sigmoid()(i)
        f = Sigmoid()(f)
        g = ops.tanh(g)
        o = Sigmoid()(o)
        # 状态更新
        c_next = f * c0 + i * g
        h_next = o * ops.tanh(c_next)
        return h_next, c_next


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm_cells = []
        
        # 构建每一层的 Cell
        for l in range(num_layers):
            cur_input_size = input_size if l == 0 else hidden_size
            self.lstm_cells.append(
                LSTMCell(cur_input_size, hidden_size, bias=bias, device=device, dtype=dtype)
            )

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        seq_len, bs, _ = X.shape
        
        # 初始化 h_0, c_0 
        if h is None:
            h_init = init.zeros(self.num_layers, bs, self.hidden_size, device=X.device, dtype=X.dtype)
            c_init = init.zeros(self.num_layers, bs, self.hidden_size, device=X.device, dtype=X.dtype)
            h = (h_init, c_init)
        
        h_in, c_in = h
        # 在层维度拆分初始状态
        h_layers = list(ops.split(h_in, axis=0))
        c_layers = list(ops.split(c_in, axis=0))
        
        # 在时间维度拆分输入序列
        curr_layer_input = ops.split(X, axis=0) # 长度为 seq_len 的 list
        
        last_layer_h_all_t = [] # 用于存储最后一层的所有时步输出

        # 逐层处理
        for l in range(self.num_layers):
            cell = self.lstm_cells[l]
            h_t = h_layers[l]
            c_t = c_layers[l]
            
            next_layer_input = []
            for t in range(seq_len):
                # 处理当前时步
                h_t, c_t = cell(curr_layer_input[t], (h_t, c_t))
                next_layer_input.append(h_t)
            
            # 更新该层最终的状态记录
            h_layers[l] = h_t
            c_layers[l] = c_t
            # 下一层输入即为本层的输出序列
            curr_layer_input = next_layer_input
        
        # 组装最后一层的输出序列 (seq_len, bs, hidden_size)
        output = ops.stack(curr_layer_input, axis=0)
        
        # 组装最后时步的各层隐藏状态 (num_layers, bs, hidden_size)
        h_n = ops.stack(h_layers, axis=0)
        c_n = ops.stack(c_layers, axis=0)
        
        return output, (h_n, c_n)

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim

        self.weight = Parameter(init.randn(*(num_embeddings, embedding_dim), 0, 1, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        x_one_hot = init.one_hot(self.num_embeddings, x, device=x.device, dtype=x.dtype)
        return x_one_hot @ self.weight