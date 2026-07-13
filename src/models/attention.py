import math
import torch
import torch.nn as nn


class NaiveMultiHeadAttentionBatched(nn.Module):
    def __init__(self, W, bias, Wo, bias_o, d_model, heads):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d = d_model // heads

        self.W = nn.Parameter(W)
        self.bias = nn.Parameter(bias)
        self.Wo = nn.Parameter(Wo)
        self.bias_o = nn.Parameter(bias_o)

    def forward(self, X):
        B, N, _ = X.shape

        W_q = self.W[:, :self.d_model]
        W_k = self.W[:, self.d_model:2*self.d_model]
        W_v = self.W[:, 2*self.d_model:]

        b_q = self.bias[:self.d_model]
        b_k = self.bias[self.d_model:2*self.d_model]
        b_v = self.bias[2*self.d_model:]

        Q = torch.matmul(X, W_q) + b_q
        K = torch.matmul(X, W_k) + b_k
        V = torch.matmul(X, W_v) + b_v

        Q = Q.view(B, N, self.heads, self.d).transpose(1, 2)
        K = K.view(B, N, self.heads, self.d).transpose(1, 2)
        V = V.view(B, N, self.heads, self.d).transpose(1, 2)

        P = torch.matmul(Q, K.transpose(-2, -1))
        S = torch.softmax(P / math.sqrt(self.d), dim=-1)
        O = torch.matmul(S, V)

        O = O.transpose(1, 2).contiguous().view(B, N, self.d_model)

        output = torch.matmul(O, self.Wo) + self.bias_o
        return output
