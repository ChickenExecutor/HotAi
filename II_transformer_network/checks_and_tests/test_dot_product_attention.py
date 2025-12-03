import numpy as np
from numpy import random
import torch

from model import dot_product_attention


def test_dot_product_attention():
    sequence_length = 24
    dim_k = 32
    dim_q = 32
    dim_v = 16
    batch_size = 8

    queries = random.random((batch_size, 1, sequence_length, dim_q))
    keys = random.random((batch_size, 1, sequence_length, dim_k))
    values = random.random((batch_size, 1, sequence_length, dim_v))

    mask = torch.zeros((batch_size, 1, 1, sequence_length))

    attention = dot_product_attention.DotProductAttention()
    res = attention(torch.Tensor(queries), torch.Tensor(keys), torch.Tensor(values), mask)
    assert type(res) is torch.Tensor
    assert res.shape == (batch_size, 1, sequence_length, dim_v)

    mask = torch.ones((batch_size, 1, 1, sequence_length))
    for i in range(batch_size):
        mask[i, 0, :, i % sequence_length] = 0

    attention = dot_product_attention.DotProductAttention()
    res = attention(torch.Tensor(queries), torch.Tensor(keys), torch.Tensor(values), mask)
    assert type(res) is torch.Tensor
    assert res.shape == (batch_size, 1, sequence_length, dim_v)
    for b in range(batch_size):
        assert np.linalg.norm(abs(res[b, 0, b % sequence_length, :] - values[b, 0, b % sequence_length, :])) < 1e-3
