import torch

from model.multi_head_attention import MultiHeadAttention


def test_multihead_attention():
    heads = 8
    sequence_length = 9
    dim_k = 32
    dim_q = 32
    dim_v = 16
    dim_model = 64
    batch_size = 7

    multihead_attention = MultiHeadAttention(heads, dim_q, dim_k, dim_v, dim_model)
    queries = torch.randn((batch_size, sequence_length, dim_model))
    keys = torch.randn((batch_size, sequence_length, dim_model))
    values = torch.randn((batch_size, sequence_length, dim_model))

    output = multihead_attention(queries, keys, values)
    assert type(output) is torch.Tensor
    assert output.shape == (batch_size, sequence_length, dim_model)

    mask = torch.zeros((batch_size, 1, sequence_length))
    output = multihead_attention(queries, keys, values, mask=mask)
    assert type(output) is torch.Tensor
    assert output.shape == (batch_size, sequence_length, dim_model)

    mask = torch.ones((batch_size, 1, sequence_length))
    mask[:, 0] = 0
    output = multihead_attention(queries, keys, values, mask=mask)
    assert type(output) is torch.Tensor
    assert output.shape == (batch_size, sequence_length, dim_model)